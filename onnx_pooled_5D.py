"""
Convert the POOLED 5d model (trainingLSTM_NASDAQ_5D_v3.ipynb) to ONNX.

One model for all symbols. Input = 27 stationary features + N symbol one-hot
(N = number of pooled symbols), so input_size is e.g. 34 for 7 symbols.
Paths differ from the per-symbol converter: models/pooled_5d/.
"""
import os
import json
import torch
import torch.nn as nn
import onnx

POOLED_DIR  = "models/pooled_5d"
PT_PATH     = f"{POOLED_DIR}/pooled_best_bilstm.pt"
PARAMS_PATH = f"{POOLED_DIR}/pooled_best_params.json"
ONNX_PATH   = f"{POOLED_DIR}/pooled_bilstm.onnx"


# ── Model (must match v3 training notebook) ───────────────────────────────────
class StockPctChangeBiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm         = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, bidirectional=True,
                                    dropout=dropout if num_layers > 1 else 0)
        self.attention    = nn.Linear(hidden_size * 2, 1)
        self.pos_bias     = nn.Parameter(torch.zeros(1))
        self.ln           = nn.LayerNorm(hidden_size * 2)
        self.dropout      = nn.Dropout(dropout)
        self.fc           = nn.Linear(hidden_size * 2, output_size)
        self.skip_fc      = nn.Linear(hidden_size * 2, output_size)
        self.output_blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        lstm_out, _  = self.lstm(x)
        seq_len      = lstm_out.size(1)
        pos_weights  = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(-1)
        attn_raw     = self.attention(lstm_out) + self.pos_bias * pos_weights
        attn_weights = torch.softmax(attn_raw, dim=1)
        context      = (attn_weights * lstm_out).sum(dim=1)
        out_attn     = self.fc(self.dropout(self.ln(context)))
        last_hidden  = lstm_out[:, -1, :]
        out_skip     = self.skip_fc(last_hidden)
        alpha        = torch.sigmoid(self.output_blend)
        return alpha * out_attn + (1 - alpha) * out_skip


def infer_hyperparams(state_dict: dict) -> dict:
    """Authoritative from checkpoint shapes; cross-checked vs pooled_best_params.json."""
    ih_l0       = state_dict["lstm.weight_ih_l0"]
    hidden_size = ih_l0.shape[0] // 4
    input_size  = ih_l0.shape[1]                       # 27 feats + N symbol one-hot
    num_layers  = sum(1 for k in state_dict
                      if k.startswith("lstm.weight_ih_l") and "_reverse" not in k)
    output_size = state_dict["fc.weight"].shape[0]

    hp = {"input_size": input_size, "hidden_size": hidden_size,
          "num_layers": num_layers, "output_size": output_size,
          "seq_len": None, "json_verified": False}

    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            bp = json.load(f).get("best_params", {})
        for key in ("hidden_size", "num_layers"):
            if key in bp and bp[key] != hp[key]:
                raise ValueError(
                    f"{key} mismatch: checkpoint={hp[key]} vs params.json={bp[key]} "
                    f"-- stale .pt or JSON, refusing to convert."
                )
        hp["seq_len"]       = bp.get("seq_len")
        hp["json_verified"] = True
    else:
        print(f"note: {PARAMS_PATH} not found -> using checkpoint shapes only")
    return hp


def convert():
    if not os.path.exists(PT_PATH):
        print(f"[SKIP] not found -> {PT_PATH}")
        return

    print(f"Loading {PT_PATH} ...")
    state_dict = torch.load(PT_PATH, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    hp = infer_hyperparams(state_dict)
    print(f"params -> input_size={hp['input_size']} (feats + symbol one-hot), "
          f"hidden_size={hp['hidden_size']}, num_layers={hp['num_layers']}, "
          f"output_size={hp['output_size']}, seq_len={hp['seq_len']}, "
          f"json_verified={hp['json_verified']}")

    model = StockPctChangeBiLSTMAttention(
        input_size  = hp["input_size"],
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        output_size = hp["output_size"],
    )
    model.load_state_dict(state_dict)   # raises on any shape mismatch (extra safety)
    model.eval()

    seq = hp["seq_len"] or 20           # dynamic axis anyway; real seq_len for the trace
    dummy_input = torch.randn(1, seq, hp["input_size"])

    os.makedirs(POOLED_DIR, exist_ok=True)
    torch.onnx.export(
        model, dummy_input, ONNX_PATH,
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"},
                      "output": {0: "batch_size"}},
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print(f"OK Saved & validated -> {ONNX_PATH}")
    print("NOTE: ONNX input must be [batch, seq_len, "
          f"{hp['input_size']}] = 27 stationary features + symbol one-hot in the SAME "
          "column order as training (POOL_SYMS order). Downstream inference must build that.")


if __name__ == "__main__":
    convert()