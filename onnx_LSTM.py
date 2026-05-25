import torch
import torch.nn as nn
import onnx
import os

symbols = ["8300", "1010", "8100", "2050", "2060", "1180", "3080", "3090"]

# ── Model Definition ──────────────────────────────────────────────────────────
class StockPctChangeBiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention    = nn.Linear(hidden_size * 2, 1)
        self.pos_bias     = nn.Parameter(torch.zeros(1))
        self.ln           = nn.LayerNorm(hidden_size * 2)
        self.dropout      = nn.Dropout(dropout)
        self.fc           = nn.Linear(hidden_size * 2, 2)
        self.skip_fc      = nn.Linear(hidden_size * 2, 2)
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


# ── Hyperparameters ───────────────────────────────────────────────────────────
# Adjust these to match what you used during training
INPUT_SIZE  = 14   # number of input features per timestep
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LEN     = 30   # sequence length used during training (for dummy input shape)
BATCH_SIZE  = 1    # ONNX export uses batch size 1; dynamic axes allow any size


def convert(symbol: str):
    pt_path   = f"models/{symbol}/{symbol}_best_bilstm.pt"
    onnx_dir  = f"models/{symbol}"
    onnx_path = f"{onnx_dir}/{symbol}_bilstm.onnx"

    # ── Resolve actual PT path ─────────────────────────────────────────────
    # The naming pattern in your question: symbol_name/symbol_name_8300_best_bilstm.pt
    # e.g. models/8300/8300_8300_best_bilstm.pt  — adjust if different
    candidates = [
        f"models/{symbol}/{symbol}_best_bilstm.pt",
    ]
    pt_path = None
    for c in candidates:
        if os.path.exists(c):
            pt_path = c
            break

    if pt_path is None:
        print(f"[SKIP] {symbol}: no .pt file found (tried {candidates})")
        return

    print(f"[{symbol}] Loading {pt_path} ...")
    model = StockPctChangeBiLSTMAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)

    checkpoint = torch.load(pt_path, map_location="cpu")

    # Support both raw state_dict and checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # ── Dummy input ────────────────────────────────────────────────────────
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

    os.makedirs(onnx_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # ── Validate ───────────────────────────────────────────────────────────
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"[{symbol}] ✓ Saved & validated → {onnx_path}")


# ── Run all conversions ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for sym in symbols:
        try:
            convert(sym)
        except Exception as e:
            print(f"[{sym}] ✗ Error: {e}")