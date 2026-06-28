"""
model_def.py — model architecture definition, copied exactly from your
training notebook (trainingLSTM_NASDAQ_5D_v3.ipynb, cell defining
StockPctChangeBiLSTMAttention).

diagnostic.py imports this to instantiate the model and load your saved
state_dict. If you change the architecture in your training notebook,
update this file to match — they must be identical or torch.load_state_dict
will fail or silently mismatch.
"""

import torch
import torch.nn as nn


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
        self.fc           = nn.Linear(hidden_size * 2, 1)
        self.skip_fc      = nn.Linear(hidden_size * 2, 1)
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