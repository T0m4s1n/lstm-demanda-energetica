from __future__ import annotations

import torch
from torch import nn


class BaseLSTM(nn.Module):
    name = "lstm base"
    key = "lstm_base"

    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _sequence, (state, _cells) = self.lstm(x)
        last = self.dropout(state[-1])
        return self.output_layer(last).squeeze(-1)


class DeepLSTM(nn.Module):
    name = "lstm profunda"
    key = "lstm_deep"

    def __init__(self, n_features: int):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.drop3 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm1(x)
        sequence = self.drop1(sequence)
        sequence, _ = self.lstm2(sequence)
        sequence = self.drop2(sequence)
        _sequence, (state, _cells) = self.lstm3(sequence)
        last = self.drop3(state[-1])
        hidden = torch.relu(self.fc(last))
        return self.output_layer(hidden).squeeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(torch.tanh(self.projection(sequence))).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class AttentionLSTM(nn.Module):
    name = "lstm con atención"
    key = "lstm_attention"

    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 96, batch_first=True)
        self.sequence_dropout = nn.Dropout(0.3)
        self.attention = TemporalAttention(96)
        self.fc = nn.Linear(192, 32)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, (state, _cells) = self.lstm(x)
        sequence = self.sequence_dropout(sequence)
        context, _weights = self.attention(sequence)
        joined = torch.cat([context, state[-1]], dim=1)
        hidden = self.dropout(torch.relu(self.fc(joined)))
        return self.output_layer(hidden).squeeze(-1)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        _context, weights = self.attention(sequence)
        return weights


ARCHITECTURES = {
    BaseLSTM.key: BaseLSTM,
    DeepLSTM.key: DeepLSTM,
    AttentionLSTM.key: AttentionLSTM,
}


def create_model(key: str, n_features: int) -> nn.Module:
    try:
        model_cls = ARCHITECTURES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown architecture: {key}") from exc
    return model_cls(n_features)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
