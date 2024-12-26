import math

import torch
from torch import nn as nn


class DigitTransformer(nn.Module):
    def __init__(
            self,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            max_length=7,
            vocab_size=12,  # digits(0..9)=10, PAD=10, MINUS=11 => total 12
            num_classes=2
    ):
        """
        A miniature Transformer model for classifying sequences of digit tokens (including minus sign).
        """
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        # Embedding for digits + PAD + MINUS
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=10)  # 10 = PAD

        # Basic sinusoidal positional encoding
        self.pos_encoding = nn.Parameter(self._init_pos_encoding(), requires_grad=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def _init_pos_encoding(self):
        """
        Initialize sinusoidal positional encoding:
        shape => (1, max_length, d_model)
        """
        pos_enc = torch.zeros(self.max_length, self.d_model)
        for pos in range(self.max_length):
            for i in range(0, self.d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / self.d_model)))
        return pos_enc.unsqueeze(0)  # shape: (1, max_length, d_model)

    def forward(self, x):
        """
        x: (batch_size, sequence_length) containing digit/minus tokens
        """
        # Embedding
        emb = self.embedding(x)
        # Add positional encoding
        emb = emb + self.pos_encoding[:, :emb.size(1), :]

        # Transformer Encoder
        out = self.transformer_encoder(emb)  # (B, L, d_model)

        # Pool across sequence dimension (mean pool)
        pooled = out.mean(dim=1)  # (B, d_model)

        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
