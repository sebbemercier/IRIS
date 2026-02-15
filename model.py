# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn

class IrisSLM(nn.Module):
    """
    SLM dédié à l'analyse de sentiment et détection d'anomalies.
    Architecture: Bi-LSTM avec Attention.
    """
    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.sentiment_head = nn.Linear(hidden_dim * 2, 3) # Négatif, Neutre, Positif
        self.anomaly_head = nn.Linear(hidden_dim * 2, 1)   # Score d'anomalie 0-1

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Mécanisme d'attention simple
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.sentiment_head(context), torch.sigmoid(self.anomaly_head(context))