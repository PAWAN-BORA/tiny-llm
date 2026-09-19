import torch
import torch.nn as nn

from embedding.attention import SelfAttention
from embedding.token_position_embedding import TokenPositionEmbedding

class Transformer(nn.Module):
    
    def __init__(self, vocab_size:int, context_length:int, embedding_dim:int) -> None:
        super().__init__()

        self.embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim
        )

        self.attention = SelfAttention(
            embedding_dim=embedding_dim,
            context_length=context_length

        )
        self.output = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)

        logits = self.output(x)

        return logits
