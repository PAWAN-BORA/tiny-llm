import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    mask: torch.Tensor
    def __init__(self, embedding_dim, context_length) -> None:
        super().__init__()

        self.qyery = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(context_length, context_length)
            )
        )

    def forward(self, x):
        Q = self.qyery(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)  
        scores = scores / math.sqrt(x.shape[-1])
        scores = scores.masked_fill(
            self.mask[:x.shape[1], :x.shape[1]]==0, 
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        print(attention_weights)
        output = attention_weights @ V

        return output
