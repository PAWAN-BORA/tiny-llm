import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, embedding_dim:int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.position_embedding  = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        _batch_size, sequence_length = x.shape
        token_embeddings = self.token_embedding(x)
        positions = torch.arange(sequence_length, device=x.device)
        position_embedding = self.position_embedding(positions)

        return token_embeddings + position_embedding;
        # return self.embedding(x);



    
