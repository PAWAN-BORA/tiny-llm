import torch
from embedding.attention import SelfAttention

def test_attention():
    embedding_dim = 4
    batch_size = 2
    sequence_length = 5

    attention = SelfAttention(embedding_dim)

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("x:", x)
    print("x.shape:", x.shape)

    output = attention(x)

    print("output:", output)
    print("output.shape:", output.shape)
