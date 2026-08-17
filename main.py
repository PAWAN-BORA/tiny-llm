from dataset.text_dataset import TextDataset
from embedding.token_embedding import TokenEmbedding
from tokenizer.char_tokenizer import CharTokenizer


def main():
    # word = "Hello Word caf"

    text = """abcde"""
    tokenizer = CharTokenizer()
    tokens = tokenizer.encode(text=text);
    dataset = TextDataset(tokens=tokens, context_length=3)
    vocab_size = tokenizer.vocab_size;
    embedding_size = 8;
    embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_size)
    # print("Number of samples:", len(dataset))
    x, y = dataset[0]
    print("X:", x)
    print("X: shape", x.shape)
    # print("Y:", y)
    # print(embedding.embedding.weight)
    output = embedding(x)

    print("output:", output)
    print("output shape:", output.shape)

    print(embedding.embedding.weight.shape)
    print(embedding.embedding.weight)
    # print("X decoded:", tokenizer.decode(x.tolist()))
    # print("Y decoded:", tokenizer.decode(y.tolist()))
    # print(tokens)

    # encoded_word = toknizer.decode(tokens)
    # print(encoded_word)


if __name__ == "__main__":
    main()
