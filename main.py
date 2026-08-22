from data.data_loader import create_data_loader
from dataset.text_dataset import TextDataset
from embedding.token_embedding import TokenEmbedding
from tokenizer.char_tokenizer import CharTokenizer


def main():
    # word = "Hello Word caf"

    text = """English grammar is the set of structural rules of the English language. This includes the structure of words, phrases, clauses, sentences, and whole texts."""
    tokenizer = CharTokenizer()
    tokens = tokenizer.encode(text=text);
    context_length = 16
    # dataset = TextDataset(tokens=tokens, context_length=context_length)
    vocab_size = tokenizer.vocab_size;
    embedding_size = 8;
    embedding = TokenEmbedding(vocab_size=vocab_size, context_length=context_length, embedding_dim=embedding_size)
    loader = create_data_loader(
        tokens=tokens,
        context_length=16,
        batch_size=4,
    )
    x, y = next(iter(loader))
    print("X:", x)
    print("Y:", y)

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    # print("Number of samples:", len(dataset))
    # x, y = dataset[0]
    # print("X:", x)
    # print("X: shape", x.shape)
    # print("X: device", x.device)
    # print("Y:", y)
    # print(embedding.embedding.weight)
    # output = embedding(x)

    # print("output:", output)
    # print("output shape:", output.shape)

    # print(embedding.embedding.weight.shape)
    # print(embedding.embedding.weight)
    # print("X decoded:", tokenizer.decode(x.tolist()))
    # print("Y decoded:", tokenizer.decode(y.tolist()))
    # print(tokens)

    # encoded_word = toknizer.decode(tokens)
    # print(encoded_word)


if __name__ == "__main__":
    main()
