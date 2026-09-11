from data.data_loader import create_data_loader
from dataset.text_dataset import TextDataset
from embedding.token_position_embedding import TokenPositionEmbedding
from tokenizer.char_tokenizer import CharTokenizer


def main():
    # word = "Hello Word caf"

    text = """English grammar is the set of structural rules of the English language. This includes the structure of words, phrases, clauses, sentences, and whole texts."""
    tokenizer = CharTokenizer()
    tokens = tokenizer.encode(text=text);
    print(tokens)
    context_length = 16
    # dataset = TextDataset(tokens=tokens, context_length=context_length)
    vocab_size = tokenizer.vocab_size;
    embedding_size = 8;
    embedding = TokenPositionEmbedding(vocab_size=vocab_size, context_length=context_length, embedding_dim=embedding_size)
    loader = create_data_loader(
        tokens=tokens,
        context_length=context_length,
        batch_size=4,
    )
    x, y = next(iter(loader))
    print("X:", x)
    print("X shape:", x.shape)
    # print("Y:", y)
    output = embedding(x)
    # print("Output:", output)
    print("Output Shape:", output.shape)

    # print("Y shape:", y.shape)
    # print("Loader:", len(loader))
    # for batch_x, batch_y in loader:
    #     print("X:", batch_x.shape)
    #     # print("Y:", batch_y)


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
