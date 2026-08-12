from dataset.text_dataset import TextDataset
from tokenizer.char_tokenizer import CharTokenizer;




def main():
    # word = "Hello Word caf"

    text = """
    hello world.
    hello pawan.
    the cat sat on the mat.
    the cat is happy.
    """
    tokenizer = CharTokenizer()
    tokens = tokenizer.encode(text=text);
    dataset = TextDataset(tokens=tokens, context_length=8)
    print("Number of samples:", len(dataset))
    x, y = dataset[5]
    print("X:", x)
    print("Y:", y)
    print("X decoded:", tokenizer.decode(x.tolist()))
    print("Y decoded:", tokenizer.decode(y.tolist()))
    # print(tokens)

    # encoded_word = toknizer.decode(tokens)
    # print(encoded_word)


if __name__ == "__main__":
    main()
