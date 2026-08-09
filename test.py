from tokenizer.char_tokenizer import CharTokenizer;




def main():
    # word = "Hello Word caf"

    text = """
    hello world.
    hello pawan.
    the cat sat on the mat.
    the cat is happy.
    """
    toknizer = CharTokenizer()
    tokens = toknizer.encode(text=text);
    print(tokens)

    encoded_word = toknizer.decode(tokens)
    print(encoded_word)


if __name__ == "__main__":
    main()
