import string
class CharTokenizer:
    
    def __init__(self):
        chars = (
            string.ascii_letters +
            string.digits +
            string.punctuation +
            " \n" +
            "."
        )
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text:str):
        return [self.stoi[c] for c in text]

    def decode(self, ids:list[int]):
        return "".join(self.itos[i] for i in ids)
