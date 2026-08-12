import torch
# from torch.utils.data import Dataset

class TextDataset:
    def __init__(self, tokens, context_length):
        """
        tokens:
            Complete tokenized text.

        context_length:
            Number of tokens the model can see at once.
        """
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        print(self.tokens.shape)
        print(self.tokens.dtype)
        print(self.tokens.device)

    def __len__(self):
         # We need one additional token for the target.
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):
        # Input sequence
        x = self.tokens[
            index:index + self.context_length
        ]

        # Same sequence shifted by one token.
        # This is what the model needs to predict.
        y = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return x, y
