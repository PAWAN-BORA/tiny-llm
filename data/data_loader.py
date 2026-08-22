
from dataset.text_dataset import TextDataset
from torch.utils.data import DataLoader


def create_data_loader(
    tokens,
    context_length,
    batch_size,
):
    dataset = TextDataset(
        tokens=tokens,
        context_length=context_length
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
        
    )
    return loader;

