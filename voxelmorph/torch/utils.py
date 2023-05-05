from torch.utils.data import Dataset

class RegData(Dataset):
    def __init__(self, generator, length = 100):
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs, y_true = next(self.generator)
        return inputs[0].squeeze(0), inputs[1].squeeze(0), y_true[0].squeeze(0), y_true[1].squeeze(0)
