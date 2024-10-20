import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

class Preprocessor:
    def __init__(self, data_dir, val_split=0.2):
        self.data_dir = data_dir
        self.val_split = val_split
        self.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = self._get_transform()

    def _get_transform(self):
        return tt.Compose([
            tt.Resize(224),
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize(*self.stats)
        ])

    def get_datasets(self):
        dataset = ImageFolder(self.data_dir, transform=self.transform)
        train_size = int((1 - self.val_split) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        classes = dataset.classes
        return train_dataset, test_dataset, classes

    def get_dataset_lengths(self):
        train_dataset, test_dataset, _ = self.get_datasets()
        return len(train_dataset), len(test_dataset)
