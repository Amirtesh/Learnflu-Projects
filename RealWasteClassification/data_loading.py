import torch
from torch.utils.data import DataLoader

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
            
    def __len__(self):
        return len(self.dl)
    
class DataLoaderCreator:

    def __init__(self,train_ds,test_ds,batch_size):
        self.train_ds=train_ds
        self.test_ds=test_ds
        self.batch_size=batch_size
        self.device=get_default_device()

    def get_dataloader(self):
        train_dl=DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_dl=DataLoader(self.test_ds, self.batch_size * 2, num_workers=4, pin_memory=True)

        train_dl=DeviceDataLoader(train_dl, self.device)
        test_dl=DeviceDataLoader(test_dl, self.device)

        return train_dl, test_dl