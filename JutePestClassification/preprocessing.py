import torchvision.transforms as tt
from torchvision.datasets import ImageFolder

class Preprocessor:
    def __init__(self,train_dir,test_dir,val_dir):
        self.train_dir=train_dir
        self.test_dir=test_dir
        self.val_dir=val_dir
        self.stats=([0.485,0.456,0.406],[0.229,0.224,0.225])
        self.train_transform=self._get_train_transform()
        self.val_test_transform=self._get_val_test_transform()

    def _get_train_transform(self):
        return tt.Compose([
            tt.Resize(224),
            tt.RandomCrop(224),
            tt.RandomHorizontalFlip(),
            tt.RandomVerticalFlip(),
            tt.RandomRotation(10),
            tt.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
            tt.RandomPerspective(distortion_scale=0.5,p=0.5,interpolation=3),
            tt.RandomAffine(degrees=10,translate=(0.1,0.1),scale=(0.8,1.2),shear=5),
            tt.GaussianBlur(kernel_size=5,sigma=(0.1,2.0)),
            tt.RandomGrayscale(p=0.2),
            tt.RandomResizedCrop(size=224,scale=(0.8,1.0)),
            tt.RandomSolarize(threshold=192.0),
            tt.ToTensor(),
            tt.Normalize(*self.stats)
        ])

    def _get_val_test_transform(self):
        return tt.Compose([
            tt.Resize(224),
            tt.RandomCrop(224),
            tt.ToTensor(),
            tt.Normalize(*self.stats)
        ])

    def get_datasets(self):
        train_ds=ImageFolder(self.train_dir,transform=self.train_transform)
        test_ds=ImageFolder(self.test_dir,transform=self.val_test_transform)
        valid_ds=ImageFolder(self.val_dir,transform=self.val_test_transform)
        classes=train_ds.classes
        return train_ds,test_ds,valid_ds,classes
    
    def get_dataset_lengths(self):
        return len(self.train_ds),len(self.test_ds),len(self.valid_ds)