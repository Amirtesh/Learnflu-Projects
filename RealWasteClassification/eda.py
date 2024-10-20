import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class EDA:
    def __init__(self,classes,stats):
        self.classes=classes
        self.stats=stats

    def denormalize(self,images,means,stds): 
        if len(images.shape)==3:
            images=images.unsqueeze(0)
        means=torch.tensor(means).reshape(1,3,1,1)
        stds=torch.tensor(stds).reshape(1,3,1,1)
        return images*stds+means

    def show_image(self,img_tensor,label):
        print('Label: ',self.classes[label],'('+str(label)+')')
        img_tensor=self.denormalize(img_tensor,*self.stats)[0].permute((1,2,0))
        plt.imshow(img_tensor)
        plt.axis('off')
        plt.show()

    def show_batch(self,dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images.cpu(), nrow=16).permute(1, 2, 0))
            plt.axis('off')
            plt.show()
            break
