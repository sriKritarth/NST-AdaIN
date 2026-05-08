from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch

class CustomImageDataset(Dataset):

    def __init__(self, root_dir , transform = None):
        super(CustomImageDataset , self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.filename = list()
        for dir in os.listdir(root_dir):
            if dir.lower().endswith(('.jpg' , '.png' , '.jpeg')):
                self.filename.append(dir)

    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir , self.filename[index])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
                


def get_transform(content_size , final_size , crop):

    transform = []
    if content_size > 0:
        transform.append(transforms.Resize(content_size))

    if crop:
        transform.append(transforms.RandomCrop(final_size))
    
    else:
        transform.append(transforms.Resize((final_size , final_size)))

    transform.append(transforms.ToTensor())

    return transforms.Compose(transform)

def adaptive_instance_normalization(content_feat , style_feat):
    
    #[batch_size , channel , h , w]
    size = content_feat.size()
    content_mean , content_std = calc_std_mean(content_feat)

    style_mean , style_std = calc_std_mean(style_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



def calc_std_mean(feat , eps = 1e-5):
    
    #[batch_size , channel , h , w]

    size = feat.size()

    assert (len(size) == 4) # Check dimensions = 4

    batch_size  = feat.size(0)
    channels = feat.size(1)

    feat_mean = feat.view(batch_size , channels , -1).mean(dim=2).view(batch_size , channels , 1 , 1)

    feat_var = feat.view(batch_size , channels , -1).var(dim=2 , unbiased = False) + eps

    feat_std = feat_var.sqrt().view(batch_size , channels , 1 , 1)

    return feat_mean , feat_std
    