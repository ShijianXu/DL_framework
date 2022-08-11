# code from https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/dataset.py
import os

from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True


class TinyCelebA(Dataset):
    def __init__(self, root_dir, sample_nums, transform):
        super(TinyCelebA, self).__init__()

        self.root_dir = root_dir
        all_imgs = os.listdir(root_dir)
        self.sample_nums = sample_nums
        self.imgs = all_imgs[:sample_nums]
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.root_dir, self.imgs[index]))
        if self.transform is not None:
            x = self.transform(x)

        target = None
        return x, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    train_dataset = TinyCelebA(
        './data/celeba/img_align_celeba',
        sample_nums=10000,
        transform=train_transforms
    )
    
    for idx, data in enumerate(train_dataset):
        print(len(data))
        print(data[0].shape)
        break