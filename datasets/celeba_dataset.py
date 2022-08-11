# code from https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/dataset.py

from torchvision.datasets import CelebA
from torchvision import transforms

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True

if __name__ == "__main__":
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    train_dataset = MyCelebA(
        './data/', 
        split='train',
        transform=train_transforms,
        download=False,
    )
    
    for idx, data in enumerate(train_dataset):
        print(len(data))
        print(data[0].shape, data[1].shape)
        break