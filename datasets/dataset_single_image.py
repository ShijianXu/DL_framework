import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetfromImage(Dataset):
    """Sample from a distribution defined by an image using PyTorch Dataset."""

    def __init__(self, img, max_val=1.0, sample_size=1000):
        """
        Initializes the dataset with an image and samples according to defined probabilities.
        Args:
            img (numpy.ndarray): The image array where pixel values define the probability.
            max_val (float): The maximum value for coordinate scaling.
            sample_size (int): Number of samples to pre-generate.
        """
        h, w = img.shape
        xx, yy = np.meshgrid(np.linspace(-max_val, max_val, w),
                             np.linspace(-max_val, max_val, h))
        means = np.stack([xx.ravel(), yy.ravel()], axis=1)
        probs = img.ravel() / img.sum()
        noise_std = np.array([max_val / w, max_val / h])

        self.samples = []
        indices = np.random.choice(means.shape[0], size=sample_size, p=probs)
        for idx in indices:
            m = means[idx]
            sample = np.random.randn(*m.shape) * noise_std + m
            self.samples.append(sample)

        self.samples = np.array(self.samples)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns one sample of data.
        Args:
            index (int): The index of the item.
        
        Returns:
            Tensor: A single sample from the dataset.
        """
        return torch.from_numpy(self.samples[index]).float()
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    img = np.random.rand(100, 100)
    dataset = DatasetfromImage(img, max_val=4.0, sample_size=40000)

    # Create a DataLoader instance
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Use the DataLoader in a training loop or elsewhere
    for data in dataloader:
        print(data.shape)
        break