# Custom dataset for loading circle data using sklearn.


import torch
from torch.utils import data
from sklearn.datasets import make_circles

class CircleDataset(data.Dataset):
    """Custom Dataset for loading circle data."""
    def __init__(self, num_samples):
        self.points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
        self.points = torch.tensor(self.points, dtype=torch.float32)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]
    

if __name__ == "__main__":
    dataset = CircleDataset(num_samples=512*1000)
    print(len(dataset))