# https://github.com/joeylitalien/noise2noise-pytorch 

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as tvF


class AbstractDataset(Dataset):
    """ Abstract dataset class for Noise2Noise """
    def __init__(self, root_dir, crop_size=128, clean_targets=False) -> None:
        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Perform random square crop of fixed size.
        Works with list so that all items get the same cropped window.
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f"Error: Crop size: {self.crop_size}, Image size: ({w}, {h})"
        
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))
            
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs

    def __getitem__(self, index):
        raise NotImplementedError('Abstract method not implemented!')

    def __len__(self):
        return len(self.imgs)


class Noise2NoiseDataset(AbstractDataset):
    def __init__(self, root_dir, crop_size=128, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None
    ):
        super(Noise2NoiseDataset, self).__init__(root_dir, crop_size, clean_targets)

        self.imgs = os.listdir(root_dir)

        # Noise params (max std for Gaussian noise)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self, img):
        w, h = img.size
        c = len(img.getbands())

        if self.noise_type == 'gaussian':
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)

            noise = np.random.normal(0, std, size=(h, w, c))

            noise_img = np.array(img) + noise   # from PIL => numpy array

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)       # from numpy array => PIL

    def _corrupt(self, img):
        if self.noise_type in ['gaussian']:
            return self._add_noise(img)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img = Image.open(img_path).convert('RGB')

        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # torchvision.transforms.functional.to_tensor() converts a PIL Image or numpy.ndarray to tensor. 
        # This to_tensor() also normalize to [0.0, 1.0]
        # torchvision.transforms.ToTensor(), this will Converts a PIL Image or numpy.ndarray (H x W x C) 
        # in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
        # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) 
        # or if the numpy.ndarray has dtype = np.uint8.
        
        # print(img.getextrema())
        source = tvF.to_tensor(self._corrupt(img))
        
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target



if __name__ == '__main__':
    dataset = Noise2NoiseDataset('./data/DIV2K_valid_HR')
    for idx, data in enumerate(dataset):
        save_image(data[0], "./logs/source.png")
        save_image(data[1], "./logs/target.png")
        break