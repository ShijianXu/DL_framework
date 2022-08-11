import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image

class Sampler(object):
    def __init__(self,
        config,
        model,
        latent_dim,
        ckpt_path
    ):
        self.config = config
        self.model = model
        self.latent_dim = latent_dim

        self.ckpt_path = ckpt_path

        # check device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # to device
        self.model.to(self.device)

        # load ckpt
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.ckpt_path))

    def sample(self, sample_num=144):
        print("Sampling ...")
        self.model.eval()
        self.model.reset_metric()

        samples = self.model.sample(sample_num, self.device)
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_image(samples.cpu().data,
            os.path.join(output_dir,
            f"test_sample.png"),
            normalize=True,
            nrow=12)