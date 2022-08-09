import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image

class Tester(object):
    def __init__(self,
        config,
        model,
        dataloader,
        ckpt_path,
        print_freq=400,
    ):
        self.config = config
        self.model = model
        self.dataloader = dataloader

        self.print_freq = print_freq
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

    def test_with_gt(self):
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                inputs, target = batch
                output = self.model(inputs.to(self.device))
                self.model.compute_metric(output, target.to(self.device))

        self.model.display_metric_value()

    def test_without_gt(self):
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                inputs = batch['img'].to(self.device)
                output = self.model(inputs).cpu()
                inputs_name = batch['img_name']+'_noisy.png'
                output_name = batch['img_name']+'_output.png'
                save_image(inputs, os.path.join(output_dir, inputs_name))
                save_image(output, os.path.join(output_dir, output_name))

    def test(self):
        print("Testing ...")
        self.model.eval()
        self.model.reset_metric()

        if self.config.test_require_gt:
            self.test_with_gt()
        else:
            self.test_without_gt()