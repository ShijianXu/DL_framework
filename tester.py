import torch
from tqdm import tqdm
import utils

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

    def test(self):
        self.model.eval()
        self.model.reset_metric()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                inputs, target = batch
                output = self.model(inputs.to(self.device))
                self.model.compute_metric(output, target.to(self.device))

        print(f'Test accuracy: {self.model.get_metric_value()} %')
