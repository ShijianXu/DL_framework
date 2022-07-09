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

        self.load_model(self.ckpt_path)

    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(ckpt_path))

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                inputs, target = batch
                output = self.model(inputs)
                cor, tot = self.model.accuracy(output, target)
                correct += cor
                total += tot

        print(f'Test accuracy: {100 * correct // total} %')
