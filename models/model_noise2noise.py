import torch
import torch.nn as nn
from . import Abstract_Model

class Noise2Noise(Abstract_Model):
    def __init__(self,
        backbone,
        backend,
        backbone_created=False,
        backend_created=False
    ):
        super(Noise2Noise, self).__init__(backbone, backend, backbone_created, backend_created)

        # for test/val
        self.total = 0
        self.correct = 0

    def compute_metric(self, output, target):
        """Computes PSNR value"""
        # TODO: compute PSNR value
        
        with torch.no_grad():
            self.total += target.size(0)
            _, predicted = torch.max(output.data, 1)
            self.correct += (predicted == target).sum().item()

    def get_metric_value(self):
        return 100 * self.correct // self.total

    def display_metric_value(self, epoch):
        print(f'Epoch: {epoch}, validate accuracy: {self.get_metric_value()} %')

    def reset_counter(self):
        self.total = 0
        self.correct = 0