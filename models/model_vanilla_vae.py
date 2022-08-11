import torch
import utils
from . import Abstract_Model

class VanillaVAE(Abstract_Model):
    def __init__(self,
        backbone,
        backend,
        backbone_created=False,
        backend_created=False
    ):
        super(VanillaVAE, self).__init__(backbone, backend, backbone_created, backend_created)

        # for test/val
        self.psnr_m = utils.AverageMeter()

    def compute_metric(self, output, target):
        """Computes PSNR value"""

        batch_size = output.size(0)
        output = output.cpu()
        target = target.cpu()

        with torch.no_grad():
            for i in range(batch_size):
                psnr_value = PSNR(output[i], target[i]).item()
                self.psnr_m.update(psnr_value)

    def get_metric_value(self):
        return self.psnr_m.avg

    def display_metric_value(self):
        print(f'PSNR value: {self.get_metric_value()}')

    def reset_metric(self):
        self.psnr_m.reset()