import torch
from . import Abstract_Model

class CLS_Model(Abstract_Model):
    def __init__(self,
        backbone,
        backend,
        backbone_created=False,
        backend_created=False
    ):
        super(CLS_Model, self).__init__(backbone, backend, backbone_created, backend_created)

        # for test/val
        self.total = 0
        self.correct = 0

    def compute_metric(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            self.total += target.size(0)
            _, predicted = torch.max(output.data, 1)
            self.correct += (predicted == target).sum().item()

    def get_metric_value(self):
        return 100 * self.correct // self.total

    def display_metric_value(self):
        print(f'Accuracy: {self.get_metric_value()} %')

    def reset_metric(self):
        self.total = 0
        self.correct = 0