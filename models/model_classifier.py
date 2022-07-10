import importlib

import torch
import torch.nn as nn

class CLS_Model(nn.Module):
    def __init__(self,
        backbone,
        backend,
    ):
        super(CLS_Model, self).__init__()

        # build modules
        self.backbone = CLS_Model.build_module(backbone)
        self.backend = CLS_Model.build_module(backend)

        self.num_params_backbone = sum(param.numel() for param in self.backbone.parameters())
        self.num_params_backend = sum(param.numel() for param in self.backend.parameters())

        # for test/val
        self.total = 0
        self.correct = 0

    @staticmethod
    def build_module(module_config):
        class_name = module_config["name"].split('.')[-1]
        module_name = module_config["name"][:(len(module_config["name"]))-len(class_name)-1]
        
        class_type = getattr(importlib.import_module(module_name), class_name)
        obj = class_type(**module_config["config"])

        return obj

    def forward(self, x):
        x = self.backbone(x)
        out = self.backend(x)
        return out

    def accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            self.total += target.size(0)
            _, predicted = torch.max(output.data, 1)
            self.correct += (predicted == target).sum().item()

    def get_test_acc(self):
        return 100 * self.correct // self.total

    def get_num_params(self):
        print(f"Backbone num of params: {self.num_params_backbone}")
        print(f"Backend num of params: {self.num_params_backend}")
        return self.num_params_backbone + self.num_params_backend

    def reset_counter(self):
        self.total = 0
        self.correct = 0