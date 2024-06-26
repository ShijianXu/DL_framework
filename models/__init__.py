import importlib
import torch.nn as nn

class Abstract_Model(nn.Module):
    def __init__(self,
        backbone,
        backend,
        backbone_created=False,
        backend_created=False
    ):
        super(Abstract_Model, self).__init__()

        # build modules
        if not backbone_created:
            self.backbone = Abstract_Model.build_module(backbone)
        else:
            self.backbone = backbone

        # There is always a backbone, but there is no backend for some models
        if backend is None:
            self.backend = None
        elif not backend_created:
            self.backend = Abstract_Model.build_module(backend)
        else:
            self.backend = backend

        self.num_params_backbone = sum(param.numel() for param in self.backbone.parameters())
        self.num_params_backend = sum(param.numel() for param in self.backend.parameters()) \
            if self.backend is not None else 0

    @staticmethod
    def build_module(module_config):
        class_name = module_config["name"].split('.')[-1]
        module_name = module_config["name"][:(len(module_config["name"]))-len(class_name)-1]

        class_type = getattr(importlib.import_module(module_name), class_name)
        obj = class_type(**module_config["config"])

        return obj

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Abstract method not implemented.")
    
    def process_batch(self, *args, **kwargs):
        raise NotImplementedError("Abstract method not implemented.")

    def everything_to(self, device):
        """
        For internal tensors defined in the model,
        manually move them to the device.
        """
        return

    # def forward(self, x):
    #     x = self.backbone(x)
    #     if self.backend is not None:
    #         out = self.backend(x)
    #     else:
    #         out = x
    #     return out

    def get_num_params(self):
        print(f"Backbone num of params: {self.num_params_backbone}")
        print(f"Backend num of params: {self.num_params_backend}")
        return self.num_params_backbone + self.num_params_backend

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError("Abstract method not implemented.")

    def compute_metric(self, *args, **kwargs):
        raise NotImplementedError("Abstract method not implemented.")

    def get_metric_value(self):
        raise NotImplementedError("Abstract method not implemented.")

    def display_metric_value(self):
        raise NotImplementedError("Abstract method not implemented.")

    def reset_metric(self):
        raise NotImplementedError("Abstract method not implemented.")