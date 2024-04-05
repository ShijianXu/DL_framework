import torch

from . import Abstract_Model

class NF_Model(Abstract_Model):
    def __init__(self, 
        backbone, 
        backend, 
        backbone_created=False, 
        backend_created=False,
        img_shape=[1, 28, 28]
    ):
        super().__init__(backbone, backend, backbone_created, backend_created)

        # Create prior distribution for final latent space
        # used for sampling
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

        # Record the shape of the input image
        self.img_shape = img_shape

    def compute_loss(self, output, criterion):
        """
        Computes the negative log-likelihood loss
        """
        z, log_det_J = output[0], output[1]
        return criterion(z, log_det_J)

    def sample(self, num_samples, device):
        """
        Sample num_samples of images from the flow model
        """
        z = self.prior.sample((num_samples, *self.img_shape)).to(device)

        # Transform z to x by inverting the flow
        x, log_det_J = self.backbone.inverse(z)
        return x

    def compute_metric(self, source, output, target):
        """Computes bits per dimension (BPD) value"""
        pass

    def forward(self, x):
        z, log_det = self.backbone(x)
        