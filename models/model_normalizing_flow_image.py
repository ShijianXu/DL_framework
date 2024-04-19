import torch
import torch.nn as nn

class Image_NormalizingFlow(nn.Module):
    def __init__(self, flows, prior):
        """
        Args:
            flows: List of nn.Module, each nn.Module is a flow layer
        """

        super(Image_NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = prior          # prior distribution for the latent space
        self.num_params_flows = sum(p.num_params for p in self.flows if hasattr(p, 'num_params'))

    def encode(self, x):
        """
        Encode the input x into a latent representation z,
        additionally return the log determinant of the Jacobian for this transformation.
        """
        x, log_det_J = x, torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            x, log_det_J = flow(x, log_det_J)
        return x, log_det_J
    
    @torch.no_grad()
    def decode(self, z, log_det_J):
        """
        Decode the latent representation z back to the original image x
        """
        with torch.no_grad():
            for flow in reversed(self.flows):
                z, log_det_J = flow.reverse(z, log_det_J)
            return z, log_det_J

    def forward(self, x):
        """
        Forward pass through the normalizing flow.
        Args:
            x: torch.Tensor, the input image
        """
        z, log_det_J = self.encode(x)
        return z, log_det_J

    def compute_loss(self, z, log_det_J, criterion):
        """
        Compute the loss given the log probability of the input image.
        Args:
            log_p_x: torch.Tensor, the log probability of the input image
            criterion: nn.Module, the loss function to use
        """
        loss = criterion(self.prior, z, log_det_J)
        return loss    

    def process_batch(self, batch, criterion, device):
        """
        Process a batch of images through the normalizing flow.
        Args:
            batch: torch.Tensor, a batch of images
            criterion: nn.Module, the loss function to use
            device: str, the device to use
        """
        images = batch[0].to(device)
        z, log_det_J = self(images)
        loss = self.compute_loss(z, log_det_J, criterion)
        return loss
    
    @torch.no_grad()
    def generate(self, img_shape, device):
        """
        Sample images from the prior distribution.
        Args:
            img_shape: Tuple, the shape of the images to sample
            device: str, the device to use
        """
        z = self.prior.generate(img_shape).to(device)
        log_det_J = torch.zeros(img_shape[0], device=device)
        img, log_det_J =  self.decode(z, log_det_J)
        return img

    def get_num_params(self):
        return self.num_params_flows
    
    def everything_to(self, device):
        pass

    def reset_metric(self):
        pass