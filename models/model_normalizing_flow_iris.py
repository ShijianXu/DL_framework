import torch
import torch.nn as nn

class Iris_NormalizingFlow(nn.Module):
    def __init__(self, flows, prior, is_image=False):
        """
        Args:
            flows: List of nn.Module, each nn.Module is a flow layer
        """

        super(Iris_NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = prior          # prior distribution for the latent space
        self.is_image = is_image
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
        Decode the latent representation z back to the original input x
        """
        with torch.no_grad():
            for flow in reversed(self.flows):
                z, log_det_J = flow.reverse(z, log_det_J)
            return z, log_det_J

    def forward(self, x):
        """
        Forward pass through the normalizing flow.
        Args:
            x: torch.Tensor, the input
        """
        z, log_det_J = self.encode(x)
        return z, log_det_J

    def compute_loss(self, z, log_det_J, criterion):
        """
        Compute the loss given the log probability of the input.
        Args:
            log_p_x: torch.Tensor, the log probability of the input
            criterion: nn.Module, the loss function to use
        """
        loss = criterion(self.prior, z, log_det_J)
        return loss

    def process_batch(self, batch, criterion, device):
        """
        Process a batch of inputs through the normalizing flow.
        Args:
            batch: torch.Tensor, a batch of inputs
            criterion: nn.Module, the loss function to use
            device: str, the device to use
        """
        inputs = batch[0].to(device)
        if self.is_image:
            inputs = inputs.view(inputs.size(0), -1)    # flatten the image
        z, log_det_J = self(inputs)
        loss = self.compute_loss(z, log_det_J, criterion)
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, device):
        """
        Sample from the prior distribution.
        Args:
            num_samples: number of samples to sample
            device: str, the device to use
        """
        z = self.prior.sample((num_samples,)).to(device)
        log_det_J = torch.zeros(num_samples, device=device)
        x, log_det_J = self.decode(z, log_det_J)
        img = x.view(num_samples, 1, 14, 14) if self.is_image else x    # reshape the image
        return img

    def get_num_params(self):
        return self.num_params_flows
    
    def everything_to(self, device):
        pass

    def reset_metric(self):
        pass