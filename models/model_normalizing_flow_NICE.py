import torch
import torch.nn as nn

class NICE(nn.Module):
    def __init__(self, in_dim, flows, scale_layer, prior):
        """
        Initialize the NICE model.
        Args:
            flows: list, a list of coupling layers
            prior: torch.distributions, the prior distribution for the latent space
        """
        super(NICE, self).__init__()
        self.in_dim = in_dim
        self.flows = nn.ModuleList(flows)
        self.scale_layer = scale_layer
        self.prior = prior          # prior distribution for the latent space
        self.num_params_flows = sum(p.num_params for p in self.flows if hasattr(p, 'num_params'))

    def encode(self, x):
        """
        Encode the input x into a latent representation z.
        For NICE, because of the additive coupling layers, 
        we don't need to compute the log determinant of the Jacobian.
        The determinant of the Jacobian is always 1,
        the log determinant is always 0.
        """
        for flow in self.flows:
            x = flow(x)
        x, log_det_J = self.scale_layer(x)
        return x, log_det_J
    
    @torch.no_grad()
    def decode(self, z):
        """
        Decode the latent representation z back to the original image x
        """
        with torch.no_grad():
            x = self.scale_layer.reverse(z)
            for flow in reversed(self.flows):
                x = flow.reverse(x)
            return x
        
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
    
    def dequantization(self, x):
        # add a uniform noise of 1/256 to the data 
        # and rescale it to be in [0, 1]^D
        noise = torch.distributions.Uniform(0.0, 1.0).sample(x.shape).to(x.device)
        x = (x*255 + noise) / 256.0
        return x
    
    def process_batch(self, batch, criterion, device):
        """
        Process a batch of images through the normalizing flow.
        Args:
            batch: torch.Tensor, a batch of images
            criterion: nn.Module, the loss function to use
            device: str, the device to use
        """
        x = batch[0].to(device)

        # dequantize input
        x = self.dequantization(x)
        x = x.view(x.size(0), -1)

        z, log_det_J = self(x)
        loss = self.compute_loss(z, log_det_J, criterion)
        return loss
    
    @torch.no_grad()
    def generate(self, num_samples, device):
        """
        Generate samples from the model.
        """
        z = self.prior.sample((num_samples, self.in_dim)).to(device)
        x = self.decode(z)

        x = x.view(num_samples, 1, 28, 28)
        return x
    
    def get_num_params(self):
        return self.num_params_flows
    
    def everything_to(self, device):
        pass

    def reset_metric(self):
        pass