import torch
import torch.nn as nn

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.module_finite_scalar_quantization import FSQ

class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, len(levels), kernel_size=1),
                FSQ(levels),
                nn.Conv2d(len(levels), 32, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, FSQ):
                x, indices = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1), indices
    
    def compute_loss(self, out, x, criterion):
        """
        Compute the loss given the log probability of the input image.
        Args:
            log_p_x: torch.Tensor, the log probability of the input image
            criterion: nn.Module, the loss function to use
        """
        loss = criterion(out, x)
        return loss
    
    def process_batch(self, batch, criterion, device):
        """
        Process a batch of images through the normalizing flow.
        Args:
            batch: torch.Tensor, a batch of images
            criterion: nn.Module, the loss function to use
            device: str, the device to use
        """
        x = batch[0].to(device)        
        out, indices = self(x)

        loss = self.compute_loss(out, x, criterion)
        return {"loss": loss}
        
    def get_num_params(self):
        return self.num_params
    
    def everything_to(self, device):
        pass

    def reset_metric(self):
        pass
    

if __name__ == "__main__":
    x = torch.randn(1, 1, 28, 28)
    model = SimpleFSQAutoEncoder(levels=[8, 6, 5])
    y, indices = model(x)
    