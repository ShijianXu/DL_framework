import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from models.model_normalizing_flow_iris import Iris_NormalizingFlow
from models.module_coupling_layer import IrisCouplingLayer
from losses import IrisLoss


# Model part
class Conditioner(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output = nn.Linear(hidden_dim, out_dim*2)      # output 2 values for scale and shift
        
    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        for layer in self.hidden:
            x = F.leaky_relu(layer(x))

        x = self.output(x).chunk(2, dim=-1)
        return x

data_dim = 4        # 4 features in the iris dataset
num_flows = 5
prior = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.zeros(data_dim).to(device='cuda'),
    covariance_matrix=torch.eye(data_dim).to(device='cuda')
)

flow_layers = []
for i in range(num_flows):
    flow_layers += [IrisCouplingLayer(
                        net=Conditioner(in_dim=data_dim//2, out_dim=data_dim//2, hidden_dim=100, num_layers=1),
                        split=lambda x: x.chunk(2, dim=-1)
                    )]

model_config = {
    "flows": flow_layers,
    "prior": prior
}
model = Iris_NormalizingFlow(**model_config)
print(f"Total model parameters: {model.get_num_params()}")


# Data part
iris = load_iris()
X, y = iris.data, iris.target

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

valid_dataloader = None
print("Construct train dataset with {} samples".format(len(train_dataset)))
print("Construct test dataset with {} samples".format(len(test_dataset)))

# for i, (X, y) in enumerate(train_dataloader):
#     print(X.shape, y.shape)           # torch.Size([16, 4]) torch.Size([16])
#     break


# Loss and training part
num_epochs = 2000
learning_rate = 1e-3

loss = IrisLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=400, 
                                            gamma=0.3)

# Sample part, for compatibility with the Trainer
sample_valid = False
sample_valid_freq = -1   # sample images every 5 epochs