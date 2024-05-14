import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mu = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        z_mu = self.fc2_mu(h)
        z_logvar = self.fc2_logvar(h)
        return z_mu, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, input_dim)
        
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

class LinearDynamicSystem(nn.Module):
    def __init__(self, latent_dim):
        super(LinearDynamicSystem, self).__init__()
        self.A = nn.Parameter(torch.eye(latent_dim))
        self.B = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        
    def forward(self, z, u=None):
        if u is None:
            u = torch.zeros_like(z)
        z_next = torch.matmul(z, self.A) + torch.matmul(u, self.B)
        return z_next

class SeqVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SeqVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.lds = LinearDynamicSystem(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, z_prev, u=None):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        z_next = self.lds(z_prev, u)
        x_recon = self.decoder(z)
        return x_recon, z, z_mu, z_logvar, z_next

# 超参数设置
input_dim = 10  # 观测数据的维度
latent_dim = 3  # 潜在表示的维度
learning_rate = 0.001
num_epochs = 100

# 模型和优化器
model = SeqVAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

# 模拟数据
x_t = torch.randn(100, input_dim)  # 当前时刻的观测数据
x_t1 = torch.randn(100, input_dim) # 下一时刻的观测数据

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    z_prev = torch.zeros(100, latent_dim)  # 初始隐状态，可以根据实际情况调整
    x_recon, z, z_mu, z_logvar, z_next = model(x_t, z_prev)
    
    recon_loss = loss_function(x_recon, x_t)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    transition_loss = loss_function(z_next, model.encoder(x_t1)[0])  # 使用 encoder 的 mu 作为真实值
    loss = recon_loss + kld_loss + transition_loss
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training complete.")
