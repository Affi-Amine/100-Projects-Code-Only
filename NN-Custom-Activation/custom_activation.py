import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

x = torch.linspace(-3, 3, 100).unsqueeze(1) # the unsqueeze makes it 2D tensor
y = torch.sin(x) + 0.1 * torch.randn(x.size()) # applies sin to x then adds noise

dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#  define a custom activation function
def Swish(x):
    return x * torch.sigmoid(x)

class SwishNet(nn.Module):
    def __init__(self):
        super(SwishNet, self).__init__()
        self.net = nn.Sequential(
            nn.linear(1, 32),
            Swish(),
            nn.Linear(32, 32),
            Swish(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)
        
model = SwishNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training loop
for epoch in range(1000):
    for x, y in train_loader:
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1:4d} | Loss: {loss.item():.4f}')
    
