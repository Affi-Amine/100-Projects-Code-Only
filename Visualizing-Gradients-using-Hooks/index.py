import torch
import torch.nn as nn
import torch.nn.functional as F
 
X = torch.randn(10, 3)  # 10 samples, 3 features
Y = torch.randn(10, 1)  # 10 targets

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # Input -> Hidden
        self.fc2 = nn.Linear(5, 1)  # Hidden -> Output
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
model = SmallNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
def print_grad_hook(module, grad_input, grad_output):
    print(f"\n--- Gradient for {module.__class__.__name__} ---")
    print("Grad Input:", grad_input)
    print("Grad Output:", grad_output)
    
hook = model.fc1.register_backward_hook(print_grad_hook)
 
output = model(X)
loss = criterion(output, Y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

hook.remove()