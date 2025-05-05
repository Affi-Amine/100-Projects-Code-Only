import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Feature values
Y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])  # Labels

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single feature, single output
    
    # Fixed indentation for forward method
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid to output

# Moved these lines OUTSIDE the class definition
model = LogisticRegression()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Increased learning rate

# Training loop (should NOT be inside the class)
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1:4d} | Loss: {loss.item():.4f}')

params = list(model.parameters())
print(f'Learned weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')