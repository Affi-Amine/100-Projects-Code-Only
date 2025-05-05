import torch

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

w = torch.randn(1, requires_grad=True, dtype=torch.float32) #weights
b = torch.randn(1, requires_grad=True, dtype=torch.float32) #biases

learning_rate = 0.01
epochs = 1000

# ======================
# TRAINING LOOP
# ======================

for epochs in range(epochs):
    # ----------
    # FORWARD PASS
    # ----------
    y_pred = X * w + b
    # ----------
    # LOSS CALCULATION
    # ----------
    loss = torch.mean((y_pred - Y) ** 2)
    # ----------
    # BACKPROPAGATION
    # ----------
    loss.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        # ----------
        # GRADIENT RESET
        # ----------
        w.grad.zero_()
        b.grad.zero_()
        
        if (epochs + 1) % 100 == 0:
            print(f'Epoch {epochs+1:4d} | Loss: {loss.item():.4f}')
        
print('\nTraining complete!')
print(f'Learned parameters:')
print(f'Weight (w): {w.item():.4f} (Expected ≈2.0)')
print(f'Bias (b): {b.item():.4f} (Expected ≈0.0)')