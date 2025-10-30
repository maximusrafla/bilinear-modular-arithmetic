"""
Train a bilinear layer on modular addition.

This implementation follows the bilinear layer architecture from:
"Bilinear Layers Enable Rapid Learning and Prediction in Sequence Transformers"
https://arxiv.org/abs/2410.08417

Task: Learn f(a,b) = (a + b) mod P where P=113
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

P = 113

class BilinearLayer(nn.Module):
    """Bilinear layer: output = W_out(W1(x) ⊙ W2(x))"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, input_dim, bias=False)
        
    def forward(self, x):
        return self.W_out(self.W1(x) * self.W2(x))


def generate_dataset(P, num_samples=10000):
    """Generate modular addition dataset with one-hot encoded inputs."""
    a_vals = np.random.randint(0, P, num_samples)
    b_vals = np.random.randint(0, P, num_samples)
    targets = (a_vals + b_vals) % P
    
    # One-hot encode
    a_onehot = np.zeros((num_samples, P))
    b_onehot = np.zeros((num_samples, P))
    a_onehot[np.arange(num_samples), a_vals] = 1
    b_onehot[np.arange(num_samples), b_vals] = 1
    
    X = np.concatenate([a_onehot, b_onehot], axis=1)
    
    return torch.FloatTensor(X), torch.LongTensor(targets)


def train_model(model, X, y, epochs=1000, lr=0.001, weight_decay=0.01):
    """Train the bilinear layer."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    print("Training...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean()
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {accuracy.item():.4f}")
    
    return losses, accuracies


def main():
    print(f"Training Bilinear Layer on Modular Addition (mod {P})")
    print("-" * 60)
    
    # Generate data
    X, y = generate_dataset(P, num_samples=20000)
    print(f"Dataset: {X.shape[0]} samples, input dim: {X.shape[1]}")
    
    # Initialize model
    input_dim = 2 * P
    hidden_dim = 256
    model = BilinearLayer(input_dim, hidden_dim)
    
    print(f"Model: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={P}")
    print("-" * 60)
    
    # Train
    losses, accuracies = train_model(model, X, y, epochs=1000, weight_decay=0.01)
    
    # Save results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(alpha=0.3)
    
    ax2.plot(accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/training_curves.png")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'P': P,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'final_loss': losses[-1],
        'final_accuracy': accuracies[-1]
    }, 'bilinear_model.pt')
    
    print(f"Final Loss: {losses[-1]:.4f} | Final Accuracy: {accuracies[-1]:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()