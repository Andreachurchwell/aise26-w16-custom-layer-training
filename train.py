import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import LearnedAffine

# Simple neural network using our custom affine layer
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Custom affine layer that learns gamma and beta per input feature
        self.affine = LearnedAffine(784)
        # Final linear layer mapping 784 features with 10 class logits
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        # Flatten image from (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.shape[0], -1)
        # Apply custom affine transform
        x = self.affine(x)
        # Map to 10 output classes
        x = self.fc(x)
        return x

# Fix random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    SEED = 42
    set_seed(SEED)
#  Covert images to tensors and normalize pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
#  Load FashionMNIST training and test datasets
    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
# Wrap datasets in DataLoaders for batching and shuffling
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
# Initialize model
    model = SimpleNet()
# Loss function for multi-class classification
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer that adapts learning rates ad uses weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Learning rate scheduler that decays LR by 10% each epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # print("Setup complete. Ready to train.")
    epochs = 3
    for epoch in range(1, epochs + 1):
        print(f'----- starting epoch {epoch} -----')
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        # Evaluate on validation set
        val_loss, val_acc = eval_one_epoch(model, test_loader, loss_fn)
        # Step the learning rate scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        # Print metrics
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )
# Training loop for one epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()  # Enable training mode (dropout, batchnorm, etc.)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        optimizer.zero_grad() # Clear previous gradients
        preds = model(x)   # Forward pass
        loss = loss_fn(preds, y) # Compute loss
        loss.backward() # Backpropagate gradients
        optimizer.step() # Update model weights
        # Accumulate metrics
        total_loss += loss.item() * x.size(0)
        total_correct += (preds.argmax(dim=1) == y).sum().item()
        total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc

# Evaluation loop (no gradient updates)
def eval_one_epoch(model, loader, loss_fn):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():  # Disable gradient tracking for efficiency
        for x, y in loader:
            preds = model(x)
            loss = loss_fn(preds, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (preds.argmax(dim=1) == y).sum().item()
            total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc



if __name__ == "__main__":
    main()
