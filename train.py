import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import LearnedAffine


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = LearnedAffine(784)
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.affine(x)
        x = self.fc(x)
        return x


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    SEED = 42
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SimpleNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # print("Setup complete. Ready to train.")
    epochs = 3
    for epoch in range(1, epochs + 1):
        print(f'----- starting epoch {epoch} -----')
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = eval_one_epoch(model, test_loader, loss_fn)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (preds.argmax(dim=1) == y).sum().item()
        total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
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
