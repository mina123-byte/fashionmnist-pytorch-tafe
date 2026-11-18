import torch
import csv
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



learning_rate = 1e-3
batch_size = 64
epochs = 10

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class NeuralNetwork(nn.Module):
     def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

     def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

def train_loop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 200 == 0:
            current = batch * batch_size + len(X)
            print(f"  loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
     model.eval()
     size = len(dataloader.dataset)
     num_batches = len(dataloader)
     test_loss, correct = 0, 0

     with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

     test_loss /= num_batches
     accuracy = 100 * correct / size
     print(f"  Test accuracy: {accuracy:>0.1f}%, average loss: {test_loss:>8f}")
     return test_loss, accuracy 

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

for t in range(epochs):
    print(f"Epoch {t + 1}/{epochs}")
    print("-" * 30)

    train_loop(train_dataloader, model, loss_fn, optimiser)
    test_loop(test_dataloader, model, loss_fn)

    print()

print("Done!")

# Inspect the state dict — one entry per weight matrix and bias vector
state_dict = model.state_dict()

print("Parameter names and shapes:")
print("-" * 50)
for name, tensor in state_dict.items():
    print(f"{name:>45s}  {str(tuple(tensor.shape)):>15s}")


total_params = sum(p.numel() for p in model.parameters())
print(f"Total learnable parameters: {total_params:,}")

torch.save(model.state_dict(), "model_weights.pth")
print("Model weights saved to model_weights.pth")

# Step 1: create a new model instance (random weights)
loaded_model = NeuralNetwork().to(device)

# Step 2: load the saved weights
loaded_model.load_state_dict(
    torch.load("model_weights.pth", weights_only=True)
)

print("Weights loaded successfully.")


print("Original model:")
test_loop(test_dataloader, model, loss_fn)

print("\nLoaded model:")
test_loop(test_dataloader, loaded_model, loss_fn)


import os

os.makedirs("checkpoints", exist_ok=True)

# Fresh model and optimiser for this demonstration
torch.manual_seed(42)
ckpt_model = NeuralNetwork().to(device)
ckpt_optimiser = torch.optim.SGD(ckpt_model.parameters(), lr=learning_rate)

for epoch in range(5):
    print(f"Epoch {epoch + 1}/5")
    print("-" * 30)
    train_loop(train_dataloader, ckpt_model, loss_fn, ckpt_optimiser)
    test_loss, accuracy = test_loop(test_dataloader, ckpt_model, loss_fn)

    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": ckpt_model.state_dict(),
        "optimizer_state_dict": ckpt_optimiser.state_dict(),
        "loss": test_loss,
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
    print(f"  Checkpoint saved.\n")

print("Done!")


# Load checkpoint from epoch 3
checkpoint = torch.load("checkpoints/checkpoint_epoch_3.pth", weights_only=False)

# Reconstruct model and optimiser
resumed_model = NeuralNetwork().to(device)
resumed_model.load_state_dict(checkpoint["model_state_dict"])

resumed_optimiser = torch.optim.SGD(resumed_model.parameters(), lr=learning_rate)
resumed_optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
print(f"Resumed from epoch {start_epoch}")
print(f"Loss at checkpoint: {checkpoint['loss']:.6f}\n")

# Continue training from where we left off
for epoch in range(start_epoch, 5):
    print(f"Epoch {epoch + 1}/5")
    print("-" * 30)
    train_loop(train_dataloader, resumed_model, loss_fn, resumed_optimiser)
    test_loop(test_dataloader, resumed_model, loss_fn)
    print()


def train_loop_csv(dataloader, model, loss_fn, optimiser):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimiser.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimiser.step()

        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy


def test_loop_csv(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * X.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    test_loss = total_loss / total
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy


# ==============================================
# CSV Experiment Tracking
# ==============================================

os.makedirs("results", exist_ok=True)

csv_file = "results/training_history.csv"

# Use your existing FashionMNIST model
torch.manual_seed(42)
model_csv = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimiser_csv = torch.optim.Adam(
    model_csv.parameters(),
    lr=0.001
)

epochs_csv = 10

# Create CSV file with column headings
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "learning_rate"
    ])

# Training and CSV logging
for epoch in range(epochs_csv):

    train_loss, train_accuracy = train_loop_csv(
        train_dataloader,
        model_csv,
        loss_fn,
        optimiser_csv
    )

    test_loss, test_accuracy = test_loop_csv(
        test_dataloader,
        model_csv,
        loss_fn
    )

    # Save the results after each epoch
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            optimiser_csv.param_groups[0]["lr"]
        ])

    print(
        f"Epoch {epoch + 1}/{epochs_csv} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}%"
    )

print("\nTraining completed!")
print("CSV saved to:", os.path.abspath(csv_file))
        
import csv
import os
import torch

# Create results folder
os.makedirs("results", exist_ok=True)

loss_fn = torch.nn.CrossEntropyLoss()

# ========================================
# Function to run an experiment
# ========================================

def run_experiment(name, optimiser_type, learning_rate, epochs=5):

    # Reproducibility
    torch.manual_seed(42)

    # Create fresh model
    experiment_model = NeuralNetwork().to(device)

    # Select optimiser
    if optimiser_type == "SGD":
        experiment_optimiser = torch.optim.SGD(
            experiment_model.parameters(),
            lr=learning_rate
        )
    elif optimiser_type == "Adam":
        experiment_optimiser = torch.optim.Adam(
            experiment_model.parameters(),
            lr=learning_rate
        )
    else:
        raise ValueError("Unsupported optimiser")

    # Separate CSV file for each experiment
    csv_file = f"results/{name}.csv"

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "optimiser",
            "learning_rate"
        ])

    # Training
    for epoch in range(epochs):

        train_loss, train_accuracy = train_loop_csv(
            train_dataloader,
            experiment_model,
            loss_fn,
            experiment_optimiser
        )

        test_loss, test_accuracy = test_loop_csv(
            test_dataloader,
            experiment_model,
            loss_fn
        )

        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                optimiser_type,
                learning_rate
            ])

        print(
            f"{name} | Epoch {epoch + 1}/{epochs} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Accuracy: {test_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f}"
        )

    print(f"\nResults saved to {csv_file}\n")


# ========================================
# Experiment 1: SGD
# ========================================

run_experiment(
    name="SGD_lr_0.001",
    optimiser_type="SGD",
    learning_rate=0.001,
    epochs=5
)


# ========================================
# Experiment 2: Adam
# ========================================

run_experiment(
    name="Adam_lr_0.01",
    optimiser_type="Adam",
    learning_rate=0.01,
    epochs=5
)

print("Both experiments completed!")
