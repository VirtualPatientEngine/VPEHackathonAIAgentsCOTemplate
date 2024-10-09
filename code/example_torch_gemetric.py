# example_torch_geometric.py
# 
# This script demonstrates the use of Graph Neural Networks (GNNs) for regression
# tasks in drug discovery using the PyTorch Geometric library. The dataset used
# is MoleculeNet, specifically the "lipo" dataset. The script defines a GCN
# model, trains it on the dataset, and evaluates its performance using R2 score.
# 
# Reference: https://medium.com/@mulugetas/drug-discovery-and-graph-neural-networks-gnns-a-regression-example-fc738e0f11f3
# by Mulugeta Semework


import time
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import r2_score
from pathlib import Path

warnings.filterwarnings("ignore")

# Constants
EMBEDDING_SIZE = 64
NUM_GRAPHS_PER_BATCH = 64
NUM_EPOCHS = 300 # just for example ... in reality this should be much higher (>2000)
LEARNING_RATE = 0.0007
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scratch_path = Path(__file__).resolve().parent.parent / "scratch"


def load_dataset(root: str = str(scratch_path)) -> Dataset:
    """
    Load the MoleculeNet dataset.

    Args:
        root (str): Root directory for the dataset.

    Returns:
        Dataset: Loaded dataset.
    """
    dataset = MoleculeNet(root=root, name="lipo")
    return dataset


def split_dataset(dataset: Dataset, train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
    """
    Shuffle and split the dataset into training and testing sets.

    Args:
        dataset (Dataset): The dataset to be split.
        train_ratio (float): The ratio of the dataset to be used for training.

    Returns:
        Tuple[Dataset, Dataset]: Training and testing datasets.
    """
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * train_ratio)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    return train_dataset, test_dataset


def create_data_loaders(train_dataset: Dataset, test_dataset: Dataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The testing dataset.
        batch_size (int): Batch size for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class GCN(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super(GCN, self).__init__()
        
        # GCN layers
        self.initial_conv = GCNConv(input_dim, embedding_dim)
        self.conv_layers = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim) for _ in range(3)
        ])

        # Output layer
        self.out = nn.Linear(embedding_dim * 2, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial convolution layer
        x = F.tanh(self.initial_conv(x, edge_index))
        
        # Additional convolution layers
        for conv in self.conv_layers:
            x = F.tanh(conv(x, edge_index))
        
        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
        
        # Output layer
        out = self.out(x)
        return out, x


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): The data loader for training data.
        optimizer (optim.Optimizer): The optimizer for model training.
        loss_fn (nn.Module): The loss function.

    Returns:
        Tuple[float, float]: Average loss and R2 score for the epoch.
    """
    model.train()
    total_loss = 0
    total_r2 = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_r2 += r2_score(batch.y.cpu().detach().numpy(), pred.cpu().detach().numpy())
    
    avg_loss = total_loss / len(loader)
    avg_r2 = (total_r2 / len(loader)) * 100
    return avg_loss, avg_r2


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the testing dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): The data loader for testing data.

    Returns:
        float: Average R2 score on the testing dataset.
    """
    model.eval()
    total_r2 = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            total_r2 += r2_score(batch.y.cpu().numpy(), pred.cpu().numpy())
    
    avg_r2 = (total_r2 / len(loader)) * 100
    return avg_r2


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, num_epochs: int) -> None:
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for training data.
        optimizer (optim.Optimizer): The optimizer for model training.
        loss_fn (nn.Module): The loss function.
        num_epochs (int): Number of epochs to train the model.
    """
    print("\n======== Starting training ... =======\n")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, loss_fn)
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:>3} | Loss: {train_loss:.5f} | R2: {train_r2:.2f}%")
    
    elapsed = time.time() - start_time
    print("\nTraining done!\n")
    print(f"--- training took: {elapsed // 60:.0f} minutes ---")


def main():
    # Load dataset
    dataset = load_dataset()
    
    # Split dataset into training and testing
    train_dataset, test_dataset = split_dataset(dataset)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, NUM_GRAPHS_PER_BATCH)
    
    # Initialize model, optimizer, and loss function
    model = GCN(input_dim=dataset.num_features, embedding_dim=EMBEDDING_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    # Train the model
    train_model(model, train_loader, optimizer, loss_fn, NUM_EPOCHS)
    
    # Evaluate the model
    test_r2 = evaluate_model(model, test_loader)
    print(f"\nTest R2 Score: {test_r2:.2f}%")


if __name__ == "__main__":
    main()