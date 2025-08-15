import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def create_simple_dataloader(features, labels, batch_size, shuffle=True):
    """
    Creates a simple PyTorch DataLoader from features and labels.
    Based on pytorch documentation
    Args:
        features (np.ndarray or list): The input data (e.g., your X). Input
        labels (np.ndarray or list): The target data (e.g., your y). Predicting adjusted close price.
        batch_size (int): The number of samples per batch.
        shuffle (bool, optional): shuffle the data

    Returns:
        A torch.utils.data.DataLoader object.
    """
    #pytorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # A TensorDataset wraps Tensors and provides an easy way to access them.
    # It's a simple dataset that combines your features and labels.
    dataset = TensorDataset(features_tensor, labels_tensor)

    # Finally, the DataLoader takes the dataset and handles batching,
    # shuffling, and data loading for you.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader