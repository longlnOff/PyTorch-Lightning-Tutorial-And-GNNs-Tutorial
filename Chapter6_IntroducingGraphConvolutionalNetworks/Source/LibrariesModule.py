import os
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub, Planetoid
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import lightning as L
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.nn import GCNConv
from lightning.pytorch import Trainer, seed_everything
seed_everything(42, workers=True)