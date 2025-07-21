from tinygrad import Tensor, nn
from typing import Tuple

class GCNLayer:
    """Graph Convolutional Network Layer implementation."""

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize a GCN layer.
        """
        self.dense = nn.Linear(in_features, out_features)

    def normalize_adjacency_matrix(self, adj_matrix: Tensor) -> Tensor:
        """
        Create a normalized adjacency matrix with self-loops.
        """
        # Add self-loops
        identity = Tensor.eye(adj_matrix.shape[0])
        adj_with_loops = adj_matrix + identity

        # Calculate degree matrix
        degrees = adj_with_loops.sum(axis=1)

        # Handle zero values to avoid division by zero
        degree_inv_sqrt = Tensor.where(
                degrees > 0,
                degrees.pow(-0.5),
                Tensor.zeros_like(degrees)
        )

        # Create diagonal matrices and compute D^(-1/2) * A * D^(-1/2)
        deg_inv_sqrt_diag = degree_inv_sqrt.diag()
        return Tensor.matmul(Tensor.matmul(deg_inv_sqrt_diag, adj_with_loops), deg_inv_sqrt_diag)

    def __call__(self, x: Tensor, adj_matrix: Tensor) -> Tensor:
        """
        Forward pass of the GCN layer.
        """
        # Normalize adjacency matrix
        norm_adj = self.normalize_adjacency_matrix(adj_matrix)

        # Aggregate neighborhood information
        x = Tensor.matmul(norm_adj, x)

        # Apply linear transformation and activation
        return self.dense(x).tanh()


class GCN:
    """Graph Convolutional Network model."""

    def __init__(self, in_features: int, out_features: int, n_hidden: int, n_classes: int, n_nodes: int):
        """
        Initialize a GCN model.

        Args:
            in_features: Dimension of input features
            out_features: Dimension of output features
            n_hidden: Size of hidden layer
            n_classes: Number of output classes
            n_nodes: Number of nodes in the graph
        """
        self.features = nn.Embedding(n_nodes, in_features)
        self.layer1 = GCNLayer(in_features, n_hidden)
        self.layer2 = GCNLayer(n_hidden, out_features)
        self.classifier = nn.Linear(out_features, n_classes)

    def __call__(self, adj_matrix: Tensor) -> Tensor:
        """
        Forward pass of the GCN model.
        """
        # Get initial node features from embedding
        node_features = self.features.weight

        # Apply GCN layers
        hidden_features = self.layer1(node_features, adj_matrix)
        output_features = self.layer2(hidden_features, adj_matrix)

        # Final classification
        return self.classifier(output_features)


if __name__ == "__main__":
    # Test GCN layer
    features = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    adj_matrix = Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    layer = GCNLayer(2, 2)
    layer_output = layer(features, adj_matrix)
    print(f"GCN Layer output shape: {layer_output.shape}")
    print(f"GCN Layer output:\n{layer_output.numpy()}")

    # Test full GCN model
    gcn = GCN(in_features=2, out_features=2, n_hidden=2, n_classes=2, n_nodes=3)
    model_output = gcn(adj_matrix)
    print(f"\nGCN Model output shape: {model_output.shape}")
    print(f"GCN Model output:\n{model_output.numpy()}")