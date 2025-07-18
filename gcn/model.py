
from tinygrad import Tensor, nn

class GCNLayer:
    def __init__(self, in_features, out_features):
        self.dense = nn.Linear(in_features, out_features)

    def __call__(self, x, adj_matrix):
        # Identity matrix
        identity = Tensor.eye(adj_matrix.shape[0])

        # Add self-loop
        adj_matrix = adj_matrix + identity

        # Normalize adjacency matrix
        degree_matrix = adj_matrix.sum(axis=1)

        # Traitement des valeurs nulles pour éviter les divisions par zéro
        # Créons une matrice inverse sans utiliser d'indexation booléenne
        degree_matrix_inv_sqrt = Tensor.where(
                degree_matrix > 0,  # Condition
                degree_matrix.pow(-0.5),  # Si vrai
                Tensor.zeros_like(degree_matrix)  # Si faux
        )

        # Utilisation de diagflat() au lieu de diag()
        adj_matrix = Tensor.matmul(Tensor.matmul(degree_matrix_inv_sqrt.diag(), adj_matrix), degree_matrix_inv_sqrt.diag())

        x = Tensor.matmul(adj_matrix, x)
        x = self.dense(x)
        return x.tanh()

class GCN:
    def __init__(self, in_features, out_features):
        self.layer = GCNLayer(in_features, out_features)

    def __call__(self, x, adj_matrix):
        return self.layer(x, adj_matrix)

if __name__ == "__main__":
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    adj_matrix = Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    layer = GCNLayer(2, 2)
    output = layer(x, adj_matrix)
    print(output.shape)  # Should print (3, 2)
    print(output.numpy())  # Should print the output of the GCN layer