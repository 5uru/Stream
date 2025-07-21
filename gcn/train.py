import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor, nn, TinyJit
from model import GCN

# Create adjacency matrix from edge index
def create_adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for i, j in edge_index:
        # Adjust for 1-indexing in dataset
        adj[i-1, j-1] = 1
        adj[j-1, i-1] = 1  # Undirected graph

    # Add self-loops
    return Tensor(adj) + Tensor.eye(num_nodes)

# Load data
data = np.loadtxt("out.ucidata-zachary", dtype=int, skiprows=2)
num_nodes = 34
features_dim = 2

# Create adjacency matrix and node features
adj = create_adj(data, num_nodes)
features = Tensor.eye(num_nodes)  # Use one-hot node encodings as features

# Create labels (binary community classification)
community1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]
labels = Tensor([0 if i in community1 else 1 for i in range(num_nodes)])

# Initialize model and optimizer
model = GCN(in_features=features_dim, out_features=16, n_hidden=16, n_classes=2, n_nodes=num_nodes)
optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=0.01)

# Training step function with proper return
def step():
    Tensor.training = True  # Enable dropout
    optimizer.zero_grad()
    outputs = model(adj)
    loss = outputs.sparse_categorical_crossentropy(labels)
    loss.backward()
    optimizer.step()
    return loss, outputs

jit_step = TinyJit(step)

print("Starting training...")

# Training loop
loss_history = []
accuracy_history = []
epochs = 100
best_accuracy = 0

for epoch in range(epochs):
    loss, outputs = jit_step()
    loss_val = loss.item()
    loss_history.append(loss_val)

    # Calculate accuracy for testing
    predictions = outputs.argmax(axis=1)
    accuracy = (predictions == labels).mean().item()
    accuracy_history.append(accuracy)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d}, Loss: {loss_val:.4f}, Accuracy: {accuracy:.4f}")

    # Save best model state (simple testing)
    if accuracy > best_accuracy:
        best_accuracy = accuracy

print(f"Training complete. Best accuracy: {best_accuracy:.4f}")

# Prepare visualization data
radius = 1
angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
x_positions = radius * np.cos(angles)
y_positions = radius * np.sin(angles)

# Visualize graph (optimized to avoid duplicate plotting)
plt.figure(figsize=(8, 8))
adj_numpy = adj.numpy()

# Plot edges (only once)
for i in range(num_nodes):
    for j in range(i+1, num_nodes):  # Avoid duplicates
        if adj_numpy[i][j] != 0:
            plt.plot([x_positions[i], x_positions[j]],
                     [y_positions[i], y_positions[j]], 'b-', alpha=0.3)

# Plot nodes with different colors based on community
for i in range(num_nodes):
    color = 'ro' if labels[i].item() == 0 else 'bo'  # Red for community 1, blue for 2
    plt.plot(x_positions[i], y_positions[i], color, markersize=6)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graph Communities Visualization')
plt.axis('equal')
plt.savefig('graph_visualization.png')
plt.show()

# Visualize training metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()