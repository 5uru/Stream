import numpy as np
import matplotlib.pyplot as plt
from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from model import ConvAutoEncoder
from sklearn.manifold import TSNE

# Load and normalize dataset
X_train, Y_train, X_test, Y_test = mnist()
X_train, X_test = X_train / 255.0, X_test / 255.0

batch_size = 512
latent_dim = 7
epochs = 50

model = ConvAutoEncoder(
        input_channels=1,
        input_size=28,
        hidden_channels=[32, 64, 128],
        latent_dim=latent_dim
)
optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=0.001)

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

# Training loop
for epoch in range(1, epochs + 1):
    Tensor.training = True
    for i in range(0, len(X_train), batch_size):
        X = X_train[i:i+batch_size]
        optimizer.zero_grad()
        recon_batch, _ = model(X)
        loss = mse_loss(recon_batch, X)
        loss.backward()
        optimizer.step()
    Tensor.training = False

    # Evaluate on test set
    test_loss = 0
    for i in range(0, len(X_test), batch_size):
        X = X_test[i:i+batch_size]
        recon, _ = model(X)
        test_loss += mse_loss(recon, X).item() * X.shape[0]
    test_loss /= len(X_test)
    print(f"Epoch {epoch}, Test Loss: {test_loss:.4f}")

# Visualize reconstructions
n_samples = 8
test_images = X_test[:n_samples]
test_recon, _ = model(test_images)

fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
for j in range(n_samples):
    axes[0, j].imshow(test_images[j, 0].numpy(), cmap='gray')
    axes[0, j].set_title(f"Image {j+1}")
    axes[0, j].axis('off')
    axes[1, j].imshow(test_recon[j, 0].detach().numpy(), cmap='gray')
    axes[1, j].set_title(f"Recon {j+1}")
    axes[1, j].axis('off')
plt.tight_layout()
plt.savefig("reconstructions.png")
plt.close()

# Extract latents for all test data in batches
latents = []
digits = []
for i in range(0, len(X_test), batch_size):
    X = X_test[i:i+batch_size]
    Y = Y_test[i:i+batch_size]
    _, latent = model(X)
    latents.append(latent.detach().numpy())
    digits.append(Y.numpy())
latents = np.concatenate(latents, axis=0)
digits = np.concatenate(digits, axis=0).astype(int)

# t-SNE visualization
class_names = {idx: f"Digit {idx}" for idx in np.unique(digits)}
tsne = TSNE(n_components=2, random_state=42)
latents_2d = tsne.fit_transform(latents)

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(class_names))]

plt.figure(figsize=(10, 8))
for class_id, class_label in class_names.items():
    idx = digits == class_id
    plt.scatter(
            latents_2d[idx, 0],
            latents_2d[idx, 1],
            color=colors[class_id],
            label=class_label,
            alpha=0.7,
            s=3
    )
plt.legend(title="Class")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
plt.title("MNIST Test Samples in Latent Space")
plt.grid(True)
plt.savefig("tsne_latent_space.png")