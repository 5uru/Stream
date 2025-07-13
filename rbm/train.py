from dataset import ml_100k
from model import RBM
from tinygrad import Tensor, nn
import numpy as np
import time
from tqdm import tqdm  # For progress tracking


def reconstruction_error(original, reconstructed):
    """Calculate mean squared error between original and reconstructed data."""
    return ((original - reconstructed) ** 2).mean()


def train_rbm():
    # Load and prepare dataset
    print("Loading MovieLens 100K dataset...")
    training_set, test_set = ml_100k()
    n_visible = training_set.shape[1]  # Number of movies (visible units)
    nb_users = training_set.shape[0]   # Number of users in training set
    print(f"Dataset loaded: {nb_users} users, {n_visible} movies")

    # Model hyperparameters
    n_hidden = 200       # Number of hidden units (latent features)
    batch_size = 100     # Number of samples per batch
    nb_epoch = 100       # Total training epochs
    learning_rate = 0.01 # Learning rate for optimizer
    momentum = 0.9       # Momentum for SGD optimizer
    weight_decay_factor = 0.0001  # L2 regularization strength

    # Initialize RBM model
    print(f"Initializing RBM with {n_visible} visible and {n_hidden} hidden units...")
    rbm = RBM(n_visible, n_hidden)

    # Setup optimizer
    optimizer = nn.optim.SGD([rbm.W, rbm.v_bias, rbm.h_bias],
                             lr=learning_rate,
                             momentum=momentum)

    # Enable training mode for the RBM
    Tensor.training = True

    # Track metrics
    history = {
            'train_loss': [],
            'train_error': [],
            'epoch_time': []
    }

    # Training loop
    print("Starting training...")
    for epoch in range(1, nb_epoch + 1):
        epoch_start = time.time()
        train_error = []
        train_loss = []

        # Process data in batches
        batch_indices = range(0, nb_users - batch_size, batch_size)
        for id_user in tqdm(batch_indices, desc=f"Epoch {epoch}/{nb_epoch}"):
            # Get batch of training data
            sample_data = training_set[id_user:id_user+batch_size]

            # Forward pass: v is original data, v1 is reconstruction
            v, v1 = rbm(sample_data)

            # Calculate contrastive divergence loss
            # (difference between free energies of data and reconstruction)
            loss = Tensor.abs(rbm.free_energy(v) - rbm.free_energy(v1))

            # Add L2 weight regularization to prevent overfitting
            weight_decay = weight_decay_factor * (rbm.W * rbm.W).sum()
            loss = loss + weight_decay

            # Backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record metrics
            train_error.append(reconstruction_error(v.numpy(), v1.numpy()))
            train_loss.append(loss.numpy())

        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start
        mean_loss = np.mean(train_loss)
        mean_error = np.mean(train_error)

        # Store metrics
        history['train_loss'].append(mean_loss)
        history['train_error'].append(mean_error)
        history['epoch_time'].append(epoch_time)

        print(f'Epoch: {epoch}/{nb_epoch} | '
              f'Loss: {mean_loss:.6f} | '
              f'Reconstruction Error: {mean_error:.6f} | '
              f'Time: {epoch_time:.2f}s')

        # Evaluate on test set every 10 epochs
        if epoch % 10 == 0:
            test_recon_error = evaluate_model(rbm, test_set)
            print(f'Test Reconstruction Error: {test_recon_error:.6f}')

    print("Training completed!")
    return rbm, history


def evaluate_model(model, test_data):
    """Evaluate model performance on test data."""
    v, v1 = model(test_data)
    return reconstruction_error(v.numpy(), v1.numpy())


if __name__ == "__main__":
    # Entry point when script is run directly
    trained_model, training_history = train_rbm()

    # Save model weights (implementation depends on framework)
    # trained_model.save_weights('rbm_model.h5')

    print("Script execution completed.")