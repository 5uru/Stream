from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD
import numpy as np

# Force tinygrad to use float32 instead of float64
import os
os.environ["FLOAT32"] = "1"

class RBM:
    """Restricted Boltzmann Machine implemented with tinygrad."""

    def __init__(self, n_visible, n_hidden, k=5):
        """
        Initialize the RBM model.

        Args:
            n_visible: Number of visible units (input dimension)
            n_hidden: Number of hidden units
            k: Number of Gibbs sampling steps for contrastive divergence
        """
        # Initialize weights with small random values
        self.W = Tensor.randn((n_visible, n_hidden)) * np.sqrt(2/(n_visible + n_hidden)) # Xavier initialization
        # Biases for visible and hidden layers
        self.v_bias = Tensor.zeros(n_visible)
        self.h_bias = Tensor.zeros(n_hidden)
        # Number of Gibbs sampling steps
        self.k = k

    def sample_from_p(self, p):
        """
        Sample binary values from given probabilities.

        Args:
            p: Probability tensor

        Returns:
            Binary sample (0 or 1)
        """
        # Generate random values and compare with probabilities
        return  (Tensor.rand(p.shape) < p).float()

    def v_to_h(self, v):
        """
        Compute hidden probabilities and samples given visible units.

        Args:
            v: Visible units tensor

        Returns:
            Tuple of (hidden probabilities, hidden samples)
        """
        # Calculate activation of hidden units
        wx_b = v.matmul(self.W).add(self.h_bias)
        # Apply sigmoid to get probabilities
        p_h = wx_b.sigmoid()
        # Sample binary values
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        """
        Compute visible probabilities and samples given hidden units.

        Args:
            h: Hidden units tensor

        Returns:
            Tuple of (visible probabilities, visible samples)
        """
        # Calculate activation of visible units
        wx_b = h.matmul(self.W.transpose(0, 1)).add(self.v_bias)
        # Apply sigmoid to get probabilities
        p_v = wx_b.sigmoid()
        # Sample binary values
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def __call__(self, v):
        """
        Perform forward pass through the network.

        Args:
            v: Input visible units

        Returns:
            Tuple of (original input, reconstructed visible units)
        """
        # Compute hidden activations
        _, h1 = self.v_to_h(v)

        # Perform k steps of Gibbs sampling
        h_ = h1
        for _ in range(self.k):
            _, v_ = self.h_to_v(h_)
            _, h_ = self.v_to_h(v_)

        # Return original input and reconstruction
        return v, v_

    def free_energy(self, v):
        """
        Calculate free energy of the visible units.

        Args:
            v: Visible units tensor

        Returns:
            Free energy scalar
        """
        # Visible bias term contribution
        vbias_term = (v * self.v_bias).sum(1)
        # Hidden units contribution
        wx_b = v.linear(self.W, self.h_bias)
        hidden_term = wx_b.clip(-20, 20).exp().add(1).log().sum(1)
        # Return mean negative free energy
        return (-hidden_term - vbias_term).mean()


if __name__ == "__main__":
    # Model configuration
    n_visible = 6
    n_hidden = 3
    batch_size = 10
    epochs = 100

    # Create RBM model
    rbm = RBM(n_visible, n_hidden)

    # Create optimizer
    opt = SGD([rbm.W, rbm.v_bias, rbm.h_bias], lr=0.01)

    # Enable training mode
    Tensor.training = True

    # Training loop
    print("Training RBM model...")
    for epoch in range(epochs):
        # Generate random batch (in real scenario, this would be your dataset)
        batch = Tensor(np.random.binomial(1, 0.5, (batch_size, n_visible)).astype(np.float32))

        # Forward pass
        v, v_recon = rbm(batch)

        # Calculate contrastive divergence loss
        # Minimizing this drives the model to reconstruct the input data
        loss = rbm.free_energy(v) - rbm.free_energy(v_recon)

        # Backward pass and optimization
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.numpy()}")

    # Test the trained model with a sample input
    test_input = Tensor(np.random.binomial(1, 0.5, (1, n_visible)).astype(np.float32))
    v, v_recon = rbm(test_input)

    print("\nTest Results:")
    print(f"Input: {test_input.numpy().flatten()}")
    print(f"Reconstruction: {v_recon.numpy().flatten()}")