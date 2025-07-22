from tinygrad import Tensor, nn

class RNN:
    """
    Recurrent Neural Network for sequence classification using LSTM cells.

    Processes variable-length sequences and outputs class probabilities.
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        """
        Initialize the RNN model.
        """
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers

        # LSTM cell for sequence processing
        self.rnn = nn.LSTMCell(input_size, hidden_dim, n_layers)
        # Final classification layer
        self.fc1 = nn.Linear(hidden_dim, output_size)

    def __call__(self, x, lengths, h0=None):
        """
        Forward pass through the RNN.
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initialize hidden state and cell state if not provided
        if h0 is None:
            h = Tensor.zeros(batch_size, self.hidden_dim)
            c = Tensor.zeros(batch_size, self.hidden_dim)
        else:
            h, c = h0

        # Process each time step through the LSTM
        outputs = []
        for t in range(seq_len):
            h, c = self.rnn(x[:, t, :], (h, c))
            outputs.append(h)

        # Stack outputs to get the full sequence
        r_out = Tensor.stack(outputs, dim=1)

        # Extract final hidden state for each sequence based on its length
        final_states = []
        for i in range(batch_size):
            # Get the last valid timestep (accounting for 0-indexing)
            idx = min(lengths[i] - 1, seq_len - 1)
            final_states.append(r_out[i, idx, :])

        # Stack final states and apply classification layer
        aux = Tensor.stack(final_states)

        # Apply classification layer with dropout regularization
        out = self.fc1(aux).dropout(0.5).log_softmax()
        return out


if __name__ == "__main__":
    # Test the RNN model
    rnn = RNN(input_size=100, output_size=3, hidden_dim=50, n_layers=1)

    # Create sample batch: 32 sequences, each with 10 timesteps and 100 features
    x = Tensor.randn(32, 10, 100)

    # Test with variable sequence lengths
    lengths = [10, 8, 9, 7] + [10] * 28

    # Run forward pass
    output = rnn(x, lengths)
    print(f"Output shape: {output.shape}")  # Should be (32, 3)clear