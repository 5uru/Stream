from tinygrad import Tensor, nn

class ConvEncoder:
    """
    Convolutional Encoder for AutoEncoder.
    Encodes input images into a latent vector.
    """
    def __init__(self, input_channels=1, input_size=28, hidden_channels=None, latent_dim=10):
        # Default hidden channels if not provided
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Output feature map size after convolutions (for MNIST: 128x4x4)
        self.feature_channels, self.feature_height, self.feature_width = 128, 4, 4

        # Convolutional layers with batch normalization
        self.conv_layer1 = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.conv_layer2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.conv_layer3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])

        # Fully connected layer to latent space
        self.fc = nn.Linear(self.feature_channels * self.feature_height * self.feature_width, latent_dim)

    def __call__(self, x):
        # Forward pass through conv layers and batch norm with leaky relu
        x = self.conv_layer1(x)
        x = self.bn1(x).leaky_relu(0.2)
        x = self.conv_layer2(x)
        x = self.bn2(x).leaky_relu(0.2)
        x = self.conv_layer3(x)
        x = self.bn3(x).leaky_relu(0.2)
        # Flatten and project to latent space
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class ConvDecoder:
    """
    Convolutional Decoder for AutoEncoder.
    Decodes latent vector back to image.
    """
    def __init__(self, input_channels=1, hidden_channels=[32, 64, 128], latent_dim=10,
                 feature_channels=128, feature_height=4, feature_width=4):
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.feature_channels = feature_channels
        self.feature_height = feature_height
        self.feature_width = feature_width

        # Linear layer to expand latent vector to feature map
        flattened_dim = feature_channels * feature_height * feature_width
        self.fc = nn.Linear(latent_dim, flattened_dim)

        # Prepare reversed hidden channels for deconvolution
        in_channels = hidden_channels[-1]
        out_channels = list(reversed(hidden_channels[:-1]))

        # Transposed convolutional layers with batch normalization
        self.conv_layer1 = nn.ConvTranspose2d(in_channels, out_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv_layer2 = nn.ConvTranspose2d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        # Final layer to reconstruct image
        self.conv_transpose = nn.ConvTranspose2d(out_channels[1], input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def __call__(self, x, target_height=None, target_width=None):
        # Expand latent vector and reshape to feature map
        x = self.fc(x).leaky_relu()
        x = x.reshape(x.shape[0], self.feature_channels, self.feature_height, self.feature_width)
        # Pass through transposed conv layers
        x = self.conv_layer1(x)
        x = self.bn1(x).leaky_relu(0.2)
        x = self.conv_layer2(x)
        x = self.bn2(x).leaky_relu(0.2)
        reconstruction = self.conv_transpose(x).sigmoid()

        # Optionally crop output to match target size
        if target_height is not None and target_width is not None:
            h_diff = reconstruction.size(2) - target_height
            w_diff = reconstruction.size(3) - target_width
            if h_diff >= 0 and w_diff >= 0:
                h_start = h_diff // 2
                w_start = w_diff // 2
                reconstruction = reconstruction[:, :, h_start:h_start + target_height, w_start:w_start + target_width]
        return reconstruction


class ConvAutoEncoder:
    """
    Full Convolutional AutoEncoder: combines encoder and decoder.
    """
    def __init__(self, input_channels=1, input_size=28, hidden_channels=[32, 64, 128], latent_dim=10):
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Instantiate encoder and decoder
        self.encoder = ConvEncoder(input_channels, input_size, hidden_channels, latent_dim)
        self.decoder = ConvDecoder(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                latent_dim=latent_dim,
                feature_channels=self.encoder.feature_channels,
                feature_height=self.encoder.feature_height,
                feature_width=self.encoder.feature_width
        )

    def __call__(self, x):
        # Store original input shape for cropping
        self.original_height = x.size(2)
        self.original_width = x.size(3)
        # Encode and decode
        latent = self.encode(x)
        reconstruction = self.decode(latent, target_height=self.original_height, target_width=self.original_width)
        return reconstruction, latent

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, target_height=None, target_width=None):
        # Use stored dimensions if not provided
        if target_height is None and hasattr(self, 'original_height'):
            target_height = self.original_height
        if target_width is None and hasattr(self, 'original_width'):
            target_width = self.original_width
        return self.decoder(z, target_height=target_height, target_width=target_width)


if __name__ == "__main__":
    # Example usage and shape checks
    encoder = ConvEncoder(input_channels=1, input_size=28, latent_dim=10)
    x = Tensor.randn(16, 1, 28, 28)  # Batch of 16 grayscale images
    encoded = encoder(x)
    print("Encoded shape:", encoded.shape)  # (16, latent_dim)

    z = Tensor.randn(16, 10)  # Batch of 16 latent vectors
    decoder = ConvDecoder(input_channels=1, latent_dim=10)
    print("Decoder output shape:", decoder(z).shape)  # (16, 1, 28, 28)

    autoencoder = ConvAutoEncoder(input_channels=1, input_size=28, latent_dim=10)
    reconstruction, latent = autoencoder(x)
    print("Reconstruction shape:", reconstruction.shape)  # (16, 1, 28, 28)
    print("Latent shape:", latent.shape)  # (16, latent_dim)