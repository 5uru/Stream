from tinygrad import Tensor, nn
class AlexNet:

    def __init__(self, num_classes=10):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.dense1 = nn.Linear(256 * 4 * 4, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu()
        x = x.max_pool2d(kernel_size=(2, 2), stride=2)

        x = self.conv2(x).relu()
        x = x.max_pool2d(kernel_size=(2,2), stride=2)

        x = self.conv3(x).relu()

        x = self.conv4(x).relu()

        x = self.conv5(x).relu()
        x = x.max_pool2d(kernel_size=(2, 2), stride=2)


        # Flatten the tensor
        x = x.reshape(x.shape[0], -1)

        x = x.dropout(0.5)
        x = self.dense1(x).relu()
        x = x.dropout(0.5)
        x = self.dense2(x).relu()
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    model = AlexNet(num_classes=10)
    x = Tensor.randn(1, 3, 32, 32)
    y = model(x)
    print(f"Input tensor size: {x.shape}")
    print(f"Output tensor size: {y.shape}")
    print(f"Output tensor: {y.numpy()}")