from tinygrad import Tensor, nn

class VGG:
    def __init__(self, num_classes=10):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(512 , 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu().max_pool2d()

        x = self.conv2(x).relu().max_pool2d()

        x = self.conv3(x).relu()
        x = self.conv4(x).relu().max_pool2d()

        x = self.conv5(x).relu()
        x = self.conv6(x).relu().max_pool2d()

        x = self.conv7(x).relu()
        x = self.conv8(x).relu().max_pool2d()

        #x = x.avg_pool2d()
        x = x.reshape((x.shape[0], -1))  # Flatten the tensor
        x = self.fc1(x).relu().dropout(0.5)
        x = self.fc2(x).relu().dropout(0.5)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    x = Tensor.randn((1, 3, 32, 32))  # Example input shape for VGG
    model = VGG(num_classes=10)
    y = model(x)
    print(y.shape)  # Should print: (1, 10)
    print(y.numpy())  # Print the output tensor