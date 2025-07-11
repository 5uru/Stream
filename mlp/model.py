from tinygrad import Tensor, nn

class MLP:
    def __init__(self):
        self.dense1 = nn.Linear(1*28*28, 784)
        self.dense2 = nn.Linear(784, 250)
        self.dense3 = nn.Linear(250, 100)
        self.dense4 = nn.Linear(100, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x).relu()
        x = self.dense2(x).relu()
        x = self.dense3(x).relu()
        x = self.dense4(x)
        return x



if __name__ == "__main__":
    x = Tensor.randn((1, 1, 28, 28))
    model = MLP()
    y = model(x)
    print(y.shape)  # Should print: (1, 10)
    print(y.numpy())  # Print the output tensor
