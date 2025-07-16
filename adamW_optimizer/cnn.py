from tinygrad import Tensor, nn

class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2,2))
        x = self.l2(x).relu().max_pool2d((2,2))
        return self.l3(x.flatten(1).dropout(0.5))


if __name__ == "__main__":
    import numpy as np
    x = Tensor(np.random.randn(1, 1, 28, 28).astype(np.float32))
    model = Model()
    y = model(x)
    print(y.shape)