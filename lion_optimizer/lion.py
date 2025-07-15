from cnn import Model
from tinygrad import Tensor, nn



class Lion:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = [Tensor.zeros_like(p) for p in self.params]

    def step(self, gradients):
        for i, (p, g, m) in enumerate(zip(self.params, gradients, self.momentum)):
            p *= (1 - self.lr * self.weight_decay)

            direction = self.beta1 * m + (1 - self.beta1) * g
            p -= self.lr * Tensor.sign(direction)
            self.params[i] = p
            self.momentum[i] = self.beta2 * m + (1 - self.beta2) * g

        return self.params



if __name__ == "__main__":

    cnn = Model()
    x = Tensor.randn(1, 1, 28, 28)
    params = nn.state.get_parameters(cnn)
    optimizer = Lion(params, lr=0.001)
    Tensor.training = True  # Enable training mode
    y = cnn(x)
    target = Tensor([5])  # Example target class
    loss = y.sparse_categorical_crossentropy(target)
    gradients = loss.gradient(*params)
    new_params = optimizer.step(gradients)




