from tinygrad import Tensor

class Lion:
    """
    Lion optimizer implementation

    Based on the article : https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.momentums = [Tensor.zeros_like(p) for p in self.params]

    def step(self, gradients):
        """
        Performs an optimization step with the Lion algorithm
        """
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            if self.weight_decay != 0:
                # Apply gradient regularization
                grad = grad + self.weight_decay * param

            # Recovers the previous moment
            momentum = self.momentums[i]

            # Calculates update (weighted combination of moment and current gradient)
            update = self.beta1 * momentum + (1 - self.beta1) * grad

            # Explicitly checks that the update is non-zero
            update_sign = Tensor.sign(update)

            # Force tensor realization before updating
            update_sign = update_sign.realize()

            # Explicit parameter update
            new_param = param - self.lr * update_sign
            # Explicit parameter update
            param.assign(new_param)

            # Update of the moment with beta2 and force its realization
            new_momentum = self.beta2 * momentum + (1 - self.beta2) * grad
            self.momentums[i].assign(new_momentum.realize())


