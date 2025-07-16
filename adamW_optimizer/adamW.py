from tinygrad import Tensor

class AdamW:
    """
    Adam optimizer implementation

    Based on the article : https://arxiv.org/pdf/1711.05101v3
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentums = [Tensor.zeros_like(p) for p in self.params]
        self.velocities = [Tensor.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, gradients):
        """
        Performs an optimization step with the AdamW algorithm
        """
        self.t += 1 # Increment the time step

        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            if self.weight_decay != 0:
                # Apply gradient regularization
                grad = grad + self.weight_decay * param

            # Recovers the previous moment and velocity
            momentum = self.momentums[i]
            velocity = self.velocities[i]

            # updates the moment and velocity
            new_momentum = self.beta1 * momentum + (1 - self.beta1) * grad
            new_velocity = self.beta2 * velocity + (1 - self.beta2) * (grad ** 2)

            # bias correction

            momentum_hat = new_momentum / (1 - self.beta1 ** self.t)
            velocity_hat = new_velocity / (1 - self.beta2 ** self.t)


            # Calculates update (weighted combination of moment and current gradient)
            update = (momentum_hat / (Tensor.sqrt(velocity_hat) + self.eps)) + self.weight_decay * param
            # Explicit parameter update
            new_param = param - self.lr * update
            # Explicit parameter update
            param.assign(new_param)

            # Update of the moment with beta2 and force its realization
            self.momentums[i].assign(new_momentum.realize())
            self.velocities[i].assign(new_velocity.realize())


