from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from adamW import AdamW
from cnn import Model
import matplotlib.pyplot as plt


# Data loading and normalization
X_train, Y_train, X_test, Y_test = mnist()
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN model definition
model = Model()
params = nn.state.get_parameters(model) # model parameters
# Lion optimizer configuration
optim = AdamW(
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0
)

batch_size = 128

loss_ = []
acc_ = []
# Training
for step_idx in range(5000):


    # Batch preparation
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]

    # Forward pass
    out = model(X)
    loss = out.sparse_categorical_crossentropy(Y)

    # Backward pass
    gradients = loss.gradient(*params)

    # Optimisation
    optim.step(gradients)
    loss_.append(loss.item())

    if step_idx%100 == 0:
        # Évaluation
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        acc_.append(acc)
        print(f"Epoch {step_idx:4d}, loss {loss.item():.2f}, accuracy {acc*100.:.2f}%")
        print("-" * 50)

# plot results and save on png
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_, label='Loss')
plt.title('Loss during training')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc_, label='Accuracy', color='orange')
plt.title('Accuracy during training')
plt.xlabel('Steps (each 100)')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

