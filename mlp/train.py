from model import MLP
from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
import timeit
from tinygrad import TinyJit
import matplotlib.pyplot as plt

# Dataset
X_train, Y_train, X_test, Y_test = mnist()
X_train = X_train / 255 # Convert to Tensor and float
X_test = X_test / 255 # Convert to Tensor and float

# Print dataset statistics
print("MNIST Dataset Statistics:")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")

print("Sample:", X_train[0].numpy())
# Print first few samples from the training set
for i in range(5):
    print(f"Sample {i+1}: {X_train[i].shape}, Label: {Y_train[i].numpy()}")

num_classes = 10  # MNIST has 10 classes
# Initialize the MLP model
model = MLP()

# Optimize the model using Adam optimizer
optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=0.001)

# Step for training the model
batch_size = 128
def step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optimizer.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optimizer.step()
    return loss

jit_step = TinyJit(step)

print("Starting training...")

# Timing the training step
start_time = timeit.default_timer()
loss_history = []
acc_history = []
for step in range(7000):
    loss = jit_step()
    loss_history.append(loss.item())
    if step%100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        acc_history.append(acc)
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%, time {timeit.default_timer() - start_time:.2f}s")
print(f"Total training time: {timeit.default_timer() - start_time:.2f}s")
# Plotting the loss and accuracy and save on png
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss')
plt.title('Loss over Training Steps')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc_history, label='Accuracy', color='orange')
plt.title('Accuracy over Training Steps')
plt.xlabel('Training Steps (every 100 steps)')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
print("Training completed. Loss and accuracy history saved to 'training_history.png'.")
