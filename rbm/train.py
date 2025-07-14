import pandas as pd
import numpy as np
import random
from tinygrad import Tensor
from rbm import RBM
from matplotlib import pyplot as plt

# Load the datasets( https://github.com/zygmuntz/goodbooks-10k)
ratings = pd.read_csv('data/ratings.csv')
to_read = pd.read_csv('data/to_read.csv')
books = pd.read_csv('data/books.csv')

# Sort ratings by user_id and limit to 200,000 entries
temp = ratings.sort_values(by=['user_id'], ascending=True)
ratings = temp.iloc[:150000, :]

# Reset the index and create a 'List Index' column
ratings = ratings.reset_index(drop=True)
ratings['List Index'] = ratings.index
readers_group = ratings.groupby("user_id")

# Create a DataFrame for readers with their ratings
total = []
for readerID, curReader in readers_group:
    temp = np.zeros(len(ratings))

    for num, book in curReader.iterrows():
        temp[book['List Index']] = book['rating'] / 5.0

    total.append(temp)

# Shuffle the total list and split into training and validation sets
random.shuffle(total)
train = total[:1000]
valid = total[1000:]


# Define the number of hidden and visible units
hiddenUnits = 64
visibleUnits = len(ratings)

# Prepare the training data as batches
batch_size = 100
epochs = 5

# Create the RBM model
rbm = RBM(visible_units=visibleUnits, hidden_units=hiddenUnits, learning_rate=0.6, batch_size=batch_size)



errors = []
energy_train = []
energy_valid = []


for epoch in range(epochs):
    epoch_error = 0

    for i in range(0, len(train), batch_size):
        batch = train[i:i+batch_size] # Ensure the batch is of size batch_size
        batch = Tensor(batch).float()
        batch_error = rbm.contrastive_divergence(batch) # Perform contrastive divergence on the batch
        epoch_error += batch_error.item()

    # Calculate the average error for the epoch
    avg_epoch_error = epoch_error / len(train)
    errors.append(avg_epoch_error)

    # Calculate free energy for training and validation sets
    train_tensor = Tensor(train).float()
    energy_train.append(rbm.free_energy(train_tensor).mean().numpy())

    valid_tensor = Tensor(valid).float()
    energy_valid.append(rbm.free_energy(valid_tensor).mean().numpy())

    print(f'Epoch {epoch+1}/{epochs}, Error: {avg_epoch_error:.4f}')

# Plot the free energy over epochs
fig, ax = plt.subplots()
ax.plot(energy_train, label='train')
ax.plot(energy_valid, label='valid')
leg = ax.legend()
plt.xlabel("Epoch")
plt.ylabel("Free Energy")
plt.savefig("free_energy.png")
plt.show()
# Plot the error over epochs
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig("error.png")
plt.show()
