import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tinygrad import Tensor, nn
from model import RNN
from functools import lru_cache
import time
import gc
from tqdm.auto import tqdm

# Configuration
CONFIG = {
        "batch_size": 32,  # Increased batch size for faster training
        "epochs": 50,
        "learning_rate": 0.001,  # Slightly higher learning rate
        "hidden_dim": 32,  # Increased hidden dimension
        "n_layers": 1,
        "early_stopping_patience": 5,
        "embedding_dim": 300,
        "output_classes": 2,
}

# Load and cache spaCy model
@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except:
        print("Installing spaCy model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

nlp = get_nlp()

# Cached token processing
@lru_cache(maxsize=2048)
def get_normalized_tokens(text):
    doc = nlp(text)
    return [w for w in doc if w.has_vector and not w.is_punct and not w.is_stop]

def load_data():
    """Load and preprocess data."""
    with open("Sentences_AllAgree.txt", "r", encoding="ISO-8859-1") as sentences:
        lines = sentences.readlines()

    phrases = [line.split('@')[0] for line in lines]
    opinions = [line.split('@')[1] for line in lines]

    # Map opinions directly to binary labels (negative=1, else=0)
    labels = np.array([1 if opinion == 'negative\n' else 0 for opinion in opinions])

    # Split data
    idx_data = np.arange(len(phrases))
    idx_train, idx_test, y_train, y_test = train_test_split(idx_data, labels, test_size=0.2, random_state=0)
    idx_train, idx_val, y_train, y_val = train_test_split(idx_train, y_train, test_size=0.2, random_state=0)

    print(f"Training samples: {len(idx_train)}")
    print(f"Validation samples: {len(idx_val)}")
    print(f"Test samples: {len(idx_test)}")

    return phrases, idx_train, idx_val, idx_test, y_train, y_val, y_test

# Precompute document lengths and vectors cache
def precompute_lengths(phrases, indices):
    """Precompute sequence lengths for all documents in indices."""
    print("Precomputing document lengths...")
    return [len(get_normalized_tokens(phrases[i])) for i in tqdm(indices)]

# Process batch with vectorization
def process_batch(phrases, indices, lengths):
    """Process batch with efficient vectorization."""
    batch_docs = [get_normalized_tokens(phrases[i]) for i in indices]

    # Dynamic padding - use max in batch or actual length, whichever is smaller
    max_batch_len = max((len(doc) for doc in batch_docs), default=1)
    max_batch_len = min(max_batch_len, max(lengths)) if lengths else 1

    # Pre-allocate array for better performance
    batch_array = np.zeros((len(batch_docs), max_batch_len, CONFIG["embedding_dim"]), dtype=np.float32)

    for i, doc in enumerate(batch_docs):
        for j, token in enumerate(doc[:max_batch_len]):
            if token.has_vector and len(token.vector) == CONFIG["embedding_dim"]:
                batch_array[i, j] = token.vector

    return Tensor(batch_array)

def evaluate_model(model, phrases, indices, doc_lengths, labels, batch_size):
    """Evaluate model on given dataset."""
    correct_preds = 0
    total_loss = 0.0
    num_batches = int(np.ceil(len(indices) / batch_size))

    Tensor.training = False

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(indices))

        batch_indices = indices[start_idx:end_idx]
        batch_lengths = [doc_lengths[idx] for idx in range(start_idx, end_idx)]
        batch_y = labels[start_idx:end_idx]

        x_batch = process_batch(phrases, batch_indices, batch_lengths)
        y_tensor = Tensor(batch_y, requires_grad=False)

        # Forward pass
        output = model(x_batch, batch_lengths)
        loss = output.sparse_categorical_crossentropy(y_tensor)

        # Get predictions
        predictions = output.numpy().argmax(axis=1)
        correct_preds += np.sum(predictions == batch_y)
        total_loss += loss.numpy()

        del x_batch, output, loss

    return total_loss / num_batches, correct_preds / len(labels)

def train_model():
    """Main training function."""
    start_time = time.time()

    # Load data
    phrases, idx_train, idx_val, idx_test, y_train, y_val, y_test = load_data()

    # Convert indices to numpy arrays for faster slicing
    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)

    # Precompute document lengths
    train_docs_len = precompute_lengths(phrases, idx_train)
    val_docs_len = precompute_lengths(phrases, idx_val)
    test_docs_len = precompute_lengths(phrases, idx_test)

    # Initialize model
    model = RNN(
            input_size=CONFIG["embedding_dim"],
            output_size=CONFIG["output_classes"],
            hidden_dim=CONFIG["hidden_dim"],
            n_layers=CONFIG["n_layers"]
    )

    # Initialize optimizer with learning rate scheduling
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=CONFIG["learning_rate"])

    # Arrays to store metrics
    train_losses = np.zeros(CONFIG["epochs"])
    train_accs = np.zeros(CONFIG["epochs"])
    val_losses = np.zeros(CONFIG["epochs"])
    val_accs = np.zeros(CONFIG["epochs"])

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()

        # Shuffle training data
        shuffle_indices = np.random.permutation(len(idx_train))
        shuffled_idx_train = idx_train[shuffle_indices]
        shuffled_train_lens = [train_docs_len[i] for i in shuffle_indices]
        shuffled_y_train = y_train[shuffle_indices]

        # Training phase
        Tensor.training = True
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        num_batches = int(np.ceil(len(shuffled_idx_train) / CONFIG["batch_size"]))

        for i in range(num_batches):
            start_idx = i * CONFIG["batch_size"]
            end_idx = min((i + 1) * CONFIG["batch_size"], len(shuffled_idx_train))

            batch_indices = shuffled_idx_train[start_idx:end_idx]
            batch_lengths = shuffled_train_lens[start_idx:end_idx]
            batch_y = shuffled_y_train[start_idx:end_idx]

            # Process batch and get tensors
            x_batch = process_batch(phrases, batch_indices, batch_lengths)
            y_batch = Tensor(batch_y)

            # Forward and backward pass
            optimizer.zero_grad()
            output = model(x_batch, batch_lengths)
            loss = output.sparse_categorical_crossentropy(y_batch)
            loss.backward()
            optimizer.step()

            # Track metrics
            batch_loss = loss.numpy()
            epoch_loss += batch_loss

            predictions = output.numpy().argmax(axis=1)
            correct_preds += np.sum(predictions == batch_y)
            total_preds += len(batch_y)

            del x_batch, y_batch, output, loss

        # Calculate training metrics
        train_losses[epoch] = epoch_loss / num_batches
        train_accs[epoch] = correct_preds / total_preds

        # Evaluation phase
        val_loss, val_acc = evaluate_model(
                model, phrases, idx_val, val_docs_len, y_val, CONFIG["batch_size"]
        )

        val_losses[epoch] = val_loss
        val_accs[epoch] = val_acc

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - "
              f"Loss: {train_losses[epoch]:.4f} - Acc: {train_accs[epoch]:.4f} - "
              f"Val Loss: {val_losses[epoch]:.4f} - Val Acc: {val_accs[epoch]:.4f} - "
              f"Time: {epoch_time:.2f}s")

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Force garbage collection between epochs
        gc.collect()

    # Final evaluation on test set
    test_loss, test_acc = evaluate_model(
            model, phrases, idx_test, test_docs_len, y_test, CONFIG["batch_size"]
    )

    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")

    # Plot training history (only for completed epochs)
    completed_epochs = epoch + 1
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses[:completed_epochs], label='Training Loss')
    plt.plot(val_losses[:completed_epochs], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs[:completed_epochs], label='Training Accuracy')
    plt.plot(val_accs[:completed_epochs], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    train_model()