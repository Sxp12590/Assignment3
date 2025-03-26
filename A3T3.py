import numpy as np
import tensorflow as tf

# 1. Load a text dataset (using Shakespeare's sonnets as an example)
path_to_file = tf.keras.utils.get_file('sonnets.txt', 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
text = open(path_to_file, 'r').read()

# Print a snippet of the text
print("Text snippet:", text[:500])

# 2. Convert text to a sequence of characters (one-hot encoding)
# Create a list of unique characters
chars = sorted(set(text))
print(f"Total unique characters: {len(chars)}")

# Create mappings from character to integer and vice versa
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

# Convert the entire text to integer indices
text_as_int = np.array([char_to_index[char] for char in text])

# Parameters for the model
SEQ_LENGTH = 100  # Length of the sequence
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 10
CHUNK_SIZE = len(text_as_int) // SEQ_LENGTH

# Create sequences and targets
sequences = []
targets = []

for i in range(0, len(text_as_int) - SEQ_LENGTH, SEQ_LENGTH):
    seq_in = text_as_int[i:i+SEQ_LENGTH]
    seq_out = text_as_int[i+1:i+SEQ_LENGTH+1]
    sequences.append(seq_in)
    targets.append(seq_out)

sequences = np.array(sequences)
targets = np.array(targets)

# 3. Define the RNN model using LSTM (Non-stateful)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 256, input_length=SEQ_LENGTH),
    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 4. Train the model
model.fit(sequences, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

# 5. Generate new text by sampling characters one at a time (without reset_states)
def generate_text(model, start_string, temperature=1.0, num_generate=500):
    input_eval = [char_to_index[char] for char in start_string]
    input_eval = tf.expand_dims(input_eval, 0)  # Add batch dimension

    # Empty list to hold the generated text
    generated_text = start_string

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]  # Get the last character predictions

        # Scale the logits by temperature
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Convert the predicted character to text
        generated_text += index_to_char[predicted_id]

        # Use the predicted character as the next input
        input_eval = tf.expand_dims([predicted_id], 0)

    return generated_text

# Generate new text using a seed string
seed_string = "What am I doing?"
generated_text = generate_text(model, seed_string, temperature=0.7)
print("\nGenerated Text:\n", generated_text)
