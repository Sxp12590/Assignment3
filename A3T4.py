import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the IMDB sentiment dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# 2. Preprocess the text data
# Padding sequences to make them of equal length
SEQ_LENGTH = 256
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=SEQ_LENGTH)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=SEQ_LENGTH)

# 3. Build and compile the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=SEQ_LENGTH),
    tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the model
history = model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_data=(test_data, test_labels))

# 5. Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# 6. Make predictions on test data
predictions = (model.predict(test_data) > 0.5).astype("int32")

# 7. Generate confusion matrix and classification report
cm = confusion_matrix(test_labels, predictions)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(test_labels, predictions))

# 8. Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


