**Assignment3**
Name: Pothuru Sivarkamani Naga Mruthyunjaya
Student Id: 700751259

**Q1: Implementing a Basic Autoencoder**

Loading the MNIST Dataset:
We load the MNIST dataset using tensorflow.keras.datasets.mnist.
The data is normalized by dividing by 255 and reshaped into a flat 784-dimensional vector for each image.
Define the Autoencoder:
The encoder takes a 784-dimensional input and encodes it into a 32-dimensional latent space using a Dense layer.
The decoder takes the 32-dimensional latent space and reconstructs it back into the original 784-dimensional space.
We use a sigmoid activation function for the output layer to scale the values between 0 and 1 (since the input data was normalized between 0 and 1).
Compile and Train the Autoencoder:
The autoencoder is compiled using the Adam optimizer and binary_crossentropy loss function, as the pixel values are between 0 and 1.
The model is trained using the fit() function with the MNIST training data. We train for 50 epochs with a batch size of 256.
Plot Original vs Reconstructed Images:
After training, the autoencoder is used to generate reconstructions of the test set.
We plot the first 10 original images and their corresponding reconstructed images for comparison.
Modify Latent Dimension:
We try two other latent dimensions: 16 (smaller) and 64 (larger).
The model is retrained for each latent dimension, and the results are plotted to observe how the latent dimension size affects the quality of the reconstructed images.
Results Analysis:
Latent Dimension 32: Provides a balance between compression and reconstruction quality.
Latent Dimension 16: The reconstructions might lose more details because the encoding space is too small to fully capture the information of the original image.
Latent Dimension 64: The quality of the reconstructions is usually better because the latent space is larger and can capture more details of the original image, but it's less compressed.
This demonstrates how the size of the latent space (the bottleneck) impacts the ability of the autoencoder to effectively reconstruct images.

**Q2: Implementing a Denoising Autoencoder**
Loading the MNIST Dataset:

The MNIST dataset is loaded, and each image is normalized to the range [0, 1] and reshaped into a 784-dimensional vector.
Adding Gaussian Noise:

We define a function add_noise() that adds Gaussian noise with mean = 0 and standard deviation = 0.5 to the images. This noisy data is clipped to the range [0, 1] to ensure pixel values are valid.
We apply this noise function to both the training and test sets.
Denoising Autoencoder Model:
The autoencoder model is built using a simple architecture with an input layer of size 784, a hidden layer (encoder) of size 32, and a decoder that reconstructs the image back to size 784.
The model is compiled using the Adam optimizer and binary cross-entropy loss function. The target during training is the clean, unmodified image (x_train), while the input to the model is the noisy image (x_train_noisy).
Training the Denoising Autoencoder:
The model is trained on noisy input images but with the clean images as the target.
Visualization of Noisy vs. Reconstructed Images:
We visualize the first 10 images in the test set by displaying the noisy input, the original clean image, and the image reconstructed by the denoising autoencoder.
Comparison with Basic Autoencoder:
We also define and train a basic autoencoder (without adding noise to the input) for comparison. We visualize the performance of the basic autoencoder by displaying its reconstructed images.
Results and Observations:
Denoising Autoencoder: The model should be able to reconstruct clean images from noisy inputs. It will learn to "remove" the noise and return the original image.
Basic Autoencoder: The basic autoencoder, which does not handle noisy inputs, might struggle to reconstruct clean images from noisy inputs, as it has not been trained to deal with noise.
Real-World Use Case for Denoising Autoencoders:
One real-world scenario where denoising autoencoders can be useful is in medical imaging (e.g., MRI scans or X-rays). These images are often noisy due to various factors like equipment limitations, patient movement, or environmental conditions. A denoising autoencoder can help in reconstructing high-quality images from noisy inputs, improving the accuracy of medical diagnoses and reducing the need for expensive equipment or procedures.

**Q3: Implementing an RNN for Text Generation**

Load the Text Dataset:
A small excerpt from "The Little Prince" is used as an example. You can replace this with any text dataset by loading it from a file.
Preprocess the Text:
The unique characters in the text are extracted and mapped to integer indices.
The text is then encoded as a sequence of integers, where each integer corresponds to a character.
Prepare Input Sequences:
The text is split into sequences of length seq_length (e.g., 40 characters). For each sequence, the target is the next character that follows the sequence.
Build the LSTM Model:
The model is defined as a simple LSTM-based neural network. The input is a sequence of characters, and the model predicts the next character using a softmax activation.
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
Train the Model:
The model is trained on the prepared sequences and labels for a specified number of epochs.
Generate Text:
After training, the generate_text function is used to generate new text by sampling characters one at a time.
Temperature scaling is applied to control the randomness of the predictions. When the temperature is high, the model's output becomes more random, while a low temperature makes the predictions more deterministic (i.e., the model will likely choose the most probable character).
Temperature Scaling:
Low temperature (e.g., 0.5): Makes the model output more focused and deterministic. The model will prefer high-probability characters.
High temperature (e.g., 1.2): Increases the randomness, making the model more likely to pick less probable characters.
Temperature = 1.0: A balance between randomness and determinism.
Real-World Applications of RNNs for Text Generation:
Story or poem generation: An RNN model trained on a corpus of text can generate new, creative pieces based on an initial seed.
Code generation: RNNs can be trained on programming language syntax to generate code based on a given prompt.
Chatbots: RNN-based models are commonly used for generating human-like responses in chatbot applications.

**Q4: Sentiment Classification Using RNN**

Loading the IMDB dataset:
The tensorflow.keras.datasets.imdb function loads the IMDB dataset. We limit the vocabulary to the 10,000 most frequent words (num_words=10000).
The data is split into x_train, y_train, x_test, and y_test. The input x_train and x_test are lists of word indices, while y_train and y_test are binary labels indicating whether the review is positive (1) or negative (0).
Preprocessing the data:
We pad the sequences to a fixed length (max_len=500), ensuring that all input sequences have the same size for compatibility with the neural network.
Building the LSTM Model:
The model consists of an embedding layer that maps words to dense vectors, followed by an LSTM layer that processes the sequence data, and a final dense layer with a sigmoid activation for binary classification (positive or negative sentiment).
The model is compiled using the Adam optimizer and binary cross-entropy loss, which is appropriate for binary classification tasks.
Training the model:
The model is trained for 5 epochs with a batch size of 64. We also provide validation data (x_test, y_test) to monitor the model's performance on the test set during training.
Evaluating the Model:
After training, we predict sentiment labels for the test set. The output from the model is a probability, so we convert it to binary labels (0 or 1) using a threshold of 0.5.
We use confusion_matrix and classification_report from sklearn.metrics to evaluate the model's performance. The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. The classification report provides metrics such as accuracy, precision, recall, and F1-score.
Precision-Recall Tradeoff:
In sentiment analysis, the precision-recall tradeoff is crucial because of the inherent class imbalance (positive reviews are more common than negative ones in some datasets). A high precision means fewer false positives, but if the recall is low, the model might miss many positive samples (i.e., negative reviews labeled as positive). We need a balance between precision and recall for optimal performance.
Output:
After running the code, you will see:
Confusion Matrix: A heatmap showing the true vs predicted labels.
Classification Report: The precision, recall, F1-score, and support for each class (positive and negative).
The precision-recall tradeoff is important for sentiment classification because different applications may prioritize either precision (fewer false positives) or recall (fewer false negatives). Understanding this tradeoff helps ensure that the model behaves as expected in real-world applications. In sentiment analysis, this could mean deciding whether to be more cautious in labeling reviews as positive (high precision) or ensuring that all positive reviews are detected (high recall).




























