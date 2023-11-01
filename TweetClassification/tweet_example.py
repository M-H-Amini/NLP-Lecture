import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tweet_visualization import readTweetDataset
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Define the inputs and outputs
X, y, df = readTweetDataset()

model = Sequential()
model.add(Dense(3, activation='softmax', input_shape=(X.shape[1],)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=32)

##  Visualization...
sns.set(font_scale=1.5)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('J')
plt.title('Cost Function')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('tweet_nn.png', layout='tight')
plt.show()

