# import packages
import numpy as np
import matplotlib.pyplot as plt
import keras

## constants
PIXELS_VALUE = 255.0

# load MNIST dataset
mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
train_images = train_images / PIXELS_VALUE
test_images = test_images / PIXELS_VALUE

# Ploting
plt.figure()
plt.imshow(train_images[2])
plt.colorbar()
plt.grid(False)
plt.show()

# Implement Neural Network
nn_model = keras.Sequential([
  keras.layers.Reshape((28, 28, 1), input_shape=(28,28)),
  keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
 ])

nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
nn_model.fit(train_images, train_labels, epochs=10)
# Evaluate the model
test_loss, test_acc = nn_model.evaluate(test_images, test_labels, verbose=2)

print("Final test accuracy: ", test_acc)
print("Final test loss: ", test_loss)
