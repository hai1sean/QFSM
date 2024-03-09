import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import extract_feats.opensmile as of
from utils import parse_opt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_epochs = 50   # Number of optimization epochs
n_layers = 1    # Number of random layers
n_train = 200    # Size of the train dataset
n_test = 50     # Size of the test dataset


SAVE_PATH = "quanvolution/" # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

config = parse_opt()
x_train, x_test, y_train, y_test = of.load_feature(config, train=True)

train_voices = x_train[:n_train]
train_labels = y_train[:n_train]

test_voices = x_test[:n_test]
test_labels = y_test[:n_test]

# train_voices = np.array(train_voices[..., tf.newaxis], requires_grad = False)
# test_voices = np.array(test_voices[..., tf.newaxis], requires_grad = False)
print(train_voices[0])
print(train_voices[0].shape)
print(type(train_voices[0]))
print(train_voices[0][0])
# mnist_dataset = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
#
# # Reduce dataset size
# train_images = train_images[:n_train]
# train_labels = train_labels[:n_train]
# test_images = test_images[:n_test]
# test_labels = test_labels[:n_test]
#
# # Normalize pixel values within 0 and 1
# train_images = train_images / 255
# test_images = test_images / 255
#
# # Add extra dimension for convolution channels
# train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
# test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

dev = qml.device("default.qubit", wires=5)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 5))

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(5):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(5)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(5)]


def quanv(voice):
    """Convolves the input voice with many applications of the same quantum circuit."""
    out = np.zeros((28, 5))

    for i in range(5):
        for j in range (28):
            q_results = circuit(
                [
                    voice[j, 0],
                    voice[j + 1, 0],
                    voice[j + 2, 0],
                    voice[j + 3, 0],
                    voice[j + 4, 0]
                ]
            )
            out[j, i] = sum(q_results)
    return out

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    # for j in range(0, 28, 2):
    #     for k in range(0, 28, 2):
    #         # Process a squared 2x2 region of the image with a quantum circuit
    #         q_results = circuit(
    #             [
    #                 image[j, k, 0],
    #                 image[j, k + 1, 0],
    #                 image[j + 1, k, 0],
    #                 image[j + 1, k + 1, 0]
    #             ]
    #         )
    #         # Assign expectation values to different channels of the output pixel (j/2, k/2)
    #         for c in range(4):
    #             out[j // 2, k // 2, c] = q_results[c]
    # return out


if PREPROCESS == True:
    q_train_voices = []
    print("Quantum pre-processing of train voices:")
    for idx, voice in enumerate(train_voices):
        print("{}/{}        ".format(idx + 1, n_train), end="\r")
        q_train_voices.append(quanv(voice))
    q_train_voices = np.asarray(q_train_voices)

    q_test_voices = []
    print("\nQuantum pre-processing of test voices:")
    for idx, voice in enumerate(test_voices):
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_voices.append(quanv(voice))
    q_test_voices = np.asarray(q_test_voices)

    # Save pre-processed voices
    np.save(SAVE_PATH + "q_train_voices.npy", q_train_voices)
    np.save(SAVE_PATH + "q_test_voices.npy", q_test_voices)


# Load pre-processed voices
q_train_voices = np.load(SAVE_PATH + "q_train_voices.npy")
q_test_voices = np.load(SAVE_PATH + "q_test_voices.npy")


# n_samples = 4
# n_channels = 4
# fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
# for k in range(n_samples):
#     axes[0, 0].set_ylabel("Input")
#     if k != 0:
#         axes[0, k].yaxis.set_visible(False)
#     axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")
#
#     # Plot all output channels
#     for c in range(n_channels):
#         axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
#         if k != 0:
#             axes[c, k].yaxis.set_visible(False)
#         axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")
#
# plt.tight_layout()
# plt.show()

##############################################################################
# Below each input image, the :math:`4` output channels generated by the
# quantum convolution are visualized in gray scale.
#
# One can clearly notice the downsampling of the resolution and
# some local distortion introduced by the quantum kernel.
# On the other hand the global shape of the image is preserved,
# as expected for a convolution layer.

##############################################################################
# Hybrid quantum-classical model
# ------------------------------
#
# After the application of the quantum convolution layer we feed the resulting
# features into a classical neural network that will be trained to classify
# the :math:`10` different digits of the MNIST dataset.
#
# We use a very simple model: just a fully connected layer with
# 10 output nodes with a final *softmax* activation function.
#
# The model is compiled with a *stochastic-gradient-descent* optimizer,
# and a *cross-entropy* loss function.


def MyModel():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


##############################################################################
# Training
# ^^^^^^^^
#
# We first initialize an instance of the model, then we train and validate
# it with the dataset that has been already pre-processed by a quantum convolution.

q_model = MyModel()

q_history = q_model.fit(
    q_train_voices,
    train_labels,
    validation_data=(q_test_voices, test_labels),
    batch_size=20,
    epochs=n_epochs,
    verbose=1,
)

##############################################################################
# In order to compare the results achievable with and without the quantum convolution layer,
# we initialize also a "classical" instance of the model that will be directly trained
# and validated with the raw MNIST images (i.e., without quantum pre-processing).

c_model = MyModel()

c_history = c_model.fit(
    train_voices,
    train_labels,
    validation_data=(test_voices, test_labels),
    batch_size=20,
    epochs=n_epochs,
    verbose=1,
)


##############################################################################
# Results
# ^^^^^^^
#
# We can finally plot the test accuracy and the test loss with respect to the
# number of training epochs.

import matplotlib.pyplot as plt

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(q_history.history["val_accuracy"], "-ob", label="With quantum layer")
ax1.plot(c_history.history["val_accuracy"], "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history["val_loss"], "-ob", label="With quantum layer")
ax2.plot(c_history.history["val_loss"], "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.show()


##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook.
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/andrea_mari.txt
