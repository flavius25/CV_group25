import tensorflow as tf
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(len(train_images), len(train_labels), len(test_images))






