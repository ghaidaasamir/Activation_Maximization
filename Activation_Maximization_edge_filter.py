import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# a simple vertical edge detection filter
def create_edge_filter():
    filter = tf.constant([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=tf.float32)
    filter = tf.reshape(filter, [3, 3, 1, 1]) 
    return filter

def apply_filter(image, filter):
    return tf.nn.conv2d(image, filter, strides=[1, 1, 1, 1], padding="VALID")

input_image = tf.image.decode_jpeg(tf.io.read_file('Activation_Maximization/test_image.png'))

resized_image = tf.image.resize(input_image, [224, 224])

resized_image = tf.expand_dims(resized_image, axis=0) 

img = tf.image.rgb_to_grayscale(resized_image)

def loss(input_image):
    activation = apply_filter(input_image, create_edge_filter())
    return -tf.reduce_mean(activation)  # Negate to maximize

@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss_value = loss(img)
    grads = tape.gradient(loss_value, img)
    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    img += learning_rate * normalized_grads  # Add to maximize
    return img

for _ in range(30):
    img = gradient_ascent_step(img, learning_rate=1.0)

plt.figure(figsize=(6, 6))
plt.imshow(img[0, :, :, 0], cmap='gray')
plt.title("Generated Image to Maximize Edge Detection")
plt.show()
