import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
import matplotlib.pyplot as plt

base_model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
layer = base_model.get_layer(layer_name)

# filter index
filter_index = 0  # Index of the filter you are interested in
print(base_model.summary())
print(layer.output.shape)

# layer_output = layer.output[:, :, :, filter_index]
layer_output = layer.output[:, :, :, :3]
activation_model = tf.keras.models.Model(inputs=base_model.input, outputs=layer_output)

# loss function to maximize activation
def loss(input_image):
    activation = activation_model(input_image)
    return tf.reduce_mean(activation)

# Gradient ascent
@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss_value = loss(img)
    grads = tape.gradient(loss_value, img)
    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    img += learning_rate * normalized_grads
    # The image is updated by moving it along the direction of the normalized gradients. 
    # The learning_rate controls how big a step is taken in the direction of the gradient. 
    # Since this is gradient ascent, the image is adjusted in the direction that increases the loss, thereby increasing the activation of the filter.
    return img

# a random image
img = tf.random.uniform((1, 224, 224, 3))

# gradient ascent
for _ in range(30):
    img = gradient_ascent_step(img, learning_rate=10.0)

plt.imshow(np.clip(img[0].numpy(), 0, 1))
plt.show()