import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import ops
import numpy as np
import matplotlib.pyplot as plt

"""
## Prepare the data
"""

num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


"""
## Configure the hyperparameters
"""

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier


"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


"""
## Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation as a layer

Takes the dimention of the patch and the input image,
then the dimentions of a input patch is used to calculate the number of patches can be produced.
Patches of set size are extracted in sequence and reshaped from 4D to 3D, patch dimention is flattened along axis.
"""

def Patches(patch_size, augmented):
    input_shape = tf.shape(augmented)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    patches = tf.image.extract_patches(augmented, sizes = [1, patch_size, patch_size, 1], strides = [1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
    patches = tf.reshape(patches, [batch_size,
                num_patches_h * num_patches_w,
                patch_size * patch_size * channels])
    
    return patches
    

def PatchEncoder(patches):
    # Linearly embed the patches
    positions = ops.expand_dims(
            ops.arange(start=0, stop=num_patches, step=1), axis=0
            )
    positions = tf.reshape(positions, (1, 1, 1, -1))
    embed = layers.Conv2D(projection_dim, kernel_size=1, strides=1, padding='same', activation='linear')(positions)
    projections = layers.Dense(units=projection_dim)(patches)
    
    return projections + embed   # Adds positional embeddings to every patch in the sequence 


"""
Let's display patches for a sample image
"""

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size, resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")


def spatial_transformer(feature_map, affine_mat, output_size, channels):    # Performs spatial transformation, returns the modified feature map tensor
        # feature map: [None, H, W, C]
        # affine_mat: [None, 6]
        # output_size: (H, W)
        num_batch, in_h, in_w = tf.shape(feature_map)[0], tf.shape(feature_map)[1], tf.shape(feature_map)[2]    # Batch number, height and width
        out_h, out_w = output_size  # Output size is the original 4D tensor size, preserved 

        # step 1. affine grid
        x = tf.linspace(-1.0, 1.0, out_w)
        y = tf.linspace(-1.0, 1.0, out_h)
        regular_x, regular_y = tf.meshgrid(x, y)
        reg_flatx, reg_flaty = tf.reshape(regular_x, [-1]), tf.reshape(regular_y, [-1]) # [None, HW]
        regular_grid = tf.stack([reg_flatx, reg_flaty, tf.ones_like(reg_flatx)])  # [3, HW]
        regular_grid = tf.expand_dims(regular_grid, axis=0)  # [1, 3, HW]
        regular_grid = tf.tile(regular_grid, [num_batch, 1, 1])  # [None, 3, HW]

        theta = tf.cast(tf.reshape(affine_mat, [-1, 2, 3]), tf.float32)
        regular_grid = tf.cast(regular_grid, tf.float32)
        sampled_grid = tf.matmul(theta, regular_grid)  # [None, 2, 3] x [None, 3, HW]=>[None, 2, HW]

        # step 2. sampler
        max_x = tf.cast(in_w - 1, 'int32')
        max_y = tf.cast(in_h - 1, 'int32')
        x = 0.5 * (1.0 + sampled_grid[:, 0, :]) * tf.cast(in_w, tf.float32)
        y = 0.5 * (1.0 + sampled_grid[:, 1, :]) * tf.cast(in_h, tf.float32)  # [None, HW], float32, range: [0, H]

        x0 = tf.cast(tf.floor(x), tf.int32)     # Round down to set the bound for the 4 edges
        y0 = tf.cast(tf.floor(y), tf.int32)
        x0 = tf.clip_by_value(x0, 0, max_x - 1)
        y0 = tf.clip_by_value(y0, 0, max_y - 1)

        x1 = tf.clip_by_value(x0 + 1, 0, max_x)
        y1 = tf.clip_by_value(y0 + 1, 0, max_y)  # [None, HW], int32, range: [0, H]

        batch_idx = tf.reshape(tf.range(0, num_batch), [-1, 1])
        batch_idx = tf.tile(batch_idx, (1, out_h * out_w))  # [None, HW]
        y0x0 = tf.gather_nd(feature_map, tf.stack([batch_idx, y0, x0], axis=-1))
        y1x0 = tf.gather_nd(feature_map, tf.stack([batch_idx, y1, x0], axis=-1))
        y0x1 = tf.gather_nd(feature_map, tf.stack([batch_idx, y0, x1], axis=-1))
        y1x1 = tf.gather_nd(feature_map, tf.stack([batch_idx, y1, x1], axis=-1))  # [None*HW, C]

        x0, x1 = tf.cast(x0, tf.float32), tf.cast(x1, tf.float32)
        y0, y1 = tf.cast(y0, tf.float32), tf.cast(y1, tf.float32)

        w00 = (x1 - x) * (y1 - y)
        w10 = (x1 - x) * (y - y0)
        w01 = (x - x0) * (y1 - y)
        w11 = (x - x0) * (y - y0)  # [None, HW]

        return tf.add_n([tf.reshape(w00, [-1, out_h, out_w, 1]) * tf.reshape(y0x0, [-1, out_h, out_w, channels]),
                         tf.reshape(w10, [-1, out_h, out_w, 1]) * tf.reshape(y1x0, [-1, out_h, out_w, channels]),
                         tf.reshape(w01, [-1, out_h, out_w, 1]) * tf.reshape(y0x1, [-1, out_h, out_w, channels]),
                         tf.reshape(w11, [-1, out_h, out_w, 1]) * tf.reshape(y1x1, [-1, out_h, out_w, channels])])


"""
## Build the ViT model

The ViT model consists of multiple Transformer blocks,
which use the `layers.MultiHeadAttention` layer as a self-attention mechanism
applied to the sequence of patches. The Transformer blocks produce a
`[batch_size, num_patches, projection_dim]` tensor, which is processed via an
classifier head with softmax to produce the final class probabilities output.

Unlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),
which prepends a learnable embedding to the sequence of encoded patches to serve
as the image representation, all the outputs of the final Transformer block are
reshaped with `layers.Flatten()` and used as the image
representation input to the classifier head.
Note that the `layers.GlobalAveragePooling1D` layer
could also be used instead to aggregate the outputs of the Transformer block,
especially when the number of patches and the projection dimensions are large.
"""


def create_vit_classifier():
    if sampling_size is None:
        sampling_size = (input_shape[0], input_shape[1])    # Defaults to width and height
        
    inputs = keras.Input(shape=input_shape)
    
    # Augment data.
    augmented = data_augmentation(inputs)

    input_ph = tf.reshape(augmented, [None, *augmented])   # Might be a problem, needs to test the shape of "augmented"
    
    locnet = layers.Flatten()(input_ph)     # Flattern the input tensor to be passed into a dense layer
    locnet = layers.Dense(20, activation='relu')(locnet)    # Applies relu activation to achieve non-linearity, sets the output shape to [None, 20]
        
    custom_kernel_initializer = tf.initializers.Zeros()     # Initializes the weight matrix to all 0
    custom_bias_initializer = tf.constant_initializer([1.0, 0, 0, 0, 1, 0])     # Initializes the bias matrix to identity matrix, encouraging mirror transformations

    affine_mat = layers.Dense(units = 6,    # Creates the affine matrix, the output is set to [None, 6] coresponding to the 6 factors needed for spatial transformation
                              activation = 'tanh',      # Ensures the value range is within 0 and 1
                              kernel_initializer = custom_kernel_initializer, 
                              bias_initializer = custom_bias_initializer)(locnet)

    input_tensor = spatial_transformer(input_ph, affine_mat, sampling_size, input_shape[-1])    # Performs spatial transformation, returns the modified feature map tensor    

    # Create patches.
    patches = Patches(patch_size, input_tensor)  #May have problem with compatitble shape
    # Encode patches.
    encoded_patches = PatchEncoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model


"""
## Compile, train, and evaluate the mode
"""


def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_history("loss")
plot_history("top-5-accuracy")
