import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
This solution implements a spatial transformer which corrects the orientations of the input image
content through spatial transformations on the x, y axis. 
The corrected feature maps are then fed through a standard multi-layer CNN to be trained the
standard way.

Input training and validation image sequences are controlled by two ImageDataGenerator under the tensorflow.keras api.
"""


##Hyperparameters + Global Variables##
num_classes = 3
batch_size = 128
num_epochs = 30
LR = 0.001
train_data_generator = 
validation_data_generator = 



def train(input_shape, num_classes, sampling_size=None):
        if sampling_size is None:
            sampling_size = (input_shape[0], input_shape[1])    # Defaults to width and height
            
        input_shape = (None, *input_shape)
        input_ph = tf.keras.Input(shape=input_shape, dtype=tf.float32)  # Empty placeholder tensor that repersents the input data, 4D
        
        locnet = layers.Flatten()(input_ph)     # Flattern the input tensor to be passed into a dense layer
        locnet = layers.Dense(20, activation='relu')(locnet)    # Applies relu activation to achieve non-linearity, sets the output shape to [None, 20]
        
        custom_kernel_initializer = tf.initializers.Zeros()     # Initializes the weight matrix to all 0
        custom_bias_initializer = tf.constant_initializer([1.0, 0, 0, 0, 1, 0])     # Initializes the bias matrix to identity matrix, encouraging mirror transformations

        affine_mat = layers.Dense(units = 6,    # Creates the affine matrix, the output is set to [None, 6] coresponding to the 6 factors needed for spatial transformation
                                  activation = 'tanh',      # Ensures the value range is within 0 and 1
                                  kernel_initializer = custom_kernel_initializer, 
                                  bias_initializer = custom_bias_initializer)(locnet)

        input_tensor = spatial_transformer(input_ph, affine_mat, sampling_size, input_shape[-1])    # Performs spatial transformation, returns the modified feature map tensor
        
        model = CNN(input_tensor)   # Passes the modified tensor into the CNN, returns the compiled model
        
        my_callback = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)]
        
        history = model.fit(train_data_generator, 
                            steps_per_epoch=len(train_data_generator), 
                            epochs = num_epochs, 
                            validation_data = validation_data_generator, 
                            validation_steps = len(validation_data_generator), 
                            callbacks = my_callback)
        
        return history


def CNN(input_tensor):
    x = layers.Conv2D(32, (3, 3), activation = 'relu')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation = 'relu')(x)
    x = layers.MaxPooling2D((3, 3))(x)
    
    x = layers.Conv2D(32, (3, 3), padding = "same", activation = 'relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)    

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    
    optimizer = tf.optimizers.Adam(learning_rate = LR)
    model.compile(optimizer = optimizer,  # Adam optimizer
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
    
    return model


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