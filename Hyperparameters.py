
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback
import DeepStrain.Defaults as Defaults

cg = Defaults.Parameters()


def learning_rate_step_decay_classic(epoch, lr, decay = cg.decay_rate, initial_power=cg.initial_power, start_epoch = cg.start_epoch):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    lrate = (1/ (1 + decay * (epoch + start_epoch))) * (10 ** initial_power)

    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)

def learning_rate_step_decay_slower(epoch, lr, decay = cg.decay_rate, initial_power=-5, start_epoch = cg.start_epoch):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    lrate = (1/ (1 + decay * (epoch + start_epoch))) * (10 ** initial_power)

    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)


def loss_smooth(y_true, y_pred):
    # Calculate smoothness loss for each dimension
    smoothness_x = tf.reduce_mean(tf.square(tf.image.image_gradients(y_pred[...,0])))
    smoothness_y = tf.reduce_mean(tf.square(tf.image.image_gradients(y_pred[...,1])))
    # Take the average of the smoothness losses
    return  (smoothness_x + smoothness_y) / 2.0

def loss_dice(y_true, y_pred):

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return 1.0 - dice

def loss_mae(y_true, y_pred):
    # Create a mask for pixels greater than 0 in y_true
    mask = tf.cast(y_true > 0, tf.float32)

    # Apply the mask to y_true and y_pred
    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask

    # Calculate the absolute difference between masked_y_true and masked_y_pred
    return  tf.reduce_mean(tf.abs(masked_y_true - masked_y_pred))


def dice_loss_selected_class(y_true, y_pred):
    y_true_selected = tf.gather(y_true, [0, 2, 3], axis=-1)
    y_pred_selected =tf.gather(y_pred, [0, 2, 3], axis=-1)

    intersection = tf.reduce_sum(y_true_selected * y_pred_selected)
    union = tf.reduce_sum(y_true_selected) + tf.reduce_sum(y_pred_selected)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return 1.0 - dice





# def custom_mae_loss(y_true, y_pred):
#     '''if a row has [0,0], doesn't take it into consideration'''
#     # Create a mask for rows with non-zero values in y_true
#     mask = tf.math.reduce_any(y_true[:, :, 0:] != 0, axis=-1)
    
#     # Apply the mask to y_true and y_pred
#     masked_y_true = tf.boolean_mask(y_true, mask)
#     masked_y_pred = tf.boolean_mask(y_pred, mask)
    
#     # Calculate the absolute difference
#     absolute_diff = tf.abs(masked_y_true - masked_y_pred)
    
#     # Calculate the mean of the absolute difference
#     loss = tf.reduce_mean(absolute_diff)
    
#     return loss

# def dice_coefficient_calculation(y_true, y_pred, class_value):
#     class_mask_true = tf.cast(tf.equal(y_true, class_value), dtype=tf.float32)
#     class_mask_pred = tf.cast(tf.equal(y_pred, class_value), dtype=tf.float32)

#     intersection = tf.reduce_sum(class_mask_true * class_mask_pred)
#     union = tf.reduce_sum(class_mask_true) + tf.reduce_sum(class_mask_pred)
#     return (2.0 * intersection + 1e-5) / (union + 1e-5)


# def dice_loss_two_classes(y_true, y_pred):
#     dice_class_1  = dice_coefficient_calculation(y_true, y_pred, class_value = 1)
#     dice_class_2 = dice_coefficient_calculation(y_true, y_pred, class_value = 2)

#     return 1.0 - (dice_class_1 + dice_class_2) / 2.0