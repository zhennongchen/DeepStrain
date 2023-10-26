#!/usr/bin/env python

import sys
sys.path.append("../")
import os
import glob
import numpy as np
import nibabel as nb
import pandas as pd

import tensorflow as tf
from contextlib import redirect_stdout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax

import DeepStrain.functions_collection as ff
import DeepStrain.models.networks as network
import DeepStrain.Defaults as Defaults
import DeepStrain.Hyperparameters as hyper
import DeepStrain.Build_list.Build_list as Build_list
import DeepStrain.Generator_motion as Generator_motion
from DeepStrain.models.dense_image_warp import dense_image_warp3d as warp

cg = Defaults.Parameters()

##############################################
trial_name = 'fine_tune_carmen'
data_sheet = os.path.join(cg.deep_dir,'data/Patient_list','Patient_list_for_motion_checked.xlsx')
patient_index = 0
maximum_epoch = 300
##############################################

# build list
b = Build_list.Build(data_sheet)
patient_id ,our_id,_ ,ed ,es , image_folder ,seg_folder ,pred_seg_folder ,_ , start_slice, end_slice = b.build_for_personalized_motion(patient_index)

print(patient_id)

# create model
V_0_input = Input(shape = [128,128,16,1]) 
V_t_input = Input(shape = [128,128,16,1]) 
M_t_input = Input(shape = [128,128,16,3])
M_t_split = tf.split(M_t_input, M_t_input.shape[-1], -1)

input   = Concatenate(axis=-1)([V_0_input, V_t_input])
motion_estimates = network.encoder_decoder(input, nchannels=3, map_activation=None)
V_0_pred = warp(V_t_input, motion_estimates, name = 'warp' )
M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t_input)), motion_estimates) for mt in M_t_split], -1)    
M_0_pred  = softmax(M_0_pred, name = 'softmax') 

model = Model(inputs = [V_0_input, V_t_input, M_t_input], outputs = [motion_estimates,V_0_pred, M_0_pred])

model_file =  os.path.join(cg.deep_dir,'models/trained/carmen_Jan2021.h5')
model.load_weights(model_file)


# compile model
opt = Adam(lr = 1e-4)
model.compile(optimizer= opt, 
                loss = [hyper.loss_smooth, 'MAE', hyper.loss_dice],
                loss_weights=[0.1, 0.1, 0.5])

# # set callbacks
model_fld = os.path.join(cg.deep_dir, 'models', trial_name, patient_id, 'models')
ff.make_folder([os.path.dirname(os.path.dirname(model_fld)), os.path.dirname(model_fld), model_fld, os.path.join(os.path.dirname(model_fld), 'logs')])
csv_logger = CSVLogger(os.path.join(os.path.dirname(model_fld), 'logs', 'training-log.csv')) 

class CustomCallback(Callback):
    def __init__(self, model_fld, max_epoch, stopping_patience = 5):
        super(CustomCallback, self).__init__()
        self.model_fld = model_fld
        self.no_improvement_count = 0
        self.min_loss = float('inf')
        self.max_epoch = max_epoch
        self.stopping_patience = stopping_patience

    def on_epoch_end(self, epoch, logs=None):
   
        current_loss = logs.get('loss')
        if current_loss < self.min_loss:
            self.min_loss = current_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.stopping_patience or epoch == self.max_epoch - 1:  # early stopping
            epoch_str = str(epoch + 1).zfill(3)  # Format epoch number
            print(epoch_str)
            self.model.stop_training = True
            save_model(self.model, os.path.join(self.model_fld,  'model-'+epoch_str+'.hdf5'))  # Save the last model

callbacks = [csv_logger,
                CustomCallback(model_fld,maximum_epoch), 
                    LearningRateScheduler(hyper.learning_rate_step_decay_slower),   
                    ]

# # Fit
datagen = Generator_motion.DataGenerator(patient_id,
                                        image_folder,
                                        seg_folder,
                                        ed,
                                        np.array([es - ed]),
                                        heart_slices = [start_slice, end_slice],
                                        batch_size = 1,
                                        img_shape = [128,128,16],        
                                        normalize = True,)


model.fit_generator(generator = datagen,
                    epochs = 300,
                    callbacks = callbacks,
                    verbose = 1,)



