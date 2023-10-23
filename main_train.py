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
from tensorflow.keras.layers import Input

import DeepStrain.functions_collection as ff
import DeepStrain.models.networks as network
import DeepStrain.Defaults as Defaults
import DeepStrain.Hyperparameters as hyper

cg = Defaults.Parameters()

trial_name = 'fine_tune_carson'
val_batch = 0

input_shape = (128,128,1)
model_inputs = [Input(input_shape)]
M = network.encoder_decoder(model_inputs[0], nchannels = 4, map_activation='softmax')
model_outputs = [M]
model = Model(inputs=model_inputs, outputs=model_outputs)

model_file =  os.path.join(cg.deep_dir,'models/trained/carson_Jan2021.h5')
model.load_weights(model_file)

# compile model
opt = Adam(lr = 1e-4)
model.compile(optimizer= opt, 
                  loss= [hyper.dice_loss_selected_class],)


# set callbacks
model_fld = os.path.join(cg.deep_dir, 'models', trial_name, 'models', 'batch_'+str(val_batch))
model_name = 'model' 
filepath=os.path.join(model_fld,  model_name +'-{epoch:03d}.hdf5')
ff.make_folder([os.path.dirname(os.path.dirname(model_fld)), os.path.dirname(model_fld), model_fld, os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs')])
csv_logger = CSVLogger(os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs',model_name + '_batch'+ str(val_batch) + '_training-log.csv')) 
callbacks = [csv_logger,
                    ModelCheckpoint(filepath,          
                                    monitor='val_loss',
                                    save_best_only=False,),
                     LearningRateScheduler(hyper.learning_rate_step_decay_classic),   
                    ]



