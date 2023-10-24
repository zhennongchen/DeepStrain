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
import DeepStrain.Build_list.Build_list as Build_list
import DeepStrain.Generator_seg as Generator_seg

cg = Defaults.Parameters()

trial_name = 'fine_tune_carson'
val_batch = 9
data_sheet = os.path.join(cg.deep_dir,'data/Patient_list','Patient_list_version1.xlsx')

# build list
batch_list = [0,1,2,3,4,5,6,7,8,9]; batch_list.pop(val_batch)
train_batch = batch_list
b = Build_list.Build(data_sheet)

_,_,_,_,_,_, img_file_trn, seg_file_trn, pred_seg_file_trn,_ = b.__build__(batch_list = train_batch)
_,_,_,_,_,_, img_file_val, seg_file_val, pred_seg_file_val,_= b.__build__(batch_list = [val_batch])

n = np.arange(0,2,1)
img_file_trn = img_file_trn[n]; seg_file_trn = seg_file_trn[n]; pred_seg_file_trn = pred_seg_file_trn[n]
n = np.arange(0,1,1)
img_file_val = img_file_val[n]; seg_file_val = seg_file_val[n]; pred_seg_file_val = pred_seg_file_val[n]

# create model
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

# Fit
datagen = Generator_seg.DataGenerator(img_file_trn ,
                                    seg_file_trn,
                                    pred_seg_file_list = pred_seg_file_trn,
                                    batch_size = 12,
                                    num_classes = 4,
                                    patient_num = img_file_trn.shape[0],
                                    slice_num = 12, 
                                    img_shape = [cg.dim[0],cg.dim[1]],
                                    shuffle = True,
                                     )

valgen = Generator_seg.DataGenerator(img_file_val ,
                                    seg_file_val,
                                    pred_seg_file_list = pred_seg_file_val,
                                    batch_size = 12,
                                    num_classes = 4,
                                    patient_num = img_file_val.shape[0],
                                    slice_num = 12, 
                                    img_shape = [cg.dim[0],cg.dim[1]],
                                    shuffle = False,
                                     ) 


model.fit_generator(generator = datagen,
                    epochs = 3,
                    validation_data = valgen,
                    callbacks = callbacks,
                    verbose = 1,)



