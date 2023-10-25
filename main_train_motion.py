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
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax

import DeepStrain.functions_collection as ff
import DeepStrain.models.networks as network
import DeepStrain.Defaults as Defaults
import DeepStrain.Hyperparameters as hyper
import DeepStrain.Build_list.Build_list as Build_list
import DeepStrain.Generator_seg as Generator_seg
from DeepStrain.models.dense_image_warp import dense_image_warp3d as warp

cg = Defaults.Parameters()


trial_name = 'fine_tune_carmen'
data_sheet = os.path.join(cg.deep_dir,'data/Patient_list','Patient_list_for_Seg.xlsx')

# build list
b = Build_list.Build(data_sheet)
_,_,_,_,_,_, img_file_trn, seg_file_trn, pred_seg_file_trn,_ = b.__build__(batch_list = [0])

n = np.arange(1,2,1)
img_file_trn = img_file_trn[n]; seg_file_trn = seg_file_trn[n]; pred_seg_file_trn = pred_seg_file_trn[n]
print('img_file_trn.shape: ', img_file_trn.shape, 'seg_file_trn.shape: ', seg_file_trn.shape, 'pred_seg_file_trn.shape: ', pred_seg_file_trn.shape)
print(img_file_trn[0:5], seg_file_trn[0:5], pred_seg_file_trn[0:5])


# create model
V_0_input = Input(shape = [128,128,16,1]) 
V_t_input = Input(shape = [128,128,16,1]) 
M_0_input = Input(shape = [128,128,16,3])
M_t_input = Input(shape = [128,128,16,3])
M_t_split = tf.split(M_t_input, M_t_input.shape[-1], -1)

input   = Concatenate(axis=-1)([V_0_input, V_t_input])
motion_estimates = network.encoder_decoder(input, nchannels=3, map_activation=None)
V_0_pred = warp(V_t_input, motion_estimates)
M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t_input)), motion_estimates) for mt in M_t_split], -1)    
M_0_pred  = softmax(M_0_pred) 
print('shape: ', V_0_pred.shape, M_0_pred.shape)

model = Model(inputs = [V_0_input, V_t_input, M_0_input, M_t_input], outputs = [motion_estimates,V_0_pred, M_0_pred])

model_file =  os.path.join(cg.deep_dir,'models/trained/carmen_Jan2021.h5')
model.load_weights(model_file)


# load data


# # compile model
# opt = Adam(lr = 1e-4)
# model.compile(optimizer= opt, 
#                   loss= [hyper.dice_loss_selected_class],)

# # set callbacks
# model_fld = os.path.join(cg.deep_dir, 'models', trial_name, 'models', 'batch_'+str(val_batch))
# model_name = 'model' 
# filepath=os.path.join(model_fld,  model_name +'-{epoch:03d}.hdf5')
# ff.make_folder([os.path.dirname(os.path.dirname(model_fld)), os.path.dirname(model_fld), model_fld, os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs')])
# csv_logger = CSVLogger(os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs',model_name + '_batch'+ str(val_batch) + '_training-log.csv')) 

# callbacks = [csv_logger,
#                     ModelCheckpoint(filepath,          
#                                     monitor='val_loss',
#                                     save_best_only=False,),
#                      LearningRateScheduler(hyper.learning_rate_step_decay_classic),   
#                     ]

# # Fit
# datagen = Generator_seg.DataGenerator(img_file_trn ,
#                                     seg_file_trn,
#                                     pred_seg_file_list = pred_seg_file_trn,
#                                     batch_size = 16,
#                                     num_classes = 4,
#                                     patient_num = img_file_trn.shape[0], 
#                                     slice_num = 16, 
#                                     img_shape = [cg.dim[0],cg.dim[1]],
#                                     shuffle = True,
#                                     augment = True,
#                                     normalize = True,
#                                      )

# valgen = Generator_seg.DataGenerator(img_file_val ,
#                                     seg_file_val,
#                                     pred_seg_file_list = pred_seg_file_val,
#                                     batch_size = 16,
#                                     num_classes = 4,
#                                     patient_num = img_file_val.shape[0],
#                                     slice_num = 16, 
#                                     img_shape = [cg.dim[0],cg.dim[1]],
#                                     shuffle = False,
#                                     augment = False,
#                                     normalize = True,
#                                      ) 


# model.fit_generator(generator = datagen,
#                     epochs = 200,
#                     validation_data = valgen,
#                     callbacks = callbacks,
#                     verbose = 1,)



