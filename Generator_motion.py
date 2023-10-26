# tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import os
import nibabel as nb
from scipy.ndimage.measurements import center_of_mass
from tensorflow.keras.utils import Sequence
import DeepStrain.functions_collection as ff
import DeepStrain.Defaults as Defaults
import DeepStrain.Data_processing as Data_processing

cg = Defaults.Parameters()


class DataGenerator(Sequence):

    def __init__(self,
                patient_id,
                image_folder,
                seg_folder,
                template_tf,
                target_tf_delta,  # target time frame - template time frame
                heart_slices, 
                batch_size = 1,
                img_shape = None,        
                normalize = None,
                seed = 10):
        
        self.patient_id = patient_id
        self.image_folder = image_folder
        self.seg_folder = seg_folder
        self.template_tf = template_tf
        self.target_tf_delta = target_tf_delta
        self.heart_slices = heart_slices
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.normalize = normalize
        self.seed = seed

        self.on_epoch_end()
        
    def __len__(self):
        return self.target_tf_delta.shape[0] // (self.batch_size)

    def on_epoch_end(self):
        self.seed += 1

        self.indices = np.arange(self.target_tf_delta.shape[0])

        # print('all indexes: ', self.indices,len(self.indices))

    def __getitem__(self,index):

        # 'Generate indexes of the batch'
        current_index = (index * self.batch_size) % self.target_tf_delta.shape[0]
        
        indexes = self.indices[current_index : current_index + self.batch_size]

        # print('indexes in this batch: ',indexes)

        # allocate memory
        batch_v0_input = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (1,))
        batch_vt_input = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (1,))
        batch_mt_input = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (3,))

        batch_MVF_output = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (3,))
        batch_v0_output = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (1,))
        batch_m0_output = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (3,))

        # load template frame:
        v0_file = os.path.join(self.image_folder, 'Org3D_frame' + str(self.template_tf) + '.nii.gz' )
        m0_file = os.path.join(self.seg_folder, 'pred_seg_frame' + str(self.template_tf) + '.nii.gz' )
        if os.path.isfile(m0_file) == 0:
            m0_file = os.path.join(self.seg_folder, 'SAX_ED_seg.nii.gz' )
        v0 = nb.load(v0_file).get_fdata()[:,:,self.heart_slices[0]:self.heart_slices[1]]
        m0 = np.round(nb.load(m0_file).get_fdata()[:,:,self.heart_slices[0]:self.heart_slices[1]]).astype(np.int)
        center = center_of_mass(m0==2)
        v0 = Data_processing.crop_or_pad(v0[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :], self.img_shape, value = np.min(v0))
        m0 = Data_processing.crop_or_pad(m0[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :], self.img_shape, value = 0)

        if self.normalize is not None:
            v0 = ff.normalize(v0)
        v0 = v0[...,np.newaxis]
        m0 = Data_processing.one_hot(m0, num_classes = 3)
        # print('v0 shape: ', v0.shape, 'm0 shape: ', m0.shape, 'max and min of v0: ', np.max(v0), np.min(v0), 'unique of m0: ', np.unique(m0))
        
        for i, j in enumerate(indexes):
            # load template time frame
            delta = self.target_tf_delta[j]
            target_tf = (self.template_tf + delta)
            if target_tf > 25:
                target_tf = target_tf - 25
            # print('index here: ', i, j,delta, ' target time frame: ', target_tf)
            vt_file = os.path.join(self.image_folder, 'Org3D_frame' + str(target_tf) + '.nii.gz' )
            mt_file = os.path.join(self.seg_folder, 'pred_seg_frame' + str(target_tf) + '.nii.gz' )
            if os.path.isfile(mt_file) == 0:
                mt_file = os.path.join(self.seg_folder, 'SAX_ES_seg.nii.gz' )
            vt = nb.load(vt_file).get_fdata()[:,:,self.heart_slices[0]:self.heart_slices[1]]
            mt = np.round(nb.load(mt_file).get_fdata()[:,:,self.heart_slices[0]:self.heart_slices[1]]).astype(np.int)
            vt = Data_processing.crop_or_pad(vt[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :], self.img_shape, value = np.min(vt))
            mt = Data_processing.crop_or_pad(mt[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :], self.img_shape, value = 0)

            if self.normalize is not None:
                vt = ff.normalize(vt)
            vt = vt[...,np.newaxis]
            mt = Data_processing.one_hot(mt, num_classes = 3)
            # print('vt shape: ', vt.shape, 'mt shape: ', mt.shape, 'max and min of vt: ', np.max(vt), np.min(vt), 'unique of mt: ', np.unique(mt))

            batch_v0_input[i] = v0
            batch_vt_input[i] = vt
            batch_mt_input[i] = mt
            batch_v0_output[i] = v0
            batch_m0_output[i] = m0

        # return [batch_v0_input, batch_vt_input], [batch_MVF_output]
        return [batch_v0_input, batch_vt_input, batch_mt_input], [batch_MVF_output, batch_v0_output, batch_m0_output]