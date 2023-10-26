# tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import os
import nibabel as nb

from tensorflow.keras.utils import Sequence
import DeepStrain.functions_collection as ff
import DeepStrain.Defaults as Defaults
import DeepStrain.Data_processing as Data_processing

from DeepStrain.data.nifti_dataset import resample_nifti
from DeepStrain.data.base_dataset import _roll2center_crop
from scipy.ndimage.measurements import center_of_mass

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

        
        for i, j in enumerate(indexes):
            delta = self.target_tf_delta[j]
            target_tf = (self.template_tf + delta)
            if target_tf > 25:
                target_tf = target_tf - 25
            # print('index here: ', i, j,delta, ' target time frame: ', target_tf)

            v0_file = os.path.join(self.image_folder, 'Org3D_frame' + str(self.template_tf) + '.nii.gz' )
            m0_file = os.path.join(self.seg_folder, 'pred_seg_frame' + str(self.template_tf) + '.nii.gz' )
            if os.path.isfile(m0_file) == 0:
                m0_file = os.path.join(self.seg_folder, 'SAX_ED_seg.nii.gz' )

            vt_file = os.path.join(self.image_folder, 'Org3D_frame' + str(target_tf) + '.nii.gz' )
            mt_file = os.path.join(self.seg_folder, 'pred_seg_frame' + str(target_tf) + '.nii.gz' )
            if os.path.isfile(mt_file) == 0:
                mt_file = os.path.join(self.seg_folder, 'SAX_ES_seg.nii.gz' )

            V_nifti_ED = nb.load(v0_file)
            V_nifti_ES = nb.load(vt_file)
            M_nifti_ED = nb.load(m0_file)
            M_nifti_ES = nb.load(mt_file)

            M_ED = np.round(M_nifti_ED.get_fdata()).astype(int)
            M_nifti_ED = nb.Nifti1Image(M_ED, affine=M_nifti_ED.affine, header=M_nifti_ED.header)
            M_ES = np.round(M_nifti_ES.get_fdata()).astype(int)
            M_nifti_ES = nb.Nifti1Image(M_ES, affine=M_nifti_ES.affine, header=M_nifti_ES.header)

            # fourth: prepare input
            V_nifti = nb.funcs.concat_images((V_nifti_ED, V_nifti_ES))
            M_nifti = nb.funcs.concat_images((M_nifti_ED, M_nifti_ES))

            V_nifti = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16)
            M_nifti = resample_nifti(M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16)

            # calculate center of mass using the first frame as reference. This is needed for cropping to 128x128
            center = center_of_mass(M_nifti.get_fdata()[:,:,:,0]==2) # RV = 1, Myocardium = 2, LV = 3
            print('center: ',center)
            V = _roll2center_crop(x=V_nifti.get_fdata(), center=center)
            M = _roll2center_crop(x=M_nifti.get_fdata(), center=center)
      
            # 
            V = ff.normalize_image(V)
            nx, ny, nz, nt = V.shape
            V_0 =  np.repeat(V[:,:,:,:1], nt-1, axis=-1)
            V_t =  V[:,:,:,1:]
            M_0 =  Data_processing.one_hot(np.round(M[:,:,:,0]).astype(np.int), num_classes = 3)
            M_t =  Data_processing.one_hot(np.round(M[:,:,:,0]).astype(np.int), num_classes = 3)
            # print('after normalize, shape: ',V_0.shape, V_t.shape, M_0.shape, M_t.shape, ' Labels: ', np.unique(M_t))

            batch_v0_input[i] = V_0
            batch_vt_input[i] = V_t
            batch_mt_input[i] = M_t
            batch_v0_output[i] = V_0
            batch_m0_output[i] = M_0
            # batch_MVF_output[i] = nb.load(os.path.join(cg.deep_dir, 'models/fine_tune_carmen/ID_0015/mvf.nii.gz')).get_fdata()
        return [batch_v0_input, batch_vt_input, batch_mt_input], [batch_MVF_output, batch_v0_output, batch_m0_output]
    

# class DataGenerator(Sequence):

#     def __init__(self,
#                 patient_id,
#                 image_folder,
#                 seg_folder,
#                 template_tf,
#                 target_tf_delta,  # target time frame - template time frame
#                 heart_slices, 
#                 batch_size = 1,
#                 img_shape = None,        
#                 normalize = None,
#                 seed = 10):
        
#         self.patient_id = patient_id
#         self.image_folder = image_folder
#         self.seg_folder = seg_folder
#         self.template_tf = template_tf
#         self.target_tf_delta = target_tf_delta
#         self.heart_slices = heart_slices
#         self.batch_size = batch_size
#         self.img_shape = img_shape
#         self.normalize = normalize
#         self.seed = seed

#         self.on_epoch_end()
        
#     def __len__(self):
#         return self.target_tf_delta.shape[0] // (self.batch_size)

#     def on_epoch_end(self):
#         self.seed += 1

#         self.indices = np.arange(self.target_tf_delta.shape[0])

#         # print('all indexes: ', self.indices,len(self.indices))

#     def __getitem__(self,index):

#         # 'Generate indexes of the batch'
#         current_index = (index * self.batch_size) % self.target_tf_delta.shape[0]
        
#         indexes = self.indices[current_index : current_index + self.batch_size]

#         # print('indexes in this batch: ',indexes)

#         # allocate memory
#         batch_v0_input = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (1,))
#         batch_vt_input = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (1,))
#         batch_mt_input = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (3,))

#         batch_MVF_output = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (3,))
#         batch_v0_output = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (1,))
#         batch_m0_output = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1], self.img_shape[2]]) + (3,))

#         # load template frame:
#         v0_file = os.path.join(self.image_folder, 'Org3D_frame' + str(self.template_tf) + '.nii.gz' )
#         m0_file = os.path.join(self.seg_folder, 'pred_seg_frame' + str(self.template_tf) + '.nii.gz' )
#         if os.path.isfile(m0_file) == 0:
#             m0_file = os.path.join(self.seg_folder, 'SAX_ED_seg.nii.gz' )
            
#         v0_nii = nb.load(v0_file)
#         m0_nii = nb.load(m0_file)

#         v0_nii = resample_nifti(v0_nii, order=1, in_plane_resolution_mm=1.25, number_of_slices=16)
#         m0_nii = resample_nifti(m0_nii, order=0, in_plane_resolution_mm=1.25, number_of_slices=16)

#         v0 = v0_nii.get_fdata()
#         m0 = np.round(m0_nii.get_fdata()).astype(np.int)
#         center = center_of_mass(m0==2)
#         v0 = v0[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :]
#         m0 = m0[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :]

#         if self.normalize is not None:
#             v0 = ff.normalize(v0)
#         v0 = v0[...,np.newaxis]
#         m0 = Data_processing.one_hot(m0, num_classes = 3)
#         # print('v0 shape: ', v0.shape, 'm0 shape: ', m0.shape, 'max and min of v0: ', np.max(v0), np.min(v0), 'unique of m0: ', np.unique(m0))
        
#         for i, j in enumerate(indexes):
#             # load template time frame
#             delta = self.target_tf_delta[j]
#             target_tf = (self.template_tf + delta)
#             if target_tf > 25:
#                 target_tf = target_tf - 25
#             # print('index here: ', i, j,delta, ' target time frame: ', target_tf)
#             vt_file = os.path.join(self.image_folder, 'Org3D_frame' + str(target_tf) + '.nii.gz' )
#             mt_file = os.path.join(self.seg_folder, 'pred_seg_frame' + str(target_tf) + '.nii.gz' )
#             if os.path.isfile(mt_file) == 0:
#                 mt_file = os.path.join(self.seg_folder, 'SAX_ES_seg.nii.gz' )

#             vt_nii = nb.load(vt_file)
#             mt_nii = nb.load(mt_file)
#             vt_nii = resample_nifti(vt_nii, order=1, in_plane_resolution_mm=1.25, number_of_slices=16)
#             mt_nii = resample_nifti(mt_nii, order=0, in_plane_resolution_mm=1.25, number_of_slices=16)

#             vt = vt_nii.get_fdata()
#             mt = np.round(mt_nii.get_fdata()).astype(np.int)
#             vt = vt[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :]
#             mt = mt[int(center[0]) - 64 : int(center[0]) + 64, int(center[1]) - 64 : int(center[1]) + 64, :]


#             if self.normalize is not None:
#                 vt = ff.normalize(vt)
#             vt = vt[...,np.newaxis]
#             mt = Data_processing.one_hot(mt, num_classes = 3)
#         #     # print('vt shape: ', vt.shape, 'mt shape: ', mt.shape, 'max and min of vt: ', np.max(vt), np.min(vt), 'unique of mt: ', np.unique(mt))

#             batch_v0_input[i] = v0
#             batch_vt_input[i] = vt
#             batch_mt_input[i] = mt
#             batch_v0_output[i] = v0
#             batch_m0_output[i] = m0
#             batch_MVF_output[i] = nb.load(os.path.join(cg.deep_dir, 'models/fine_tune_carmen/ID_0015/mvf.nii.gz')).get_fdata()

#         return [batch_v0_input, batch_vt_input], [batch_MVF_output]
#         # return [batch_v0_input, batch_vt_input, batch_mt_input], [batch_MVF_output, batch_v0_output, batch_m0_output]

