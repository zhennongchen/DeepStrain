import os
import time
import numpy as np
import nibabel as nb
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from data.nifti_dataset import resample_nifti
from data.base_dataset import _roll2center_crop
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label, regionprops

from tensorflow.keras.optimizers import Adam
from models import deep_strain_model
from utils import myocardial_strain
from scipy.ndimage import gaussian_filter

import DeepStrain.functions_collection as ff
import DeepStrain.Defaults as Defaults
from DeepStrain.data import base_dataset
import DeepStrain.Data_processing as Data_processing

cg = Defaults.Parameters()

def get_mask(V):  # by trained netS
    nx, ny, nz, nt = V.shape
    
    M = np.zeros((nx,ny,nz,nt))
    v = V.transpose((2,3,0,1)).reshape((-1,nx,ny)) # (nz*nt,nx,ny)

    v = ff.normalize_image(v)
   
    m = netS(v[:,nx//2-64:nx//2+64,ny//2-64:ny//2+64,None])
    M[nx//2-64:nx//2+64,ny//2-64:ny//2+64] += np.argmax(m, -1).transpose((1,2,0)).reshape((128,128,nz,nt))
    
    return M

class Options():
    
    def __init__(self):
        
        self.datadir = os.path.join(cg.deep_dir,'data/ACDC')
        self.isTrain = False
        self.image_shape = (128,128,1) # 2D
        self.nlabels = 4
        self.pretrained_models_netS  = os.path.join(cg.deep_dir,'models/fine_tune_carson/models/batch_9/model-113.hdf5')
        self.pretrained_models_netME = os.path.join(cg.deep_dir,'models/trained/carmen_Jan2021.h5')
        
opt = Options()

model = deep_strain_model.DeepStrain(Adam, opt=opt)
netS  = model.get_netS()

# find all the patients:
spreadsheet = pd.read_excel(os.path.join(cg.data_dir, 'Patient_list', 'Important_HFpEF_Patient_list_unique_patient_w_readmission_finalized.xlsx' ))
spreadsheet = spreadsheet.iloc[0:200]

# iterate over all patients
for i in range(0, spreadsheet.shape[0]):
    patient_num = spreadsheet['OurID'].iloc[i]
    patient_id = ff.XX_to_ID_00XX(patient_num)
    print(patient_id)

    # save folder
    seg_save_folder = os.path.join(cg.deep_dir,  'results/fine_tune_carson/seg', patient_id)
    ff.make_folder([seg_save_folder])

    if os.path.isfile(os.path.join(seg_save_folder, 'pred_seg_frame25.nii.gz')):
        print('already predicted')
        continue

    # find all the time frames
    patient_imgs= ff.sort_timeframe(ff.find_all_target_files(['Org3D*'], os.path.join(cg.data_dir, 'nii_img', patient_id)),2,'e')

    # second: load img 
    for kk in range(0,len(patient_imgs)):
        tf = ff.find_timeframe(patient_imgs[kk], 2, 'e')
        V_nifti = nb.load(patient_imgs[kk])
        original_shape = V_nifti.get_fdata().shape
        print(original_shape)
        V_nifti_resampled = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=None)
        V = V_nifti_resampled.get_fdata()

        V = ff.normalize_image(V, axis=(0,1))
        V = V[:,:,:,None]

        # fourth: get rough mask (not cropped and centered)
        M = get_mask(V)

        # fifth: crop and center
        center_resampled = center_of_mass(M[:,:,:,0]==2)
        V = base_dataset.roll_and_pad_256x256_to_center(x=V, center=center_resampled)
        M = base_dataset.roll_and_pad_256x256_to_center(x=M, center=center_resampled)
        center_resampled_256x256 = center_of_mass(M==3)
        # print('crop to 256x256, shape: ', V.shape, M.shape)
        # we save all this info to invert the segmentation bask to its original location/resolution
        nifti_info = {'affine'           : V_nifti.affine,
                    'affine_resampled' : V_nifti_resampled.affine,
                    'zooms'            : V_nifti.header.get_zooms(),
                    'zooms_resampled'  : V_nifti_resampled.header.get_zooms(),
                    'shape'            : V_nifti.shape,
                    'shape_resampled'  : V_nifti_resampled.shape,
                    'center_resampled' : center_resampled,
                    'center_resampled_256x256' : center_resampled_256x256} 

        # sixth: get mask again
        M = get_mask(V)[128-64:128+64,128-64:128+64]
        # print('in second segmentation output, shape: ', M.shape)
        # change label: label 3 becomes label 1, label 1 becomes label 3
        M_relabel = np.zeros(M.shape)
        M_relabel[M==3] = 1.
        M_relabel[M==1] = 3.
        M_relabel[M==2] = 2.
        M = np.copy(M_relabel)

        M_nifti = base_dataset.convert_back_to_nifti(M, nifti_info, inv_256x256=True, order=0, mode='nearest')
        print('final segmentation output, shape: ', M_nifti.shape)

        # make sure the dimension is the same:
        final_shape = M_nifti.shape
    
        if original_shape[0] != final_shape[0] or original_shape[1] != final_shape[1]:
            M_data = M_nifti.get_fdata()
            new_M_data = np.zeros([original_shape[0], original_shape[1], final_shape[2], final_shape[3]])
            for z in range(M_data.shape[3]):
                new_M_data[:,:,:,z] = Data_processing.crop_or_pad(M_data[:,:,:,z], [original_shape[0], original_shape[1], final_shape[2]],value = 0)
            M_nifti = nb.Nifti1Image(new_M_data, affine=M_nifti.affine)
            print('after correction: final segmentation output, shape: ', M_nifti.shape)

        # save
        a = np.round(M_nifti.get_fdata()[:,:,:,0])
        # remove the scatters
        a = ff.remove_scatter(a, 2)
        a = ff.remove_scatter(a, 1)
        a = ff.remove_scatter(a, 3)
        a = nb.Nifti1Image(a, affine=M_nifti.affine)
        nb.save(a, os.path.join(seg_save_folder, 'pred_seg_frame' + str(tf) + '.nii.gz'))