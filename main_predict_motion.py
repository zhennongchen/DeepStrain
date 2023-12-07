import sys
sys.path.append("../")
import os
import numpy as np
import nibabel as nb
import pandas as pd
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

cg = Defaults.Parameters()

# define model
class Options():
    
    def __init__(self):
        
        self.datadir = os.path.join(cg.data_dir)
        self.isTrain = False
        self.volume_shape = (128,128,16,1) # in network the input shape is [None, 128, 128, 16,2]
        self.pretrained_models_netS  = os.path.join(cg.deep_dir,'models/fine_tune_carson/models/batch_9/model_113.hdf5')
        self.pretrained_models_netME = os.path.join(cg.deep_dir,'models/trained/carmen_Jan2021.h5')
        
opt = Options()


model = deep_strain_model.DeepStrain(Adam, opt=opt)
netME = model.get_netME()


# find all the patients:
spreadsheet = pd.read_excel(os.path.join(cg.data_dir, 'Patient_list', 'Important_HFpEF_Patient_list_unique_patient_w_readmission_finalized.xlsx' ))
spreadsheet = spreadsheet.iloc[0:200]


# iterate over all patients
for i in range(0, spreadsheet.shape[0]):
    patient_num = spreadsheet['OurID'].iloc[i]
    patient_id = ff.XX_to_ID_00XX(patient_num)
    print(patient_id)

    # save folder
    strain_save_folder = os.path.join(cg.deep_dir, 'results/trained/mvf', patient_id)
    ff.make_folder([strain_save_folder])

    if os.path.isfile(os.path.join(strain_save_folder, 'mvf_ED_ES.nii.gz')):
        print('already predicted')
        continue    

    # define data folders
    patient_img_folder = os.path.join(cg.data_dir, 'nii_img', patient_id)
    patient_seg_folder = os.path.join(cg.data_dir, 'nii_manual_seg', patient_id)
    patient_seg_folder_2 = os.path.join(cg.deep_dir,'results/fine_tune_carson/seg/', patient_id)

    # define ED and ES
    ED = spreadsheet['ED'].iloc[i].astype(int)
    ES = spreadsheet['ES_by_visual'].iloc[i].astype(int)

    for k in range(0,25):
        if ED + k == 25:
            tf = 25
        else:
            tf = (ED + k)%25

        print('current time frame pairs: ', ED, tf)

        # load data
        V_nifti_ED = nb.load(os.path.join(patient_img_folder, 'Org3D_frame' + str(ED) + '.nii.gz'))
        V_nifti_ES = nb.load(os.path.join(patient_img_folder, 'Org3D_frame' + str(tf) + '.nii.gz'))     

        if os.path.isfile(os.path.join(patient_seg_folder, 'SAX_ED_seg.nii.gz')):
            print('using manual segmentation')
            M_nifti_ED = nb.load(os.path.join(patient_seg_folder, 'SAX_ED_seg.nii.gz'))
            M_nifti_ES = nb.load(os.path.join(patient_seg_folder, 'SAX_ES_seg.nii.gz'))
        else:
            print('using predicted segmentation')
            M_nifti_ED = nb.load(os.path.join(patient_seg_folder_2, 'pred_seg_frame' + str(ED) + '.nii.gz'))
            M_nifti_ES = nb.load(os.path.join(patient_seg_folder_2 ,'pred_seg_frame' + str(ES) + '.nii.gz'))

        M_ED = np.round(M_nifti_ED.get_fdata()).astype(int)
        M_ED[M_ED==1] = 3
        M_nifti_ED = nb.Nifti1Image(M_ED, affine=M_nifti_ED.affine, header=M_nifti_ED.header)

        M_ES = np.round(M_nifti_ES.get_fdata()).astype(int)
        M_ES[M_ES==1] = 3
        M_nifti_ES = nb.Nifti1Image(M_ES, affine=M_nifti_ES.affine, header=M_nifti_ES.header)

        # prepare input
        V_nifti = nb.funcs.concat_images((V_nifti_ED, V_nifti_ES))
        M_nifti = nb.funcs.concat_images((M_nifti_ED, M_nifti_ES))

        # data was trained with:
        #  in-plane resolution of 1.25 mm x 1.25 mm
        #  number of slices = 16
        #  variable slice thickness since we specify number of slices
        V_nifti = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16)
        M_nifti = resample_nifti(M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16)
        # print('after resample, shape: ', V_nifti.shape, M_nifti.shape)

        # calculate center of mass using the first frame as reference. This is needed for cropping to 128x128
        center = center_of_mass(M_nifti.get_fdata()[:,:,:,0]==2) # RV = 1, Myocardium = 2, LV = 3
        V = _roll2center_crop(x=V_nifti.get_fdata(), center=center)
        M = _roll2center_crop(x=M_nifti.get_fdata(), center=center)

        # predict
        V = ff.normalize_image(V)
        nx, ny, nz, nt = V.shape
        V_0 =  np.repeat(V[:,:,:,:1], nt-1, axis=-1)
        V_t =  V[:,:,:,1:]

        V_0 = np.transpose(V_0, (3,0,1,2))
        V_t = np.transpose(V_t, (3,0,1,2))
        # print('before input into the model, the shape: ', V_0.shape, V_t.shape)

        # predict motion vector
        y_t = netME([V_0, V_t]).numpy()
        y_t = gaussian_filter(y_t, sigma=(0,2,2,0,0))

        # save
        a = nb.Nifti1Image(y_t[0,:,:,:,:] , affine=V_nifti_ED.affine, header=V_nifti_ED.header)
        filename =  'mvf_ED' + str(ED) + '_tf'+ str(tf) + '.nii.gz'
        nb.save(a, os.path.join(strain_save_folder, filename))

        if tf == ES:
            nb.save(a, os.path.join(strain_save_folder, 'mvf_ED_ES.nii.gz'))

        
        
       


