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

main_path = '/mnt/camca_NAS/Deepstrain'

# define model
class Options():
    
    def __init__(self):
        
        self.datadir = os.path.join(main_path,'data/')
        self.isTrain = False
        self.volume_shape = (128,128,16,1) # in network the input shape is [None, 128, 128, 16,2]
        self.pretrained_models_netS  = os.path.join(main_path,'models/trained/carson_Jan2021.h5')
        self.pretrained_models_netME = os.path.join(main_path,'models/trained/carmen_Jan2021.h5')
        # self.pretrained_models_netME = os.path.join(main_path,'models/fine_tune_carmen/ID_0015/models/model-005.hdf5')
        
opt = Options()

model = deep_strain_model.DeepStrain(Adam, opt=opt)
netME = model.get_netME()


# find all the patients:
patient_list = pd.read_excel(os.path.join('/mnt/camca_NAS/HFpEF/data/HFpEF_data/Patient_list', 'Important_HFpEF_Patient_list_unique_patient_w_ZCnotes.xlsx'))
patient_list = patient_list[patient_list['include?'] == 'yes']
patient_list = patient_list[patient_list['StudyDate']>20150219]
print(patient_list.shape)


# iterate over all patients
for i in range(0, patient_list.shape[0]):
    patient_num = patient_list['OurID'].iloc[i]
    patient_id = ff.XX_to_ID_00XX(patient_num)
    # if patient_id != 'ID_0003':
    #     continue
    print(patient_id)

    # save folder
    strain_save_folder = os.path.join(main_path, 'results/MVF', patient_id)
    ff.make_folder([strain_save_folder])

    # if os.path.isfile(os.path.join(strain_save_folder, 'mvf_template0_target14.nii.gz')):
    #     print('already predicted')
    #     continue    

    # define data folders
    patient_img_folder = os.path.join('/mnt/camca_NAS/SAM_for_CMR/SAM_seg_final_version',patient_id, 'img')
    patient_seg_folder = os.path.join('/mnt/camca_NAS/SAM_for_CMR/SAM_seg_final_version',patient_id, 'seg')

    # define template tf and target tf
    template_tf = 0

   
    for target_tf in range(0,15):
        print('current time frame pairs: ', template_tf, target_tf)

        # load data
        V_nifti_1 = nb.load(os.path.join(patient_img_folder, 'img_tf' + str(template_tf) + '.nii.gz'))
        V_nifti_2 = nb.load(os.path.join(patient_img_folder, 'img_tf' + str(target_tf) + '.nii.gz'))     

        M_nifti_1 = nb.load(os.path.join(patient_seg_folder, 'seg_tf' + str(template_tf) + '.nii.gz'))
        M_nifti_2 = nb.load(os.path.join(patient_seg_folder ,'seg_Tf' + str(target_tf) + '.nii.gz'))

        # M_ED = np.round(M_nifti_ED.get_fdata()).astype(int)
        # M_ED[M_ED==1] = 3
        # M_nifti_ED = nb.Nifti1Image(M_ED, affine=M_nifti_ED.affine, header=M_nifti_ED.header)

        # M_ES = np.round(M_nifti_ES.get_fdata()).astype(int)
        # M_ES[M_ES==1] = 3
        # M_nifti_ES = nb.Nifti1Image(M_ES, affine=M_nifti_ES.affine, header=M_nifti_ES.header)

        # prepare input
        V_nifti = nb.funcs.concat_images((V_nifti_1, V_nifti_2))
        M_nifti = nb.funcs.concat_images((M_nifti_1, M_nifti_2))

        # data was trained with:
        #  in-plane resolution of 1.25 mm x 1.25 mm
        #  number of slices = 16
        #  variable slice thickness since we specify number of slices
        V_nifti = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16)
        M_nifti = resample_nifti(M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16)
        

        # calculate center of mass using the first frame as reference. This is needed for cropping to 128x128
        center = center_of_mass(M_nifti.get_fdata()[:,:,:,0]==2) 
        V = _roll2center_crop(x=V_nifti.get_fdata(), center=center)
        M = _roll2center_crop(x=M_nifti.get_fdata(), center=center)

        # # save preprocessed image
        # preprocessed_input_folder = os.path.join(main_path, 'results/preprocessed_inputs', patient_id)
        # ff.make_folder([preprocessed_input_folder, os.path.join(preprocessed_input_folder, 'img'), os.path.join(preprocessed_input_folder, 'seg'), os.path.join(preprocessed_input_folder, 'img_cropped'), os.path.join(preprocessed_input_folder, 'seg_cropped')])
        # nb.save(nb.Nifti1Image(V_nifti.get_fdata()[:,:,:,1], affine=V_nifti.affine, header=V_nifti.header), os.path.join(preprocessed_input_folder, 'img/img_tf' + str(target_tf) + '.nii.gz'))
        # nb.save(nb.Nifti1Image(M_nifti.get_fdata()[:,:,:,1], affine=M_nifti.affine, header=M_nifti.header), os.path.join(preprocessed_input_folder, 'seg/seg_tf' + str(target_tf) + '.nii.gz'))
        # nb.save(nb.Nifti1Image(V[:,:,:,1], affine=V_nifti.affine, header=V_nifti.header), os.path.join(preprocessed_input_folder, 'img_cropped/img_tf' + str(target_tf) + '.nii.gz'))
        # nb.save(nb.Nifti1Image(M[:,:,:,1], affine=M_nifti.affine, header=M_nifti.header), os.path.join(preprocessed_input_folder, 'seg_cropped/seg_tf' + str(target_tf) + '.nii.gz'))



        # predict
        V = ff.normalize_image(V)
        nx, ny, nz, nt = V.shape
        V_0 =  np.repeat(V[:,:,:,:1], nt-1, axis=-1)
        V_t =  V[:,:,:,1:]

        V_0 = np.transpose(V_0, (3,0,1,2))
        V_t = np.transpose(V_t, (3,0,1,2))
        print('before input into the model, the shape: ', V_0.shape, V_t.shape)

        # predict motion vector
        for kk in range(0,1000000000000000000000000000000000000000000000000000000000000000**2):
            y_t = netME([V_0, V_t]).numpy()
            y_t = gaussian_filter(y_t, sigma=(0,2,2,0,0))

        # save
        # a = nb.Nifti1Image(y_t[0,:,:,:,:] , affine=V_nifti.affine, header=V_nifti.header)
        # filename =  'mvf_template' + str(template_tf) + '_target'+ str(target_tf) + '.nii.gz'
        # nb.save(a, os.path.join(strain_save_folder, filename))
# 
        
        
       


