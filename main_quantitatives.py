# quantitative analysis of the segmentation prediction

import os
import time
import numpy as np
import nibabel as nb
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from tensorflow.keras.optimizers import Adam
from models import deep_strain_model
from utils import myocardial_strain
from scipy.ndimage import gaussian_filter

import DeepStrain.functions_collection as ff
import DeepStrain.Defaults as Defaults
from DeepStrain.data import base_dataset
import DeepStrain.Data_processing as Data_processing

cg = Defaults.Parameters()

patient_list = ff.find_all_target_files(['ID_*/SAX_ED_seg.nii.gz', 'ID_*/SAX_ES_seg.nii.gz'],os.path.join(cg.data_dir,'nii_manual_seg'))
patient_info = pd.read_excel(os.path.join(cg.data_dir,'Patient_list/Important_HFpEF_Patient_list_unique_patient_w_readmission_finalized.xlsx'))

LV_dice_list = []
myo_dice_list = []

for patient in patient_list:
    patient_id = os.path.basename(os.path.dirname(patient))

    patient_id_num = ff.ID_00XX_to_XX(patient_id)
    patient_row = patient_info.loc[patient_info['OurID'] == patient_id_num]
    ED = patient_row['ED'].values[0].astype(int)
    ES = patient_row['ES_by_visual'].values[0].astype(int)

    if 'ED' in os.path.basename(patient):
        this_phase = 'ED'
    else:
        this_phase = 'ES'
    # print(patient_id, ED, ES, this_phase)

    # load ground truth
    gt = nb.load(patient).get_fdata()
    gt = np.round(gt).astype(int)
    
    # load prediction
    if this_phase == 'ED':
        pred_file = os.path.join(cg.deep_dir, 'results/fine_tune_carson/seg', patient_id, 'pred_seg_frame' + str(ED) +'.nii.gz')
    else:
        pred_file = os.path.join(cg.deep_dir, 'results/fine_tune_carson/seg', patient_id, 'pred_seg_frame' + str(ES) +'.nii.gz')

    pred = nb.load(pred_file).get_fdata()
    pred = np.round(pred).astype(int)

    # calculate dice
    LV_dice = ff.np_categorical_dice(pred, gt, 1)
    myo_dice = ff.np_categorical_dice(pred, gt, 2)

    LV_dice_list.append(LV_dice)
    myo_dice_list.append(myo_dice)

print('LV dice: ', np.mean(LV_dice_list), np.std(LV_dice_list))
print('myo dice: ', np.mean(myo_dice_list), np.std(myo_dice_list))
    

