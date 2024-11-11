import sys
sys.path.append("../")
import os
import numpy as np
import nibabel as nb
import pandas as pd
import matplotlib.pylab as plt
import SimpleITK as sitk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from data.nifti_dataset import resample_nifti
from data.base_dataset import _roll2center_crop
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label, regionprops

from utils import myocardial_strain_zc
from scipy.ndimage import gaussian_filter
import DeepStrain.functions_collection as ff
import DeepStrain.Data_processing as Data_processing
import DeepStrain.Defaults as Defaults


main_path = '/mnt/camca_NAS/Deepstrain'
patient_list = pd.read_excel(os.path.join('/mnt/camca_NAS/HFpEF/data/HFpEF_data/Patient_list', 'Important_HFpEF_Patient_list_unique_patient_w_ZCnotes.xlsx'))
patient_list = patient_list[patient_list['include?'] == 'yes']
patient_list = patient_list[patient_list['StudyDate']>20150219]
print(patient_list.shape)

save_folder = os.path.join(main_path, 'results/strain/')
for i in range(patient_list.shape[0]):
    patient_id_num = patient_list['OurID'].iloc[i]
    patient_id = ff.XX_to_ID_00XX(patient_id_num)
    if patient_id != 'ID_0809':
        continue
    print(i,patient_id)
    patient_save_folder = os.path.join(save_folder, patient_id)

    if os.path.isfile(os.path.join(patient_save_folder, 'tf_14/strain_info.npy')):
        print('already processed')
        continue

    # define the effective slices
    effective_slices_list = []
    for i in range(0,15):
        patient_seg_folder = os.path.join(main_path,'results/preprocessed_inputs',patient_id, 'seg')
        patient_seg = nb.load(os.path.join(patient_seg_folder, 'seg_tf' + str(i) + '.nii.gz')).get_fdata()
        effective_slice = []
        for j in range(patient_seg.shape[2]):
            if np.sum(patient_seg[:,:,j]==2)>0:
                effective_slice.append(j)
        effective_slices_list.append(effective_slice)

    sets = [set(lst) for lst in effective_slices_list]
    intersection = set.intersection(*sets)
    effective_slices = list(intersection)
    # print(effective_slices)

    # get the layer info
    slices_per_layer = len(effective_slices)//3
    mod = len(effective_slices)%3
    if mod == 1 or mod == 2:
        mod = 1

    apex_layer = effective_slices[len(effective_slices) - slices_per_layer : len(effective_slices)]
    mid_layer = effective_slices[len(effective_slices) - slices_per_layer * 2 -mod : len(effective_slices) - slices_per_layer]
    base_layer = effective_slices[0: len(effective_slices) - slices_per_layer * 2 -mod]
    mid_slice = effective_slices[len(effective_slices)//2]

    # assert the first slice is 'basal'
    a = patient_seg[:,:,effective_slices[3]]
    b = patient_seg[:,:,effective_slices[10]]
    assert np.sum(a>0) > np.sum(b>0)

    template_tf = 0

    for target_tf in range(0,15):
        tf1 = template_tf
        tf2 = target_tf

        patient_save_folder_timeframe = os.path.join(patient_save_folder, 'tf_' + str(tf2))
        ff.make_folder([patient_save_folder_timeframe])

        # load img and seg
        patient_img_folder = os.path.join(main_path,'results/preprocessed_inputs',patient_id, 'img_cropped')
        patient_seg_folder = os.path.join(main_path,'results/preprocessed_inputs',patient_id, 'seg_cropped')

        V_nifti_1 = nb.load(os.path.join(patient_img_folder, 'img_tf' + str(tf1) + '.nii.gz'))
        V_nifti_2 = nb.load(os.path.join(patient_img_folder, 'img_tf' + str(tf2) + '.nii.gz'))

        M_nifti_1 = nb.load(os.path.join(patient_seg_folder, 'seg_tf' + str(tf1) + '.nii.gz'))

        # get motion vector field
        y_t = nb.load(os.path.join(main_path, 'results/MVF', patient_id, 'mvf_template'+str(tf1) + '_target'+str(tf2)+'.nii.gz')).get_fdata()
        y_t = y_t[None,...]

        mask_tf1 = M_nifti_1.get_fdata()

        # # calculate global strain
        strain = myocardial_strain_zc.MyocardialStrain(mask=mask_tf1, flow=y_t[0,:,:,:,:])
        strain.calculate_strain(lv_label=2)

        global_radial_strain = strain.Err[strain.mask_rot==2].mean()
        global_circumferential_strain = strain.Ecc[strain.mask_rot==2].mean()

        ## rotate the image
        if os.path.isfile(os.path.join(patient_save_folder, 'insertion_points.npy')):
            # print('use saved')
            insertion_points = np.load(os.path.join(patient_save_folder, 'insertion_points.npy'))
            insertion_p1 = insertion_points[0,:]
            insertion_p2 = insertion_points[1,:]
        else:
            assert False, 'no there should be pre-defined insertion points!'

        # get the rotation angle ready
        img_ed = nb.load(os.path.join(patient_img_folder, 'img_tf' + str(tf1) + '.nii.gz')).get_fdata()
        phi_angle   , cx_lv, cy_lv, cx_rv, cy_rv  = myocardial_strain_zc._get_lv2rv_angle_using_insertion_points(strain.mask, insertion_p1, insertion_p2)
        # print('insertion_p1, insertion_p2, phi_angle: ', insertion_p1, insertion_p2, phi_angle)
        rotate_f = myocardial_strain_zc.Rotate_data(strain.Err, strain.Ecc, strain.mask,img_ed, insertion_p1, insertion_p2 )
        Err_rot, Ecc_rot, mask_rot, img_rot= rotate_f.rotate_orientation(for_visualization=False)
        # only keep the effective slices
        Err_rot = Err_rot[:,:,effective_slices]
        Ecc_rot = Ecc_rot[:,:,effective_slices]
        mask_rot = mask_rot[:,:,effective_slices]
        img_rot = img_rot[:,:,effective_slices]

        # polar strain
        polar = myocardial_strain_zc.PolarMap(Err_rot, Ecc_rot, mask_rot)
        polar_strain_result = polar.project_to_aha_polar_map()
        Ecc_polar, Ecc_aha = polar.construct_AHA_map(polar_strain_result['V_ecc'], start_slice_name = 'base', start = 20, stop = 80) # Ecc_aha first element is the mean for all AHA segments, followed by 16 segments + 1 apex (Set to 0)
        Err_polar, Err_aha = polar.construct_AHA_map(polar_strain_result['V_err'], start_slice_name = 'base', start = 20, stop = 80)
        # print('Ecc_polar shape: ', Ecc_polar.shape, 'Ecc_aha shape: ', len(Ecc_aha), 'Ecc_rot shape: ', Ecc_rot.shape)


        # save the results
        strain_info = [strain, global_radial_strain, global_circumferential_strain, Err_rot, Ecc_rot, mask_rot, Ecc_polar, Err_polar, Ecc_aha[1:], Err_aha[1:]]
        slice_info = [effective_slices, 'base', base_layer, mid_layer, apex_layer]

        # save all above
        np.save(os.path.join(patient_save_folder_timeframe,'strain_info.npy'), strain_info,allow_pickle=True)
        np.save(os.path.join(patient_save_folder,'slice_info.npy'), slice_info)