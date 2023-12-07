import DeepStrain.Defaults as Defaults
import DeepStrain.functions_collection as ff

import os
import numpy as np
import nibabel as nb
import pandas as pd
import shutil
import SimpleITK as sitk

cg = Defaults.Parameters()


# delete files
patient_list = ff.find_all_target_files(['*/*strain.npy', '*/rotated*', '*/slice_info*','*/wtci*'],os.path.join(cg.deep_dir, 'results/strain'))
for p in patient_list:
    print(p)
    os.remove(p)
