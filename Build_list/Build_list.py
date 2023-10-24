import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        self.data = pd.read_excel(file_list)

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        patient_id_list = np.asarray(c['Patient_ID'])
        our_id_list = np.asarray(c['OurID'])
        batch_list = np.asarray(c['batch'])
        checked_list = np.asarray(c['checked'])
        ed_es_list = np.asarray(c['ED_ES'])
        tf_list = np.asarray(c['tf'])
        img_file_list = np.asarray(c['img_file'])
        seg_file_list = np.asarray(c['seg_file'])
        pred_seg_file_list = np.asarray(c['pred_seg_file'])
        nrrd_file_list = np.asarray(c['nrrd_file'])
        
        return patient_id_list,our_id_list,batch_list,checked_list,ed_es_list,tf_list,img_file_list,seg_file_list,pred_seg_file_list,nrrd_file_list
       
