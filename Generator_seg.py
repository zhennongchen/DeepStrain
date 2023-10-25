# tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import tensorflow as tf
import nibabel as nb
from tensorflow.keras.utils import Sequence
import DeepStrain.functions_collection as ff
import DeepStrain.models.networks as network
import DeepStrain.Defaults as Defaults
import DeepStrain.Hyperparameters as hyper
import DeepStrain.Build_list.Build_list as Build_list
import DeepStrain.Data_processing as Data_processing

cg = Defaults.Parameters()


class DataGenerator(Sequence):

    def __init__(self,
                img_file_list,
                seg_file_list,
                pred_seg_file_list,
                batch_size,
                num_classes,
                patient_num = None, 
                slice_num = None, 
                img_shape = None,
                shuffle = None,
                augment = None,
                normalize = None,
                seed = 10):

        self.img_file_list = img_file_list
        self.seg_file_list = seg_file_list
        self.pred_seg_file_list = pred_seg_file_list
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.patient_num = patient_num
        self.slice_num = slice_num
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        self.seed = seed


        self.on_epoch_end()
        
    def __len__(self):
        return self.img_file_list.shape[0] *  ( self.slice_num) // (self.batch_size)

    def on_epoch_end(self):
        
        self.seed += 1
        # print('seed is: ',self.seed)

        self.indices = []

        if self.shuffle == False:
            patient_list = np.arange(self.patient_num)
        else:
            patient_list = np.random.permutation(self.patient_num)

        for p in patient_list:
            if self.shuffle == False:
                slice_list = np.arange(self.slice_num)
            else:
                slice_list = np.random.permutation(self.slice_num)
            for s in slice_list:
                self.indices.append([p,s])
        
        self.indices = np.asarray(self.indices)
        # print('all indexes: ', self.indices,len(self.indices))

    def __getitem__(self,index):

        # 'Generate indexes of the batch'
        total_slice = self.patient_num * self.slice_num
        current_index = (index * self.batch_size) % total_slice
        if total_slice > current_index + self.batch_size:   # the total number of cases is adequate for next loop
            current_batch_size = self.batch_size
        else:
            current_batch_size = total_slice - current_index  # approaching to the tend, not adequate, should reduce the batch size
        
        indexes = self.indices[current_index : current_index + current_batch_size]

        # print('indexes in this batch: ',indexes)

        # allocate memory
        batch_x = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1]]) + (1,))
        batch_y = np.zeros(tuple([self.batch_size]) + tuple([self.img_shape[0],self.img_shape[1]]) + (self.num_classes,))
        
        volume_already_load = []
        load = False

        for i, j in enumerate(indexes):
            # print('i and j: ',i,j)
            case = j[0]
            # Is it a new case so I need to load the image volume?
            if i == 0:
                volume_already_load.append(case)
                load = True
                # print('let us start', i, case, volume_already_load[0],load)
            else:
                if case == volume_already_load[0]:
                    load = False
                    # print(i, case, volume_already_load[0],load)
                else:
                    load = True
                    volume_already_load[0] = case
                    # print(i, case, volume_already_load[0],load)
                
            if load == True:
                x_file = self.img_file_list[j[0]]; x = nb.load(x_file).get_fdata()
                # print('now loading x_file : ',x_file, ' shape: ',x.shape) 
                y_file = self.seg_file_list[j[0]]; y = np.round(nb.load(y_file).get_fdata()).astype(np.int32)
                assert np.unique(y).shape[0] == 3
                # print('now loading y_file : ',y_file, ' shape: ',y.shape, ' unique: ',np.unique(y))
                pred_seg_file = self.pred_seg_file_list[j[0]]; pred_seg = np.round(nb.load(pred_seg_file).get_fdata()).astype(np.int32)
                assert np.unique(pred_seg).shape[0] >= 3
                # print('now loading pred_seg_file : ',pred_seg_file, ' shape: ',pred_seg.shape, ' unique: ',np.unique(pred_seg))

                # find out the slices that are not zero according to y
                heart_slices = [z for z in range(y.shape[2]) if np.sum(y[:, :, z]) != 0]
                assert len(heart_slices) > 0; assert len(heart_slices) <= self.slice_num
                final_slices = ff.pick_slices(np.arange(0,y.shape[2]), heart_slices, self.slice_num)
        
                x = x[:, :, final_slices]
                y = y[:, :, final_slices]
                pred_seg = pred_seg[:, :, final_slices]
                # print('after picking slices, x, y, pred_seg shape: ',x.shape, y.shape, pred_seg.shape)

                # add pred_seg results to y and then relabel:
                yy = np.zeros(y.shape)
                yy[pred_seg == 3] = 3
                yy[y == 1] = 1
                yy[y == 2] = 2
                yy = Data_processing.relabel(yy, original_label=1, new_label=4)
                yy = Data_processing.relabel(yy, original_label=3, new_label=1)
                yy = Data_processing.relabel(yy, original_label=4, new_label=3)

                y = np.copy(yy).astype(np.int32)

                x = Data_processing.crop_or_pad(x, [x.shape[0], x.shape[1], self.slice_num])
                y = Data_processing.crop_or_pad(y, [y.shape[0], y.shape[1], self.slice_num])

            # pick slice
            img_x = x[:, :, j[1]]
            img_y = y[:, :, j[1]]

            # pick 128x128 patch according to center
            center = [img_x.shape[0]//2, img_x.shape[1]//2]
            # print('original center: ',center)
            if self.augment == True:
                # pick random 128x128 patch
                while True:
                    shift_x = int(np.random.uniform(-10, 10))
                    shift_y = int(np.random.uniform(-10, 10))
                    center_tem = [center[0] + shift_x, center[1] + shift_y]
                    if center_tem[0] - 64 >= 0 and center_tem[0] + 64 <= img_x.shape[0] and center_tem[1] - 64 >= 0 and center_tem[1] + 64 <= img_x.shape[1]:
                        center = center_tem
                        break
            else:
                center = center
            # print('finally center: ',center)
            
            img_x = img_x[center[0]-64:center[0]+64,center[1]-64:center[1]+64]
            img_y = img_y[center[0]-64:center[0]+64,center[1]-64:center[1]+64]

            if self.augment == True:
                # random flip
                flip = np.random.randint(0,2)
                if flip == 1:
                    img_x = np.flip(img_x, axis = 0)
                    img_y = np.flip(img_y, axis = 0)
                flip = np.random.randint(0,2)
                if flip == 1:
                    img_x = np.flip(img_x, axis = 1)
                    img_y = np.flip(img_y, axis = 1)


            # add one axis in the last or do the one-hot encoding
            img_x = np.expand_dims(img_x, axis = -1)
            
            img_y = Data_processing.one_hot(img_y, num_classes = self.num_classes)

            # print('finally img_x and img_y shape: ',img_x.shape, img_y.shape)

            batch_x[i] = img_x
            batch_y[i] = img_y

        if self.normalize:
            batch_x = ff.normalize_image(batch_x, axis = (1,2))
    
        return batch_x, batch_y