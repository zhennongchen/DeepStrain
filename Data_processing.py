import numpy as np
import nibabel as nb
import os
import DeepStrain.Defaults as Defaults
import DeepStrain.functions_collection as ff


def crop_or_pad(array, target, value=0):
    """
    Symmetrically pad or crop along each dimension to the specified target dimension.
    :param array: Array to be cropped / padded.
    :type array: array-like
    :param target: Target dimension.
    :type target: `int` or array-like of length array.ndim
    :returns: Cropped/padded array. 
    :rtype: array-like
    """
    # Pad each axis to at least the target.
    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]


def adapt(x, target, crop = True, expand_dims = True):
    x = nb.load(x).get_data()
    # clip the very high value
    if crop == True:
        x = crop_or_pad(x, target)
    if expand_dims == True:
        x = np.expand_dims(x, axis = -1)
    #   print('after adapt, shape of x is: ', x.shape)
    return x


def normalize_image(x):
    # a common normalization method in CT
    # if you use (x-mu)/std, you need to preset the mu and std
    
    return x.astype(np.float32) / 1000

def cutoff_intensity(x,cutoff):
 
    x[x<cutoff] = cutoff
    return x

def relabel(x,original_label,new_label):
    x[x==original_label] = new_label
    return x

def one_hot(image, num_classes):
    # Reshape the image to a 2D array
    image_2d = image.reshape(-1)

    # Perform one-hot encoding using NumPy's eye function
    encoded_image = np.eye(num_classes, dtype=np.uint8)[image_2d]

    # Reshape the encoded image back to the original shape
    encoded_image = encoded_image.reshape(image.shape + (num_classes,))

    return encoded_image


def save_partial_volumes(img_list,file_name,slice_range = None): # only save some slices of an original CT volume
    for img_file in img_list:
        f = os.path.join(os.path.dirname(img_file),file_name)

        if os.path.isfile(f) == 1:
            print('already saved partial volume')
            continue

        x = nb.load(img_file)
        img = x.get_data()
        print(img_file,img.shape)
        

        if slice_range == None:
            # slice_range = [int(img.shape[-1]/2) - 30, int(img.shape[-1]/2) + 30]
            slice_range = [0,60]
        
        if img.shape[-1] < (slice_range[1] - slice_range[0]):
            print('THIS ONE DOES NOT HAVE ENOUGH SLICES, CONTINUE')
            continue
        
        img = img[:,:,slice_range[0]:slice_range[1]]

        # ff.make_folder([f])
        img = nb.Nifti1Image(img,x.affine)
        nb.save(img, f)
