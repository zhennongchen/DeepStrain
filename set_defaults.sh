## to run this in terminal, type:
# chmod +x set_defaults.sh
# . ./set_defaults.sh   

## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# volume dimension
export CG_INPUT_X=128 
export CG_INPUT_Y=128 

export CG_OUTPUT_X=128 
export CG_OUTPUT_Y=128 

# set number of classes
export CG_NUM_CLASSES=4

# set learning epochs
export CG_EPOCHS=200
export CG_LR_EPOCHS=25 # the number of epochs for learning rate change 
export CG_START_EPOCH=0
export CG_DECAY_RATE=0.01
export CG_INITIAL_POWER=-4
export CG_REGULARIZER_COEFF=0.2

# set random seed
export CG_SEED=8

# folders for Zhennong's dataset (change based on your folder paths)
export CG_DEEP_DIR="/mnt/mount_zc_NAS/Deepstrain/"
export CG_DATA_DIR="/mnt/mount_zc_NAS/HFpEF/data/HFpEF_data/"

