# System
import os

class Parameters():

  def __init__(self):
  
    # # Number of partitions in the crossvalidation.
    # self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])
    
    # Dimension of padded input, for training.
    self.dim = (int(os.environ['CG_INPUT_X']), int(os.environ['CG_INPUT_Y']))

    # Seed for randomization.
    self.seed = int(os.environ['CG_SEED'])

    # classes:
    self.num_classes = int(os.environ['CG_NUM_CLASSES'])

    # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])
    self.initial_power = float(os.environ['CG_INITIAL_POWER'])
    self.start_epoch = int(os.environ['CG_START_EPOCH'])
    self.decay_rate = float(os.environ['CG_DECAY_RATE'])
    self.regularizer_coeff = float(os.environ['CG_REGULARIZER_COEFF'])

    # folders
    self.deep_dir = os.environ['CG_DEEP_DIR']
    self.data_dir = os.environ['CG_DATA_DIR']
