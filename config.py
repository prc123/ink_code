import torch

class config ():
    PREFIX = './data/train/1/'

    PREFIX_list = ['./data/train/1/','./data/train/2/','./data/train/3/']
    rect_train_list = [(1100, 3500, 900, 750),(1200, 8500, 900, 750),(1200, 3000, 900, 750)]
    BUFFER = 32  # Buffer size in x and y direction
    Z_START = 27 # First slice in the z direction to use
    Z_DIM = 10   # Number of slices in the z direction
    TRAINING_STEPS = 6000
    LEARNING_RATE = 0.03
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPCHO = 1
    EVAL_STEPS = 30