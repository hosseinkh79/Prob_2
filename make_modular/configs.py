import os 

# dataset path
CURRENT_PATH = os.getcwd()
DATASET_PATH = os.path.join(CURRENT_PATH, 'mnist_dataset/') 
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train/") 
TEST_DATASET_PATH = os.path.join(DATASET_PATH, "test/") 

# batchsize
batch_size = 64