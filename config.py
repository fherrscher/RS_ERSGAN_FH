import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
INPUT_DIR = "data_patches/"
CHECKPOINT_GEN = "checkpoints/gen.pth"
CHECKPOINT_DISC = "checkpoints/disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 32
LAMBDA_GP = 10
NUM_WORKERS = 4
SCALING_FACTOR = 4
HIGH_RES = 128
LOW_RES = HIGH_RES // SCALING_FACTOR

USE_TENSORBOARD = True # Set to True to use tensorboard
TB_LOG_DIR = "runs/mnist/loc_04" # Tensorboard log dir
SAVE_EPOCHS = 10 # Every X epochs plot the examples and save the model
EXAMPLE_IMAGE = "test_images/test.tif" # Example image for Tensorboard



both_transforms = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0, 0], std=[1, 1, 1, 1]),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0, 0], std=[1, 1, 1, 1]),
        ToTensorV2(),
    ]
)
