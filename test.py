import torch
import config
from torch import optim
from utils import load_checkpoint, plot_examples
from model import Generator

torch.backends.cudnn.benchmark = True


image_path = "test_images/Clip4328.tif"


gen = Generator(in_channels=4).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
load_checkpoint(
    config.CHECKPOINT_GEN,
    gen,
    opt_gen,
    config.LEARNING_RATE,
)
plot_examples(image_path, gen)
