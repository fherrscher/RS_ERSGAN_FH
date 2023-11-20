import torch
import config
import numpy as np
import rasterio as rio
from rasterio.transform import Affine


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def load_epoch(checkpoint_file, epoch):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    epoch = checkpoint["epoch"]
    return epoch


def plot_examples(input_image, gen):

    gen.eval()

    original = rio.open(input_image)
    image_array = np.array(original.read())
    image_array = np.transpose(image_array, (1, 2, 0))

    with torch.no_grad():
        upscaled_img = gen(
            config.test_transform(image=image_array)["image"]
            .unsqueeze(0)
            .to(config.DEVICE)
        )

    upscaled_array = upscaled_img.squeeze().cpu().numpy()
    width = upscaled_array.shape[2]
    height = upscaled_array.shape[1]

    new_resolution = (original.transform.a / config.SCALING_FACTOR, original.transform.e / config.SCALING_FACTOR)
    transform = Affine(new_resolution[0], original.transform.b, original.transform.c, original.transform.d, new_resolution[1], original.transform.f)

    output_path = input_image.replace('.tif', '_upscaled.tif')
    with rio.open(output_path, 'w', driver='GTiff', width=width, height=height, count=4, dtype=str(upscaled_array.dtype), transform=transform, crs=original.crs) as dst:
        dst.write(upscaled_array)

    original.close()

    gen.train()




def plot_tensorboard(gen):
    gen.eval()    

    original = rio.open(config.EXAMPLE_IMAGE)
    image_array = np.array(original.read())
    image_array = np.transpose(image_array, (1, 2, 0))
    original.close()

    with torch.no_grad():
        upscaled_img = gen(
            config.test_transform(image=image_array)["image"]
            .unsqueeze(0)
            .to(config.DEVICE)
        )

    gen.train()
    
    new_tensor = upscaled_img[:, :3, :, :]

    return new_tensor
  