import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, load_epoch, plot_tensorboard
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True


if config.USE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(config.TB_LOG_DIR)


def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    epoch,
):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))

            fake_vgg = torch.narrow(fake, 1, 0, 3) # vgg works only with 3 channels
            high_res_vgg = torch.narrow(high_res, 1, 0, 3)

            loss_for_vgg = vgg_loss(fake_vgg, high_res_vgg)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )

    if config.USE_TENSORBOARD:
        writer.add_scalar('Adv_Loss', adversarial_loss.item(), global_step=epoch)
        writer.add_scalar('VGG_Loss', loss_for_vgg.item(), global_step=epoch)
        writer.add_scalar('Gen_Loss', gen_loss.item(), global_step=epoch)
        writer.add_scalar('Critic Loss', loss_critic.item(), global_step=epoch)
        writer.add_scalar('L1 Loss', l1_loss.item(), global_step=epoch)
        writer.add_scalar('GP', gp.item(), global_step=epoch)

        if epoch % config.SAVE_EPOCHS == 0:
            writer.add_image('Image Plot', plot_tensorboard(gen), global_step=epoch, dataformats='CHW')
        writer.flush()


def main():
    dataset = MyImageFolder(root_dir=config.INPUT_DIR)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=4).to(config.DEVICE)
    disc = Discriminator(in_channels=4).to(config.DEVICE)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    epoch_count = 0

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )
        epoch_count = load_epoch(config.CHECKPOINT_GEN, epoch_count)


    for epoch in range(epoch_count, config.NUM_EPOCHS):
        total_epoch = epoch + epoch_count + 1
        print("Epoch: ", total_epoch)
        train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            g_scaler,
            d_scaler,
            total_epoch,
        )

        if config.SAVE_MODEL and total_epoch % config.SAVE_EPOCHS == 0:
            save_checkpoint(gen, opt_gen, total_epoch,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, total_epoch, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
