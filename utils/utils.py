import torch
from torchvision import transforms, datasets, utils as vutils
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


def prepare_celeba_data(path, batch_size, image_size, workers):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    data_set = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=workers)


def prepare_cartoon_data(path, batch_size, image_size, workers):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    data_set = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=workers)


def show_images(imgs, fig_size, num_of_images, title, name):
    # plt.figure(figsize=fig_size)
    # plt.axis("off")
    # plt.title(title)
    # plt.imshow(np.transpose(vutils.make_grid(imgs[:num_of_images], padding=2, normalize=True), (1, 2, 0)))
    plt.imsave('./data/' + name + '.png',
               np.transpose(vutils.make_grid(imgs[:num_of_images], nrow=4, padding=2, normalize=True), (1, 2, 0)))
    plt.show()


# generate real labels
def real_label(prediction, gpu):
    real = torch.tensor(1.0)
    data = real.expand_as(prediction)
    return data.cuda() if gpu else data


# generate fake labels
def fake_label(prediction, gpu):
    fake = torch.tensor(0.0)
    data = fake.expand_as(prediction)
    return data.cuda() if gpu else data


# Train discriminator
def train_discriminator(real_A, real_B, fake_B, net_D, loss, optim_D, gpu):
    optim_D.zero_grad()

    # Fake
    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = net_D(fake_AB)
    loss_D_fake = loss(pred_fake, fake_label(pred_fake, gpu))

    # Real
    real_AB = torch.cat((real_A, real_B), 1)
    pred_real = net_D(real_AB)
    loss_D_real = loss(pred_real, real_label(pred_real, gpu))

    loss_D = (loss_D_fake + loss_D_real) / 2
    loss_D.backward()

    optim_D.step()
    return torch.sum(loss_D).item()


# Train generator
def train_generator(real_A, real_B, fake_B, net_D, loss_G, loss_L1, optim_G, sf, gpu):
    optim_G.zero_grad()

    # 1. Fake the Discriminator
    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = net_D(fake_AB)
    loss_G_GAN = loss_G(pred_fake, real_label(pred_fake, gpu))

    # 2. Preserving the style while generating
    loss_G_L1 = loss_L1(fake_B, real_B)

    # combined loss
    loss_G = loss_G_GAN + sf * loss_G_L1

    loss_G.backward()

    optim_G.step()
    return torch.sum(loss_G).item()


# Train GAN
def train_gan(data_loader_src, data_loader_tgt, net_G, net_D, loss_bce, loss_l1,
              optim_D, optim_G, sf, batch_size, epochs, gpu):
    list_loss_G, list_loss_D = [], []
    for epoch in range(1, epochs + 1):
        start = time.time()
        sum_loss_G, sum_loss_D, num_of_samples = 0, 0, 0
        # data_loader_src is either landmarks image or human face image
        # data_loader_tgt is data_loader_src's corresponding cartoon image
        for (real_A, _), (real_B, _) in zip(data_loader_src, data_loader_tgt):

            # Train Discriminator
            real_A = real_A.cuda() if gpu else real_A
            real_B = real_B.cuda() if gpu else real_B
            # here modify the input to the generator based on the approach
            real_A_mod = real_A
            fake_B = net_G(real_A_mod)
            # Don't want to update weights of generator while training discriminator. Hence, detach()
            loss_D = train_discriminator(real_A_mod, real_B, fake_B.detach(), net_D, loss_bce, optim_D, gpu)
            sum_loss_D += loss_D

            # Train Generator
            loss_G = train_generator(real_A_mod, real_B, fake_B, net_D, loss_bce, loss_l1, optim_G, sf, gpu)
            sum_loss_G += loss_G

            num_of_samples += batch_size
            # print(f'Epoch: [{epoch}/{num_epochs}], Batch Number: [{i + 1}/{len(dataloader)}]')
            # print(f'Discriminator Loss: {loss_D}, Generator Loss: {loss_G}')

        loss_dis, loss_gen = sum_loss_D / num_of_samples, sum_loss_G / num_of_samples
        print(f'Epoch: [{epoch}/{epochs}], Run Time: {round(time.time() - start, 4)} Sec')
        print(f'Discriminator Loss: {round(loss_dis, 4)}, Generator Loss: {round(loss_gen, 4)}')
        list_loss_D.append(loss_dis)
        list_loss_G.append(loss_gen)

    return list_loss_G, list_loss_D
