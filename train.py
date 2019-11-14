import torch
import random
from torch import nn, optim
import datetime

from models.cartoon_gan import Generator
from models.dc_gan import Discriminator
from utils.utils import *


def main():
    print('############################### train.py ###############################')

    # Set random seed for reproducibility
    manual_seed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manual_seed)
    print()
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Hyper parameters
    workers = 2
    batch_size = 128
    image_size = 128
    nc = 3
    in_ngc = 3
    out_ngc = 3
    in_ndc = 3
    out_ndc = 1
    ngf = 64
    ndf = 32
    learning_rate = 0.0005
    beta1 = 0.5
    epochs = 100
    gpu = True
    load_saved_model = False

    # print hyper parameters
    print(f'number of workers : {workers}')
    print(f'batch size : {batch_size}')
    print(f'image size : {image_size}')
    print(f'number of channels : {nc}')
    print(f'generator feature map size : {ngf}')
    print(f'discriminator feature map size : {ndf}')
    print(f'learning rate : {learning_rate}')
    print(f'beta1 : {beta1}')
    print(f'epochs: {epochs}')
    print(f'GPU: {gpu}')
    print(f'load saved model: {load_saved_model}')
    print()

    # set up GPU device
    cuda = True if gpu and torch.cuda.is_available() else False

    # load CelebA dataset
    download_path = '/home/pbuddare/EEE_598/data/CelebA'
    # download_path = '/Users/prasanth/Academics/ASU/FALL_2019/EEE_598_CIU/data/Project/CelebA'
    data_loader_src = prepare_celeba_data(download_path, batch_size, image_size, workers)

    # load respective cartoon dataset
    download_path = '/home/pbuddare/EEE_598/data/Cartoon'
    # download_path = '/Users/prasanth/Academics/ASU/FALL_2019/EEE_598_CIU/data/Project/Cartoon'
    data_loader_tgt = prepare_cartoon_data(download_path, batch_size, image_size, workers)

    # show sample images
    show_images(next(iter(data_loader_src))[0], (8, 8), 16, 'Training images (Natural)', 'natural_train')
    show_images(next(iter(data_loader_tgt))[0], (8, 8), 16, 'Training images (Cartoon)', 'cartoon_train')

    # create generator and discriminator networks
    generator = Generator(in_ngc, out_ngc, ngf)
    discriminator = Discriminator(in_ndc, out_ndc, ndf)
    if cuda:
        generator.cuda()
        discriminator.cuda()

    # loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Train GAN
    loss_G, loss_D = train_gan(data_loader_src, data_loader_tgt, generator, discriminator, criterion,
                               optimizer_d, optimizer_g, batch_size, epochs, cuda)

    # save parameters
    current_time = str(datetime.datetime.now().time()).replace(':', '').replace('.', '') + '.pth'
    g_path = './project_G_' + current_time
    d_path = './project_D_' + current_time
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)

    # generate and display fake images
    test_imgs = next(iter(data_loader_src))[0]
    show_images(test_imgs, (8, 8), 16, 'Testing images (Natural)', 'natural_test')
    fake_imgs = generator(test_imgs).detach()
    show_images(fake_imgs.cpu(), (8, 8), 16, 'Fake images (Cartoon)', 'cartoon_fake')


if __name__ == '__main__':
    main()
