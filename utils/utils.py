import torch
from torch.autograd import Variable
from torchvision import transforms, datasets, utils as vutils
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import dlib
from imutils import face_utils
import imutils
import dlib
import cv2
from utils.transforms.getLandmarks import GetLandmarksImage

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

def prepare_cartoon_data_with_landmarks(path, batch_size, image_size, workers):
    transform = transforms.Compose([GetLandmarksImage(),
                                    transforms.Resize(image_size),
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


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# generate real labels
def real_label(batch_size, gpu):
    data = Variable(torch.ones(batch_size, 1, 1, 1))
    return data.cuda() if gpu else data


# generate fake labels
def fake_label(batch_size, gpu):
    data = Variable(torch.zeros(batch_size, 1, 1, 1))
    return data.cuda() if gpu else data


# Train discriminator
def train_discriminator(X, X_dash, net_D, loss, optim_D, gpu):
    optim_D.zero_grad()

    real_Y = net_D(X)
    fake_Y = net_D(X_dash)

    loss_D = loss(real_Y, real_label(X.size(0), gpu)) + loss(fake_Y, fake_label(X.size(0), gpu))
    loss_D.backward()

    optim_D.step()
    return torch.sum(loss_D).item()


# Train generator
def train_generator(X, X_dash, net_D, loss, optim_G, gpu):
    optim_G.zero_grad()

    fake_Y = net_D(X_dash)

    loss_G = loss(fake_Y, real_label(X_dash.size(0), gpu))
    loss_G.backward()

    optim_G.step()
    return torch.sum(loss_G).item()

    # needs to be implemented if adding new custom loss
    # loss_feature = custom_loss(X, X_dash)
    # loss_total = loss_G + gamma * loss_feature
    # loss_total.backward()

    # optim_G.step()
    # return torch.sum(loss_total).item()


# Train GAN
def train_gan(data_loader_src, data_loader_tgt, net_G, net_D, loss, optim_D, optim_G, batch_size, epochs, gpu):
    list_loss_G, list_loss_D = [], []
    for epoch in range(1, epochs + 1):
        start = time.time()
        sum_loss_G, sum_loss_D, num_of_samples = 0, 0, 0
        # data_loader_src is either landmarks image or human face image
        # data_loader_tgt is data_loader_src's corresponding cartoon image
        for (Z, _), (X, _) in zip(data_loader_src, data_loader_tgt):
            # Train Discriminator
            Z = Z.cuda() if gpu else Z
            X = X.cuda() if gpu else X
            # here modify the input to the generator based on the approach
            Z_dash = Z
            X_dash = net_G(Z_dash)
            # Don't want to update weights of generator while training discriminator. Hence, detach()
            loss_D = train_discriminator(X, X_dash.detach(), net_D, loss, optim_D, gpu)
            sum_loss_D += loss_D

            # Train Generator
            loss_G = train_generator(X, X_dash, net_D, loss, optim_G, gpu)

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

def landmark(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    print("{} rects predicted".format(len(rects)))

    blackImage = np.zeros((image.shape))
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(blackImage, (x, y), 1, (255, 255, 255), thickness=2)

    return blackImage

# if __name__ == "__main__":
#     resultImage = landmark(cv2.imread("/Users/yvtheja/Documents/ASU/EEE598/cartoonize/data/17349955_1.jpg", cv2.IMREAD_COLOR))
#     cv2.imwrite("boom.jpg", resultImage)