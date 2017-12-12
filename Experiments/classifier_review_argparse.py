# Imports
import argparse
import numpy as np
import math


import os
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def set_cuda():
    import torch.backends.cudnn as cudnn
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    #from subprocess import call
    #call(["nvcc", "--version"])
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    #call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    cudnn.benchmark = True

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("You are using CUDA. If it is not what you want, manually set this as False!")
    return(use_gpu)


def define_dataset():

    chosen_dataset = 'MNIST'

    datasets = {
        'MNIST': torchvision.datasets.MNIST,
        'CIFAR10': torchvision.datasets.CIFAR10,
        'ANIME': '/home/gabriel/Redes Neurais/Projeto_Final_GANS/Tutorial_2/dataset/min_anime-faces',
    }

    dataset = datasets[chosen_dataset]

    possible_parameters = {
        'MNIST': {
            'ndf': 64,
            'ngf': 64,
            'nz': 100,
            'nc': 1,
            'imageSize': 64,
            'n_classes': 10,
            'ngpu': 1,
        },
        'CIFAR10': {
            'ndf': 64,
            'ngf': 64,
            'nz': 100,
            'nc': 3,
            'imageSize': 64,
            'n_classes': 10,
            'ngpu': 1,
        },
        'ANIME': {
            'nc': 3,
            'ngpu': 1,
            'nz': 100,
            'ngf': 64,
            'ndf': 64,
            'imageSize': 64,
            'n_classes': 1
        }
    }

    ngf = possible_parameters[chosen_dataset]['ngf']
    ndf = possible_parameters[chosen_dataset]['ndf']
    nz = possible_parameters[chosen_dataset]['nz']
    nc = possible_parameters[chosen_dataset]['nc']
    imageSize = possible_parameters[chosen_dataset]['imageSize']
    n_classes = possible_parameters[chosen_dataset]['n_classes']
    ngpu = possible_parameters[chosen_dataset]['ngpu']

    # In[9]:

    if dataset == 'ANIME':
        dataset = torchvision.datasets.ImageFolder(
            root='/home/gabriel/Redes Neurais/Projeto_Final_GANS/Tutorial_2/dataset/min_anime-faces',
            transform=transforms.Compose([
                transforms.Scale((imageSize, imageSize)),
                transforms.ToTensor(),
            ])
        )
    else:
        transform = transforms.Compose([
            transforms.Scale((imageSize, imageSize)),
            transforms.ToTensor(),
            # bring images to (-1,1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_done = dataset('/tmp/datasets', train=True,
                               download=True, transform=transform)
        return(dataset_done, ngf, ndf, nz, nc, imageSize, n_classes, ngpu)


def define_dataloader(dataset_done, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset_done, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Dataloader length:', len(dataloader))
    print("Dataset:", dataloader.dataset)
    return(dataloader)


class _netD_DCGAN(nn.Module):
    def __init__(self, nz, nc, ndf, n_classes):
        super(_netD_DCGAN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=ndf,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.batch4 = nn.BatchNorm2d(ndf * 8)

        self.final_conv = nn.Conv2d(
            in_channels=ndf * 8, out_channels=n_classes + 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.batch2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.batch3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.batch4(self.conv4(x)), 0.2, inplace=True)

        x = self.final_conv(x)
        return(x)


class _netG_DCGAN(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG_DCGAN, self).__init__()
        self.convt1 = nn.ConvTranspose2d(
            in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(ngf * 8)
        self.convt2 = nn.ConvTranspose2d(
            in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(ngf * 4)
        self.convt3 = nn.ConvTranspose2d(
            in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(ngf * 2)
        self.convt4 = nn.ConvTranspose2d(
            in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch4 = nn.BatchNorm2d(ngf)

        self.final_convt = nn.ConvTranspose2d(
            in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.batch1(self.convt1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.batch2(self.convt2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.batch3(self.convt3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.batch4(self.convt4(x)), 0.2, inplace=True)

        x = self.final_convt(x)
        x = F.tanh(x)
        return (x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def initialize_tensors(batch_size, imageSize, nz, n_classes):
    input = torch.FloatTensor(batch_size, 3, imageSize, imageSize)
    print('Input images size:', input.size())
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
    print('Code size:', noise.size())

    label = torch.LongTensor(batch_size, n_classes)
    print('Label size:', label.size())
    fake_label = 10
    real_label = 1
    return(input, noise, fixed_noise, label, fake_label, real_label)


def broadcast_to_gpu(use_gpu, netD, netG, label, input, criterion, fixed_noise, noise):
    if use_gpu:
        print('Broadcasting to the GPU...')
        netD.cuda()
        netG.cuda()
        criterion = criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    else:
        print("GPU isn't available ")


def train_properties(netD, netG, beta1, beta2, lr):
    criterion = nn.CrossEntropyLoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    return(optimizerG, optimizerD, criterion)


def save_images(g_net, fixed_noise, epoch, imageSize, outputDir):
    fixed_noise = Variable(fixed_noise)
    g_net.eval()
    fake_data = g_net(fixed_noise.cuda())
    g_net.train()
    fig = plot_samples(fake_data.data.cpu().numpy(), imageSize)
    plt.savefig(
        outputDir + '/dcgan_img_{:04d}.png'.format(epoch, bbox_inches='tight'))
    plt.close(fig)


def plot_samples(samples, imageSize):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples[:25]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(imageSize, imageSize), cmap='Greys_r')
    return fig


def train_gan(num_epochs, dataloader, netD, optimizerD, netG, optimizerG, criterion, nz, outputDir, noise, fixed_noise, imageSize,
              real_labelSmooth=0, epoch_interval=100, D_steps=1, G_steps=1, fake_label=0, n_classes=10, decay=0.05):

    # This validation is subjective. WGAN-GP uses 100 steps on the critic (netD).
    assert D_steps < 5, "Keep it low, D_steps is too high."
    assert G_steps < 3, "Keep it low, G_steps is too high."
    #assert batch_size % D_steps == 0, "Use batch_size multiple of D_steps."
    real_label = 1
    print('Lets train!')
    for epoch in range(num_epochs):
        start_iter = time.time()
        D_x = 0
        D_G_z1 = 0
        D_G_z2 = 0
        errD_acum = 0
        errG_acum = 0

        print('In epoch = ', epoch, 'real_label_smooth = ', real_labelSmooth)
        for batch, data in enumerate(dataloader, 0):
            if (epoch == 0 and batch == 0):
                fig = plot_samples(data[0][0:data[0].size(
                    0), :, :, :].cpu().numpy(), imageSize=imageSize)
                plt.savefig(
                    outputDir + '/real_samples.png'.format(epoch, bbox_inches='tight'))
                plt.close(fig)
            real_labelSmooth = np.maximum(
                real_labelSmooth * (1 - decay * epoch), 0)
            for step in range(D_steps):
                #############################################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # 1A - Train the detective network in the Real Dataset
                #############################################################
                netD.zero_grad()
                start = step * (int(data[0].size()[0] / D_steps))
                end = (step + 1) * int(data[0].size()[0] / D_steps)

                real_cpu = data[0][start:end]
                real_cpu = real_cpu.cuda()
                batch_size = real_cpu.size(0)
                if np.random.random_sample() > real_labelSmooth:
                    target = data[1][start:end].long().cuda()
                else:
                    target = torch.from_numpy(np.random.randint(
                        0, n_classes, batch_size)).type(torch.LongTensor).cuda()

                input, label = Variable(real_cpu), Variable(target)

                output = netD(input)
                errD_real = criterion(output.squeeze(), label)
                errD_real.backward()

                D_x += output.data.mean()

                #######################################################
                # 1B - Train the detective network in the False Dataset
                #######################################################

                noise = Variable(torch.FloatTensor(
                    batch_size, nz, 1, 1).normal_(0, 1).cuda())
                fake = netG(noise)
                label = Variable(torch.ones(
                    batch_size).long().fill_(fake_label).cuda())
                # ".detach()" to avoid backprop through G
                output = netD(fake.detach())
                errD_fake = criterion(output.squeeze(), label)
                errD_fake.backward()  # gradients for fake and real data will be accumulated

                D_G_z1 += output.data.mean()
                errD_acum += errD_real.data[0] + errD_fake.data[0]
                optimizerD.step()

            for step in range(G_steps):
                ####################################################################################
                # (2) Update G network: maximize log(D(G(z)))
                # Train the faker with the output from the Detective (but don't train the Detective)
                ####################################################################################

                netG.zero_grad()
                label = Variable(torch.from_numpy(np.random.randint(
                    0, n_classes, batch_size)).type(torch.LongTensor).cuda())
                output = netD(fake)
                errG = criterion(output.squeeze(), label)
                errG.backward()

                D_G_z2 += output.data.mean()
                errG_acum += errG.data[0]
                optimizerG.step()

        print('epoch = ', epoch)

        end_iter = time.time()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
              % (epoch, num_epochs, errD_acum / D_steps, errG_acum / G_steps, D_x, D_G_z1, D_G_z2, end_iter - start_iter))

        # Save a grid with the pictures from the dataset, up until 64
        save_images(netG, fixed_noise, epoch,
                    imageSize=imageSize, outputDir=outputDir)

        if epoch % epoch_interval == 0:
            # do checkpointing
            save_models(netG=netG, netD=netD, outputDir=outputDir, epoch=epoch)


def define_outputDir(outputDir):

    try:
        os.makedirs(outputDir)
        print('Directory created!')
    except OSError as err:
        print("OS error: {0}".format(err))
    return(outputDir)


def save_models(netG, netD, outputDir, epoch):
    '''
    Saves model state dictionary for generator and discriminator networks.
    Inputs are the networks (netG, netD), the system path in which to save(outputDir) and the current 'epoch'.
    '''
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outputDir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outputDir, epoch))


def main():
    #######Variables####
    batch_size = 128
    beta1, beta2 = 0.5, 0.999
    lr = 2.0e-4
    num_epochs = 40
    real_labelSmooth = 0.10
    decay = 0.010
    outputDir = 'outputdir_train_classifier_decay_0d010_label_0d10'

    ##########################################################
    #
    #
    use_gpu = set_cuda()
    dataset_done, ngf, ndf, nz, nc, imageSize, n_classes, ngpu = define_dataset()
    dataloader = define_dataloader(dataset_done, batch_size)
    netG = _netG_DCGAN(nz, nc, ngf)
    netD = _netD_DCGAN(nz, nc, ndf, n_classes)
    netG.apply(weights_init)
    netD.apply(weights_init)
    #print(netG, '\n', netD)
    input, noise, fixed_noise, label, fake_label, real_label = initialize_tensors(
        batch_size, imageSize, nz, n_classes)
    optimizerG, optimizerD, criterion = train_properties(
        netD, netG, beta1, beta2, lr)
    broadcast_to_gpu(use_gpu, netD, netG, label, input,
                     criterion, fixed_noise, noise)
    outputDir = define_outputDir(
        '/home/gabriel.barros/condor_test/' + outputDir)

    # Train
    print('got here')

    train_gan(num_epochs, dataloader, netD, optimizerD, netG, optimizerG,
              criterion, nz, outputDir, noise=noise, fixed_noise=fixed_noise, imageSize=imageSize, real_labelSmooth=real_labelSmooth,
              fake_label=fake_label, n_classes=n_classes, decay=decay)
    print('finished training')


if __name__ == "__main__":
    main()

# Passar por argparse o string do file, decay e label!
