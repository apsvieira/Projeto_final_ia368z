
# coding: utf-8

# In[1]:

# get_ipython().magic('matplotlib inline')
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch as tc
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


# In[2]:

gpu_available = torch.cuda.is_available()


# ### Definition of some important parameters
# Dataset to be used is defined here. According to the dataset, some other hyperparameters are set.

# In[3]:

# This should, in the future, be set in CLI
chosen_dataset = 'CIFAR10'

datasets = {
    'MNIST': torchvision.datasets.MNIST,
    'CIFAR10': torchvision.datasets.CIFAR10
    #'FashionMNIST': torchvision.datasets.FashionMNIST
}

dataset = datasets[chosen_dataset]

possible_parameters = {
    'MNIST': {
        'ndf': 64,
        'ngf': 64,
        'code_size': 50,
        'n_channels': 1,
        'n_classes': 10,
    },
    'CIFAR10': {
        'ndf': 64,
        'ngf': 64,
        'code_size': 100,
        'n_channels': 3,
        'n_classes': 10,
    },
    #'FashionMNIST': {}
}

ngf = possible_parameters[chosen_dataset]['ngf']
ndf = possible_parameters[chosen_dataset]['ndf']
code_size = possible_parameters[chosen_dataset]['code_size']
n_channels = possible_parameters[chosen_dataset]['n_channels']
n_classes = possible_parameters[chosen_dataset]['n_classes']

# ### Define data handling structures
# - Batch size and number of child processes are to be set by user.
# - Defines data transforms to be used in train and test phases
# - Defines dataset and dataloader dictionaries

# In[4]:

# These should, in the future, be set in CLI
batch_size = 16
number_processes = 4

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(32),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
    ])
}


# In[5]:

datasets = {
    'train': dataset('../datasets', train=True, download=False, transform=data_transforms['train']),
    'test': dataset('../datasets', train=False, download=False, transform=data_transforms['test'])
}


# In[6]:

if gpu_available:
    n_train_samples = 1000
    n_test_samples = 100

else:
    n_train_samples = 500
    n_test_samples = 100

datasets['train'].train_data = datasets['train'].train_data[:n_train_samples]
datasets['train'].train_labels = datasets['train'].train_labels[:n_train_samples]

datasets['test'].test_data = datasets['test'].test_data[:n_test_samples]
datasets['test'].test_labels = datasets['test'].test_labels[:n_test_samples]


# In[7]:

dataloaders = {
    'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size,
                                         shuffle=True, num_workers=number_processes),
    'test': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size,
                                        shuffle=False, num_workers=number_processes),
}
print('Length of training dataloader:', len(dataloaders['train']))
print('Length of test dataloader:', len(dataloaders['test']))


# ### Network definitions
# Network hyperparameters are defined by the user or defined according to dataset choice.

# In[8]:

class generator_network(nn.Module):
    # This model is sequential and will be implemented as such for simplicity
    def __init__(self, code_size, ngf, n_channels):
        super().__init__()
        self.code_size = code_size
        self.ngf = ngf
        self.n_channels = n_channels
        self.fc = nn.Sequential(
            nn.Linear(self.code_size, self.ngf * 8 * 4 * 4),
            # nn.BatchNorm1d(self.ngf*8*4*4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(self.ngf * 2, self.n_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.n_channels),
            # If uncommented, 32x32 -> 64x64
            # nn.ReLU(),
            # nn.ConvTranspose2d(self.ngf, self.n_channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.n_channels),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.code_size)
        input = self.fc(input)
        input = input.view(-1, self.ngf * 8, 4, 4)
        output = self.deconv(input)
        return output


# In[9]:

class discriminator_network(nn.Module):
    def __init__(self, ndf, n_channels):
        super().__init__()
        self.ndf = ndf
        self.n_channels = n_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channels, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
        )
        self.n_conv_features = self._get_conv_output((3, 32, 32))
        self.fc = nn.Sequential(
            nn.Linear(self.n_conv_features, 192),
            nn.Linear(192, 100),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, self.n_conv_features)
        output = self.fc(x).squeeze()
        return output

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


# ### Auxiliary function definitions
# - weights_init initializes Conv layers and BatchNorm layers

# In[10]:

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.05)
        m.bias.data.fill_(0)


# ### Network creation step

# In[11]:


# print("Debuggando antes da rede D ser criada! ndf = ",
 #     ndf, "e n_channels = ", n_channels)
# ndf = 64 e n_channels = 3
D = discriminator_network(ndf, n_channels)
# print("Codesize = ", code_size, "ngf = ", ngf, " n_channels = ", n_channels)
G = generator_network(code_size, ngf, n_channels)

D.apply(weights_init)
G.apply(weights_init)

print(G, '\n', D)


# ### Defining optimization parameters
# - Optimizer: Adam
# - LR: user defined, default = 0.0003 (3e-4)
# - LR decay: linear decay over the epochs, to a minimum of 0.0001 (1e-4)
# - Loss function: BCELoss

# In[12]:

input = torch.FloatTensor(batch_size, code_size, 32, 32)
label = torch.FloatTensor(batch_size)
#label = torch.FloatTensor(batch_size, n_classes)

print('l269', label.size())


fixed_noise = torch.FloatTensor(batch_size, code_size, 1, 1).normal_(0, 1)
noise = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=3e-4)
optimizerG = torch.optim.Adam(G.parameters(), lr=3e-4)


def lin_decay(epoch): return np.clip(3 - epoch / 400, 1e-4, None)


D_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizerD, lin_decay)
G_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizerG, lin_decay)

if gpu_available:
    D.cuda()
    G.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    fixed_noise, noise = fixed_noise.cuda(), noise.cuda()


input, label = Variable(input), Variable(label)
fixed_noise, noise = Variable(fixed_noise), Variable(noise)


# ### Training function

# In[13]:

def save_images(netG, noise, outputDir, epoch):
    # the first 64 samples from the mini-batch are saved.
    fake, _ = netG(fixed_noise)
    vutils.save_image(fake.data[0:64, :, :, :],
                      '%s/fake_samples_epoch_%03d.png' % (outputDir, epoch), nrow=8)


def save_models(netG, netD, outputDir, epoch):
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outputDir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outputDir, epoch))


def train_both_networks(num_epochs, dataloader, netD, netG, d_labelSmooth, outputDir,
                        model_option=1, binary=False, epoch_interval=1):

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            start_iter = time.time()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # 1A - Train the detective network in the Real Dataset
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            # print('input antes do resize', input.size())
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            # print('input depois do resize', input.size())

            # use smooth label for discriminator
            print('real label antes de dar um fill no label', real_label)
            print('label antes do fill', label.size())
            label.data.resize_(batch_size).fill_(real_label - d_labelSmooth)
            print('input que entra no netD', input.size())
            output = netD(input)
            print('output que sai da Discriminant', output.size())
            print('label depois do fill', label.size())

            errD_real = criterion(output, label)

            errD_real.backward()

            #######################################################

            #######################################################
            # 1B - Train the detective network in the False Dataset
            #######################################################

            D_x = output.data.mean()
            # train with fake
            noise.data.resize_(batch_size, code_size, 1, 1)
            if binary:
                bernoulli_prob.resize_(noise.data.size())
                noise.data.copy_(2 * (torch.bernoulli(bernoulli_prob) - 0.5))
            else:
                noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.fill_(fake_label)
            # add ".detach()" to avoid backprop through G
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()  # gradients for fake/real will be accumulated
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()  # .step() can be called once the gradients are computed
            D_scheduler.step()

            #######################################################

            #######################################################
            # (2) Update G network: maximize log(D(G(z)))
            #  Train the faker with de output from the Detective (but don't train the Detective)
            #############3#########################################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.data.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            # True if backward through the graph for the second time()
            errG.backward(retain_graph=True)

            # if opt.model == 2:  # with z predictor
            #    errG_z = criterion_MSE(z_prediction, noise)
            #    errG_z.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()
            G_scheduler.step()

            end_iter = time.time()

            # Print the info
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter - start_iter))

            # Save a grid with the pictures from the dataset, up until 64
            if i % 100 == 0:
                # the first 64 samples from the mini-batch are saved.
                vutils.save_image(real_cpu[0:64, :, :, :],
                                  '%s/real_samples.png' % outputDir, nrow=8)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data[0:64, :, :, :],
                                  '%s/fake_samples_epoch_%03d.png' % (outputDir, epoch), nrow=8)
        if epoch % epoch_interval == 0:
            # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %
                       (outputDir, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %
                       (outputDir, epoch))


# In[14]:

train_both_networks(num_epochs=25, dataloader=dataloaders['train'], netD=D, netG=G, d_labelSmooth=0.1,
                    outputDir='./generated_images', model_option=1, binary=False, epoch_interval=1)


# In[ ]:


# In[ ]:
