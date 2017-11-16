
# coding: utf-8

# In[1]:

import os
import time
import torch
import torch as tc
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from PIL import Image
import numpy as np
import sys,os
import math
import gc

# In[2]:

import models
from models import weights_init


# # Cuda Code

# In[3]:

import torch.backends.cudnn as cudnn
import torch
import sys
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
# print('  Try to change to Device 2 - with "torch.cuda.device(2)"')
# torch.cuda.device(2)
# print('  ! Active CUDA Device is still:', torch.cuda.current_device())
#
# print('  Try again with environment vars')
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# print('  ! Active CUDA Device is still:', torch.cuda.current_device())


# # Main code

# In[4]:

#Libera as funcionalidades da biblioteca cudnn
#cudnn.benchmark = True
gc.collect()

use_gpu = torch.cuda.is_available()
#use_gpu = False
if use_gpu:
	print("You are using CUDA. If it is not what you want, manually set this as False!")


# In[5]:

nc = 3
ngpu = 1
nz = 100
ngf = 64
ndf = 64
n_extra_d = 0
n_extra_g = 1 # Aqui a jogada é que o gerador deve ser mais poderoso q o detetive
imageSize = 64


# In[6]:

#get_ipython().system('cd images && pwd')


# ## Setando as transformações

# In[7]:

#get_ipython().system('ls images/images2/')


# In[8]:

#get_ipython().system('ls dataset')


# In[9]:

dataset = dset.ImageFolder(
	root='/home/gabriel/Redes Neurais/Projeto_Final_GANS/Tutorial_2/dataset/min_anime-faces',
	transform=transforms.Compose([
			transforms.Scale((imageSize, imageSize)),
			# transforms.CenterCrop(opt.imageSize),
			transforms.ToTensor(),
			#transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
		])
)


# ## Setando o Dataloader

# In[10]:

batch_size=64


# In[11]:


dataloader = tc.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
#126 classes


# ## Setando o modelo

# ### Modelo DCGAN

# In[12]:

class _netD_1(nn.Module):
	def __init__(self, ngpu, nz, nc, ndf,  n_extra_layers_d):
		super(_netD_1, self).__init__()
		self.ngpu = ngpu
		main = nn.Sequential(
			# input is (nc) x 96 x 96
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 5,3,1 for 96x96
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
		)

		# Extra layers
		for t in range(n_extra_layers_d):
			main.add_module('extra-layers-{0}.{1}.conv'.format(t, ndf * 8),
							nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False))
			main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ndf * 8),
							nn.BatchNorm2d(ndf * 8))
			main.add_module('extra-layers-{0}.{1}.relu'.format(t, ndf * 8),
							nn.LeakyReLU(0.2, inplace=True))


		main.add_module('final_layers.conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
		main.add_module('final_layers.sigmoid', nn.Sigmoid())
		# state size. 1 x 1 x 1
		self.main = main

	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		output = nn.parallel.data_parallel(self.main, input, gpu_ids)
		return output.view(-1, 1)


# In[13]:

# DCGAN model, fully convolutional architecture
class _netG_1(nn.Module):
	def __init__(self, ngpu, nz, nc , ngf, n_extra_layers_g):
		super(_netG_1, self).__init__()
		self.ngpu = ngpu
		#self.nz = nz
		#self.nc = nc
		#self.ngf = ngf
		main = nn.Sequential(
			# input is Z, going into a convolution
			# state size. nz x 1 x 1
			nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ngf) x 32 x 32
		)

		# Extra layers
		for t in range(n_extra_layers_g):
			main.add_module('extra-layers-{0}.{1}.conv'.format(t, ngf),
							nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
			main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ngf),
							nn.BatchNorm2d(ngf))
			main.add_module('extra-layers-{0}.{1}.relu'.format(t, ngf),
							nn.LeakyReLU(0.2, inplace=True))

		main.add_module('final_layer.deconv', 
						 nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)) # 5,3,1 for 96x96
		main.add_module('final_layer.tanh', 
						 nn.Tanh())
			# state size. (nc) x 96 x 96

		self.main = main


	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		return nn.parallel.data_parallel(self.main, input, gpu_ids), 0


# In[14]:

#netG = _netG_1(ngpu, nz, nc, ngf, n_extra_g)
netG_parallel = torch.nn.DataParallel(_netG_1(ngpu, nz, nc, ngf, n_extra_g))


# In[15]:

#print(netG)
print(netG_parallel)


# In[16]:

#netD = _netD_1(ngpu, nz, nc, ndf, n_extra_d)

netD_parallel = torch.nn.DataParallel(_netD_1(ngpu, nz, nc, ndf, n_extra_d))


# In[17]:

print(netD_parallel)
#print(netD)


# # Carregando pesos pré-treinados

# In[18]:

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


# In[19]:

#Parece ser um inicializador de pesos hardcoded
#netG.apply(weights_init)
netG_parallel.apply(weights_init)
#netD.apply(weights_init)
netD_parallel.apply(weights_init)
print(())


# load=False
# if load:
#     netD.load_state_dict(tc.load('path_d'))
#     
#     netG.load_state_dict(torch.load('path_G'))
# 

# ### Parametros de treinamento

# In[20]:

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()


# In[21]:


input = torch.FloatTensor(batch_size, 3, imageSize, imageSize)
print(input.size())
noise = torch.FloatTensor(batch_size, nz, 1, 1)
print(noise.size())


# In[22]:

#parser.add_argument('--binary', action='store_true', help='z from bernoulli distribution, with prob=0.5')
binary=False
#Ele testa pergunta se vc quer que o seu Z venha da distribuição bernoulli
if binary:
	bernoulli_prob = torch.FloatTensor(batch_size, nz, 1, 1).fill_(0.5)
	fixed_noise = torch.bernoulli(bernoulli_prob)
else:
	fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)


# In[23]:

label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0


# ### Broadcast para CUDA, se quiser

# In[24]:

if use_gpu:
	#netD.cuda()
	#netG.cuda()
	netD_parallel.cuda()
	netG_parallel.cuda()
	criterion = criterion.cuda()
	criterion_MSE = criterion_MSE.cuda()
	input,label = input.cuda(), label.cuda()
	noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


# ### Transformando tudo em variable

# In[25]:


input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)


# In[26]:

print(fixed_noise.size())


# ### Setando o optimizer

# In[27]:

beta1, beta2 = 0.9,0.999
lr = 2.0e-4
#optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, beta2))
optimizerD = optim.Adam(netD_parallel.parameters(), lr = lr, betas = (beta1, beta2))

#optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, beta2))
optimizerG = optim.Adam(netG_parallel.parameters(), lr = lr, betas = (beta1, beta2))


# ### Criando o diretório vazio

# In[28]:

outputDir = 'outputdir_train_2'

try:
	os.makedirs(outputDir)
except OSError as err:
	print("OS error: {0}".format(err))


# ## Treinando !

# In[29]:

print(type(input))
print(input.size())
print(label.size())


# In[30]:

def train_both_networks(num_epochs, dataloader, netD, netG, d_labelSmooth, outputDir,
						model_option =1,binary = False, epoch_interval = 1):
	use_gpu = tc.cuda.is_available()
	
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
			input.data.resize_(real_cpu.size()).copy_(real_cpu)
			label.data.resize_(batch_size).fill_(real_label - d_labelSmooth) # use smooth label for discriminator

			output = netD(input)
			errD_real = criterion(output, label)
			errD_real.backward()
			
			#######################################################
		   
			#######################################################
			# 1B - Train the detective network in the False Dataset
			#######################################################
			
			D_x = output.data.mean()
			# train with fake
			noise.data.resize_(batch_size, nz, 1, 1)
			if binary:
				bernoulli_prob.resize_(noise.data.size())
				noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
			else:
				noise.data.normal_(0, 1)
			fake,z_prediction = netG(noise)
			label.data.fill_(fake_label)
			output = netD(fake.detach()) # add ".detach()" to avoid backprop through G
			errD_fake = criterion(output, label)
			errD_fake.backward() # gradients for fake/real will be accumulated
			D_G_z1 = output.data.mean()
			errD = errD_real + errD_fake
			optimizerD.step() # .step() can be called once the gradients are computed

			#######################################################

			#######################################################
			# (2) Update G network: maximize log(D(G(z)))
			#  Train the faker with de output from the Detective (but don't train the Detective)
			#############3#########################################
			netG.zero_grad()
			label.data.fill_(real_label) # fake labels are real for generator cost
			output = netD(fake)
			errG = criterion(output, label)
			errG.backward(retain_variables=True) # True if backward through the graph for the second time
			if model_option == 2: # with z predictor
				errG_z = criterion_MSE(z_prediction, noise)
				errG_z.backward()
			D_G_z2 = output.data.mean()
			optimizerG.step()

			end_iter = time.time()
			
			#Print the info
			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
				  % (epoch, num_epochs, i, len(dataloader),
					 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter))
			
			#Save a grid with the pictures from the dataset, up until 64
			if i % 100 == 0:
				# the first 64 samples from the mini-batch are saved.
				vutils.save_image(real_cpu[0:64,:,:,:],
						'%s/real_samples.png' % outputDir, nrow=8)
				fake,_ = netG(fixed_noise)
				vutils.save_image(fake.data[0:64,:,:,:],
						'%s/fake_samples_epoch_%03d.png' % (outputDir, epoch), nrow=8)
		if epoch % epoch_interval == 0:
			# do checkpointing
			torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outputDir, epoch))
			torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outputDir, epoch))


# In[31]:

def save_images(netG, noise, outputDir,epoch):
   # the first 64 samples from the mini-batch are saved.
   fake,_ = netG(fixed_noise)
   vutils.save_image(fake.data[0:64,:,:,:],
		   '%s/fake_samples_epoch_%03d.png' % (outputDir, epoch), nrow=8)

def save_models(netG, netD, outputDir, epoch):
   torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outputDir, epoch))
   torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outputDir, epoch))


# In[1]:

from IPython.core.debugger import set_trace
import pdb


# In[ ]:

#for i in range(10):
 #   if i == 3:
  #      pdb.set_trace()
   #     print('parou aqui')
	#    print('parou aqui 2')


# In[32]:

def train_our(num_epochs, dataloader, netD, netG, d_labelSmooth, outputDir,
						model_option =1,binary = False, epoch_interval = 1,
						D_steps = 1, G_steps = 1):
	use_gpu = tc.cuda.is_available()

	for epoch in range(num_epochs):
		start_iter = time.time()  
		D_x = 0
		D_G_z1 = 0
		D_G_z2 = 0
		errD_acum = 0
		errG_acum = 0
		
		for z in range(D_steps):
			if z > 3:
				raise ValueError('KEEP IT LOW!')
			print('z', z)
			for j, data in enumerate(dataloader, 0):


				############################
				# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
				# 1A - Train the detective network in the Real Dataset
				###########################
				# train with real
				netD.zero_grad()
				real_cpu, _ = data
				if (epoch == 0 and z == 0 ):
					vutils.save_image(real_cpu[0:64,:,:,:],
					'%s/real_samples.png' % outputDir, nrow=8)
				
				batch_size = real_cpu.size(0)
				input.data.resize_(real_cpu.size()).copy_(real_cpu)
				label.data.resize_(batch_size).fill_(real_label - d_labelSmooth) # use smooth label for discriminator

				output = netD(input)
				errD_real = criterion(output, label)
				errD_real.backward()

				#######################################################

				#######################################################
				# 1B - Train the detective network in the False Dataset
				#######################################################

				D_x += output.data.mean()
				print()
				# train with fake
				noise.data.resize_(batch_size, nz, 1, 1)
				if binary:
					bernoulli_prob.resize_(noise.data.size())
					noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
				else:
					noise.data.normal_(0, 1)
				fake,z_prediction = netG(noise)
				label.data.fill_(fake_label)
				output = netD(fake.detach()) # add ".detach()" to avoid backprop through G
				errD_fake = criterion(output, label)
				errD_fake.backward() # gradients for fake/real will be accumulated
				# ERROR MEAN
				D_G_z1 += output.data.mean()
				
				errD_acum += errD_real.data[0] + errD_fake.data[0]
				
				optimizerD.step() # .step() can be called once the gradients are computed

				#######################################################
		
		# PARADA PARA VER O Q ESTÁ ACONTENDO
		
		for a in range(G_steps):
			print('interacao = ',a, 'de ',G_steps )
			for i, data in enumerate(dataloader, 0):


				# G_steps > D_steps (G_steps \geq D_steps)
				if a > 3:
					raise ValueError('KEEP IT LOW!')
				#######################################################
				# (2) Update G network: maximize log(D(G(z)))
				#  Train the faker with de output from the Detective (but don't train the Detective)
				#############3#########################################
			   #	print('ITERACAO QUE VAI DA MERDA = ',i)


				#if i==150:
			   #		pdb.set_trace()
				netG.zero_grad()
				label.data.fill_(real_label) # fake labels are real for generator cost
				output = netD(fake)
				errG = criterion(output, label)
				errG.backward(retain_variables=True) # True if backward through the graph for the second time
				#errG.backward() # True if backward through the graph for the second time
				
				#print("DEU ESSA SAIDA")
				if model_option == 2: # with z predictor
					errG_z = criterion_MSE(z_prediction, noise)
					errG_z.backward()
				D_G_z2 += output.data.mean()
				errG_acum += errG.data[0]
				#pdb.set_trace()
				#D_G_z2 = output.data.mean()
				#errG_acum = errG
				
				optimizerG.step()


		print('epoch = ',epoch)
		
		end_iter = time.time()        
		#Print the info
		print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
			  % (epoch, num_epochs, errD_acum/D_steps, errG_acum/G_steps, D_x, D_G_z1, D_G_z2, end_iter-start_iter))

		print('chegou no print')
		#Save a grid with the pictures from the dataset, up until 64
		save_images(netG = netG, noise = fixed_noise, outputDir = outputDir, epoch = epoch)
		   
		if epoch % epoch_interval == 0:
			# do checkpointing
			save_models(netG = netG, netD = netD, outputDir = outputDir, epoch = epoch)


# In[40]:

num_epochs = 100
d_labelSmooth = 0.2

train_our(num_epochs, dataloader, netD_parallel,netG_parallel,d_labelSmooth, outputDir)




# In[ ]:




# num_epochs = 100
# d_labelSmooth = 0.2
# 
# train_both_networks(num_epochs, dataloader, netD_parallel,netG_parallel,d_labelSmooth, outputDir)
