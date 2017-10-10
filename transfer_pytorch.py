import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np 
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt 
import time
import copy
import os
import pandas as pd

data_transforms = {
	'train': transforms.Compose([
			transforms.RandomSizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	'validation': transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
}

data_dir = './data'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x) ,
			data_transforms[x]) for x in ['train', 'validation']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=128,
												shuffle=True, num_workers=4)
				for x in ['train','validation']}
dset_sizes = {x: len(dsets[x]) for x in ['train','validation']}
dset_classes = dsets['train'].classes

def imshow(inp, title=None):
	inp = inp.numpy().transpose((1,2,0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std*inp + mean
	#plt.imshow(inp)
	# if title is not None:
	# 	plt.title(title)
	#plt.pause(1)

# inputs, classes = next(iter(dset_loaders['train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[dset_classes[x] for x in classes])

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
	since = time.time()

	best_model = model
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-'*10)

		for phase in ['train', 'validation']:
			if phase == 'train':
				optimizer = lr_scheduler(optimizer, epoch)
				model.train(True)
			else:
				model.train(False)

			running_loss = 0.0
			running_corrects = 0

			for data in dset_loaders[phase]:
				inputs, labels = data

				if use_gpu:
					inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				optimizer.zero_grad()

				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)

				if phase == 'train':
					loss.backward()
					optimizer.step()

				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dset_sizes[phase]
			epoch_acc = running_corrects / dset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
					phase, epoch_loss, epoch_acc))

			if phase == 'validation' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model =  copy.deepcopy(model)

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
				time_elapsed//60, time_elapsed%60))
	print('Best validation Acc: {:4f}'.format(best_acc))

	return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
	lr = init_lr*(0.1**(epoch // lr_decay_epoch))

	if epoch%lr_decay_epoch == 0:
		print('LR is set to {}'.format(lr))

	for param_group in optimizer.param_groups:
		param_group['lr']=lr

	return optimizer

use_gpu = True

densenet = models.densenet121(pretrained=True)
#num_features = densenet.fc.in_features
num_features = 1024
print(num_features)
densenet.fc = nn.Linear(num_features, 12)
print(densenet.fc)

if use_gpu:
	densenet = densenet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
densenet = train_model(densenet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

torch.save(densenet.state_dict(), './tes.pth.tar')