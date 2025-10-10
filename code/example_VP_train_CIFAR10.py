import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from utils.VP_L2loss import VP_L2
from utils.VP_CEloss import VP_CE
from utils.architecture import CIFAR_Net

#PARAMETERS OF THE EXPERIMENT
distribution_loss = 'VP_CE'# VP_CE or VP_L2 (type of VP loss)
bag_size = 64#number of images per bag
warmup = 200#number of epochs with KLdiv pretraining
nepoch = 1001
run = 0 #random seed

#DATA
#define Data Augmentation (DA) for training set, no DA for test set
transform_TRAIN = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=8),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_TEST = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_TRAIN)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bag_size,shuffle=False, num_workers=2)
#load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_TEST)
testloader = torch.utils.data.DataLoader(testset, batch_size=bag_size,shuffle=False, num_workers=2)
#define classes
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = CIFAR_Net()#simple CNN with leaky relu activations
model.to('cuda')

#LOSSES AND OPTIMIZATION PARAMETERS
criterion_KL = nn.KLDivLoss()
lr_KL = 0.0001
reset_optimizer = True#reset optimizer after warmup
if 'VP_CE' in distribution_loss:
    criterion_VP = VP_CE(len(classes)).to('cuda')#initialize loss with number of classes
    lr_VP = 0.0001
else:
    if 'VP_L2' in distribution_loss:
        criterion_VP = VP_L2(len(classes)).to('cuda')#initialize loss with number of classes   
        lr_VP = 0.00003
    else:
        print('Program exited due to not supported distribution loss: '+distribution_loss)
        exit()
optimizer = optim.Adam(model.parameters(), lr = lr_KL, betas = (0.5, 0.999))

losses = list()
acc_train = list()
acc_test = list()
n_batch = int(np.ceil(50000/bag_size))#number_of_training_bags
n_batch_test = int(np.ceil(10000/bag_size))#number_of_test_bags
print(str(n_batch)+' bags in training set | '+str(n_batch_test)+' bags in test set')
out_test = np.zeros((n_batch_test, bag_size, len(classes)))
labels_test = np.zeros((n_batch_test,bag_size))
torch.manual_seed(run)
for epoch in range(nepoch):  # loop over the dataset multiple times
    running_loss = 0.0
    running_acc = 0.0
    if epoch%25==0:
        acc = 0
        for j, testdata in enumerate(testloader, 0):
            inputs, labels = testdata
            with torch.no_grad():
                outputs = model(inputs.to('cuda'))
                acc += (outputs.argmax(dim=1) == labels.to('cuda')).sum().float() / float( labels.size(0))
                aux_y = torch.zeros(len(labels),len(classes))
                labels_mean = torch.mean(aux_y.scatter_(1,labels.repeat(len(classes),1).transpose(1,0),1),dim=0)
                out_test[j,0:int(labels.size(0)),:] = outputs.data.cpu().numpy()
                labels_test[j,0:int(labels.size(0))] = labels.data.numpy()
        print('Test accuracy:'+str(acc.item()*100/n_batch_test))
        acc_test.append(acc.item()*100/n_batch_test)     
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        aux_y = torch.zeros(len(labels),10)
        labels_mean = torch.mean(aux_y.scatter_(1,labels.repeat(len(classes),1).transpose(1,0),1),dim=0)
        outputs = model(inputs.to('cuda'))
        if epoch < warmup: 
            loss = criterion_KL(torch.mean(outputs,dim=0).log(),labels_mean.to('cuda'))
        else:
            if reset_optimizer:
                optimizer = optim.Adam(model.parameters(), lr = lr_VP, betas = (0.5, 0.999))
                reset_optimizer = False
            loss = criterion_VP(outputs,labels_mean.to('cuda'))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == labels.to('cuda')).sum().float() / float( labels.size(0))
    print(str(epoch)+'|'+str(running_loss/n_batch)+'|'+str(running_acc.item()*100/n_batch))
    losses.append(running_loss/n_batch)
    acc_train.append(running_acc.item()*100/n_batch)

