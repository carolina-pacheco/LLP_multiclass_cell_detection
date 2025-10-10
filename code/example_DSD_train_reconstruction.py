import torch
import torch.nn as nn
import numpy as np
import random
from utils.complex import complex_mag, complex_mul
from utils.architecture import flip, weights_init, DSD_rec_only
from utils.create_synthetic_WBC_subtype_dataset import Synthetic_WBC_subtype_Dataset

def main():
    #PARAMETERS OF THE EXPERIMENT
    dataset_fold = 'train'
    data_directory = '/data/'#modify accordingly   
    DA = True#data augmentation
    lam = 5#parameter of soft NMS    
    kernel_size = 8#architecture kernel size
    soft = 'True'#soft NMS
    hard = 'True'#hard NMS
    m = 25#NMS patch size
    eps = 0.00001#learning rate
    niter = 2000
    L = 512
    #DATA
    training_set = Synthetic_WBC_subtype_Dataset(data_folder = data_directory+'data_'+dataset_fold+'/')
    print('Number of samples in training set:'+str(len(training_set)))
    #MODEL
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSD_rec_only([lam,m,kernel_size]).double()
    model.to(device)
    model.apply(weights_init)
    optimizer = torch.optim.SGD(model.parameters(),lr=eps)
    loss_fn = nn.MSELoss()
    losses_train = list()
    for t in range(niter):
        #loop through donors
        idx_donor = np.mod(t,len(training_set))      
        H , X , y = training_set[idx_donor]
        idx_batch = random.sample(list(range(H.shape[0])),5)
        H = H[idx_batch,:,:,:]
        X = X[idx_batch,:,:,:]
        if DA:#data augmentation
            hf = random.random()
            vf = random.random()
            if hf < 0.5:
                H = flip(H,2)
                X = flip(X,2)
            if vf < 0.5:
                H = flip(H,1)
                X = flip(X,1)
        out, idxs, bg = model(X.to(device),soft,hard)
        #project to hologram domain
        projection = bg.reshape((bg.shape[0],1,1,1,bg.shape[1])).expand(out.shape)+torch.ifft(complex_mul(torch.fft(out,2),training_set.T_tensor.unsqueeze(dim=0).unsqueeze(dim=0).expand(out.shape).double().to(device)),2)
        #compute and normalize hologram intensity
        A = complex_mag(projection)**2
        A = A - torch.min(A)
        A = A/torch.max(A)
        loss = loss_fn(torch.squeeze(A),torch.squeeze(H[:,:,:,0]).to(device))
        losses_train.append(loss.item())
        print(losses_train[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if np.mod(t,len(training_set)*5)==0:#save model checkpoint every 5 epochs
            torch.save(model.state_dict(),'/models/DSD_rec_niter_'+str(t)+'.pt')
            print('saved model at iteration #'+str(t))

if __name__=='__main__':
    main() 



