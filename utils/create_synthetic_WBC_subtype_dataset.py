import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.physics import WAS_xfer
from utils.complex import complex_mul, complex_cj
import pickle
import sys
import warnings
class Synthetic_WBC_subtype_Dataset(Dataset):
    """ WBC synthetic data dataset """
    def __init__(self, data_folder):
        """
        Args:
            data_folder: directory containing dictionaries of data named "Donor_*.dict"
            Each dictionary contains the following keys
            - 'H': Hologram (N_crops,L,L) numpy array, unnormalized holograms
            - 'y_prop': (1,3). Proportion of lymphocytes, monocytes and granulocytes
            Assumption: Working distance is z = 930e-6, wave_len = 637e-9, pix_size = 1.12e-6
        """
        self.directory = data_folder
        self.files = [f for f in listdir(self.directory) if (f.startswith('Donor_') and f.endswith('.dict'))]
        self.files.sort()
        self.Nholo = len(self.files)
        self.N_crops = 7 #number of holograms per donor
        self.L = 512 #lateral size of (square) patches
        self.z = 930e-6
        self.wave_len = 637e-9
        self.pix_size = 1.12e-6
        self.T, _ = WAS_xfer(self.z,self.wave_len,(self.L,self.L),self.pix_size)
        self.T_tensor = torch.stack([torch.from_numpy(np.real(self.T)),torch.from_numpy(np.imag(self.T))],dim=2)
    def __len__(self):
        return self.Nholo

    def __getitem__(self, idx):
        """ 
        Args:
            idxs: sequence of indexes for samples 
            - each element of idx is in the range [0, self.Nholo[
        """
        H_tensor = torch.zeros((self.N_crops,self.L,self.L,2),dtype=torch.double)
        X_tensor = torch.zeros((self.N_crops,self.L,self.L,2),dtype=torch.double)       
        sample = pickle.load(open(join(self.directory,self.files[idx]),'rb'))
        H = sample['H']
        H = H - np.min(H)
        H = np.sqrt(H/np.max(H))
        H_tensor[:,:,:,0] = torch.from_numpy(H)
        y = sample['y_prop']
        if 'torch.fft' not in sys.modules:
            with warnings.catch_warnings(record=True) as w:
                X_tensor = torch.ifft(complex_mul(torch.fft(H_tensor,2),complex_cj(self.T_tensor).unsqueeze(dim=0).expand(X_tensor.shape)),2)
        else:
            auxiliar = complex_cj(self.T_tensor).unsqueeze(dim=0).expand(X_tensor.shape)
            H_ = torch.complex(H_tensor[...,0],H_tensor[...,1])
            auxiliar2 = torch.fft.ifft2(torch.fft.fft2(H_)*torch.complex(auxiliar[...,0],auxiliar[...,1]))
            X_tensor[...,0] = torch.real(auxiliar2)
            X_tensor[...,1] = torch.imag(auxiliar2)
        X_tensor = X_tensor - torch.mean(torch.mean(X_tensor,1,keepdim=True),2,keepdim=True)
        return H_tensor, X_tensor, y

