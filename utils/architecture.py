import torch
import torch.nn.functional as F
import numpy as np
from utils.complex import complex_mag
import os
import torch.nn as nn

class DSD_rec_only(nn.Module):
    def __init__(self,list_of_params):
        super(DSD_rec_only,self).__init__()
        self.encoder = Encoder(list_of_params)
        self.decoder = Decoder([list_of_params[2]])
        self.conv_bg = nn.Conv2d(1,2,16,stride=4,padding=0)#background
    def forward(self,x,soft,hard):
        x = x.reshape(-1,x.shape[-3],x.shape[-2],x.shape[-1])
        #encoder branch returns sparse embeddings + mask
        x_emb, idxs = self.encoder(x,soft,hard)
        #decoder branch returns reconstructed images
        x_rec = self.decoder(x_emb)
        mu = torch.mean(torch.mean(self.conv_bg(x[...,0].unsqueeze(dim=1)),2),2)
        return x_rec, idxs, mu

class Encoder(nn.Module):
    def __init__(self,list_of_params):
        super(Encoder, self).__init__()
        self.size = list_of_params[2]#usually 8, encoding 4xsize
        self.conv1_real = nn.Conv2d(1,self.size,1,stride=1,padding=0)
        self.conv1_imag = nn.Conv2d(1,self.size,1,stride=1,padding=0)
        self.conv2_real = nn.Conv2d(self.size,2*self.size,3,stride=1,padding=1)
        self.conv2_imag = nn.Conv2d(self.size,2*self.size,3,stride=1,padding=1)
        self.conv3_real = nn.Conv2d(2*self.size,4*self.size,5,stride=1,padding=2)
        self.conv3_imag = nn.Conv2d(2*self.size,4*self.size,5,stride=1,padding=2)
        self.conv4_real = nn.Conv2d(4*self.size,4*self.size,5,stride=1,padding=2)
        self.conv4_imag = nn.Conv2d(4*self.size,4*self.size,5,stride=1,padding=2)
        self.lam = list_of_params[0]#
        self.m = list_of_params[1]#
        self.actv = nn.ReLU()
        self.pool = nn.Sequential(nn.LPPool2d(1,self.m,stride=1),nn.ReplicationPad2d(int(self.m/2)))
    def forward(self, xin, soft_NMS='True',hard_NMS='True'):
        x_real = torch.unsqueeze(xin[...,0],dim=1)
        x_imag = torch.unsqueeze(xin[...,1],dim=1)
        out_real = self.conv1_real(x_real)-self.conv1_imag(x_imag)
        out_imag = self.conv1_real(x_imag)+self.conv1_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        out_real = self.conv2_real(x_real)-self.conv2_imag(x_imag)
        out_imag = self.conv2_real(x_imag)+self.conv2_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        out_real = self.conv3_real(x_real)-self.conv3_imag(x_imag)
        out_imag = self.conv3_real(x_imag)+self.conv3_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        out_real = self.conv4_real(x_real)-self.conv4_imag(x_imag)
        out_imag = self.conv4_real(x_imag)+self.conv4_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        x = torch.stack([x_real,x_imag],dim=x_real.dim())
        if eval(soft_NMS):
            B = torch.mean(complex_mag(x),dim=1)
            A = torch.exp(self.lam*B)
            Softmx = torch.div(A,torch.squeeze(self.pool(A.reshape(A.shape[0],1,A.shape[1],A.shape[2]))))
            M = Softmx
            M[:,0:int(self.m/2),:] = 0 #ignore edge effects
            M[:,M.shape[1]-int(self.m/2):M.shape[1],:] = 0 #ignore edge effects
            M[:,:,0:int(self.m/2)] = 0 #ignore edge effects
            M[:,:,M.shape[2]-int(self.m/2):M.shape[2]] = 0 #ignore edge effects
            sparse_idx = []
            if eval(hard_NMS):
                with torch.no_grad():
                    B = torch.squeeze(self.pool(B.reshape(B.shape[0],1,B.shape[1],B.shape[2])))
                    M,sparse_idx = NMS(M.data.detach()*B.data.detach(),self.m, limit = False)
        else:
            if eval(hard_NMS):
                M = torch.mean(torch.sqrt(torch.pow(x_real,2)+torch.pow(x_imag,2)),dim=1)
                with torch.no_grad():
                    M, sparse_idx = NMS(M.data.detach(),self.m)
            else:
                M = torch.ones(x.shape[0],x.shape[2],x.shape[3]).double().to(x.device)
                sparse_idx = []
        #generate list of embeddings based on M non-zero locations
        return x*torch.reshape(M,(M.shape[0],1,M.shape[1],M.shape[2],1)).expand(-1,x.shape[1],-1,-1,2),sparse_idx

def NMS(Ms,m,limit = True,div = 4):
    sparse_mask = torch.zeros_like(Ms)
    sparse_idx = list()
    for kk in range(Ms.shape[0]):#loop over samples
        M = Ms[kk,:,:]
        aux_m,_ = torch.median(M,dim=0)
        aux_m,_ = torch.median(aux_m,dim=0)
        peaks = torch.nonzero(M>(aux_m)+(torch.max(M)-aux_m)/div)#25% above median
        if len(peaks)>0:
            xy = peaks.split(1,dim=1)#20% above median is the image-dependent threshold
            intensity = M[xy]
            _,idx_desc = torch.sort(intensity,dim=0,descending=True)#sort thresholded pixels by intensity
            ii=0
            if limit:
                 idx_desc = idx_desc[0:1000]#500
            max_n = np.inf
            while ii<min(len(idx_desc),max_n):#remove non-maximum pixels in mxm neighborhood
                remove_idx = torch.nonzero((torch.squeeze((xy[0][idx_desc]-xy[0][idx_desc[ii]])**2+(xy[1][idx_desc]-xy[1][idx_desc[ii]])**2<m**2)))
                remove_idx = remove_idx[remove_idx!=ii]
                idx_desc = np.delete(idx_desc, remove_idx)
                ii = ii +1
            if len(idx_desc)>max_n:
                idx_desc = idx_desc[0:max_n]
            sparse_mask[kk,xy[0][idx_desc],xy[1][idx_desc]] = 1
            sparse_idx.append(torch.squeeze(torch.stack([xy[0][idx_desc],xy[1][idx_desc]],dim=1)))
        else:
            sparse_mask[kk,:,:] =torch.zeros(Ms.shape[1],Ms.shape[2])
            sparse_idx.append(torch.empty(0))
    #print([len(sparse_idx[l]) for l in range(len(sparse_idx))])
    return sparse_mask, sparse_idx

class Classifier_synthetic(nn.Module):
    def __init__(self,list_of_params):
        super(Classifier_synthetic, self).__init__()
        self.fc_layer1 = nn.Linear(list_of_params[0]*8,list_of_params[0]*16)
        self.fc_layer2 = nn.Linear(list_of_params[0]*16,list_of_params[0]*32)
        self.fc_layer3= nn.Linear(list_of_params[0]*32,list_of_params[0]*64)
        self.fc_layer4= nn.Linear(list_of_params[0]*64,list_of_params[0]*16)
        self.fc_layer5= nn.Linear(list_of_params[0]*16,list_of_params[0])
        self.linear = nn.Linear(list_of_params[0],3)
        self.apply_softmax = list_of_params[1]
        self.softmx = nn.Softmax(dim=3)
    def forward(self, x):
        N = x.shape[0]
        yout = torch.zeros(N,3)
        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        x = F.relu(x)
        x = self.fc_layer3(x)
        x = F.relu(x)
        x = self.fc_layer4(x)
        x = F.relu(x)
        x = self.fc_layer5(x)
        x = F.relu(x)
        x = self.linear(x)
        if self.apply_softmax:
            x = self.softmx(x)
        yout = torch.squeeze(x)
        return yout.double().to('cuda')


def vectorize_detections(x_emb,idxs):
    sizes = np.array([len(idxs[i]) for i in range(len(idxs))])
    batch_size = len(idxs)
    N = int(sum(sizes))
    x = torch.zeros(N,x_emb.shape[1]*x_emb.shape[4])
    residual = 0
    aux = 0
    for i in range(batch_size):
        if len(idxs[i].shape)>1:
            n_sel = idxs[i].shape[0]
            x[aux:aux+n_sel,:] = torch.squeeze(x_emb[i,:,idxs[i][:,0],idxs[i][:,1]]).transpose(0,1).reshape(n_sel,x.shape[1])
        else:
            n_sel = 1
            x[aux:aux+n_sel,:] = torch.squeeze(x_emb[i,:,idxs[i][0],idxs[i][1]]).transpose(0,1).reshape(n_sel,x.shape[1])
            N = N-1
        aux = aux+ n_sel
    print(str(N)+' objects detected')
    x = x[0:N,:]
    x = x.reshape(x.shape[0],1,1,x.shape[1]).double().to('cuda')
    return x

class Decoder(nn.Module):
    def __init__(self,list_of_params):
        super(Decoder, self).__init__()
        self.size = list_of_params[0]
        self.conv1_real = nn.Conv2d(4*self.size,4*self.size,9,stride=1,padding=4,bias=False)
        self.conv1_imag = nn.Conv2d(4*self.size,4*self.size,9,stride=1,padding=4,bias=False)
        self.conv2_real = nn.Conv2d(4*self.size,2*self.size,9,stride=1,padding=4,bias=False)
        self.conv2_imag = nn.Conv2d(4*self.size,2*self.size,9,stride=1,padding=4,bias=False)
        self.conv3_real = nn.Conv2d(2*self.size,1,7,stride=1,padding=3,bias = False)
        self.conv3_imag = nn.Conv2d(2*self.size,1,7,stride=1,padding=3,bias = False)
        self.actv = nn.ReLU()
    def forward(self, x):
        x_real = x[...,0]
        x_imag = x[...,1]
        out_real = self.conv1_real(x_real)-self.conv1_imag(x_imag)
        out_imag = self.conv1_real(x_imag)+self.conv1_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        out_real = self.conv2_real(x_real)-self.conv2_imag(x_imag)
        out_imag = self.conv2_real(x_imag)+self.conv2_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        out_real = self.conv3_real(x_real)-self.conv3_imag(x_imag)
        out_imag = self.conv3_real(x_imag)+self.conv3_imag(x_real)
        x_real = self.actv(out_real)
        x_imag = self.actv(out_imag)
        x_real = out_real
        x_imag = out_imag
        return torch.stack([x_real,x_imag],dim=x_real.dim())

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight,gain=1)
        try:
            nn.init.constant_(m.bias,1)
        except:
            pass
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=10)
        try:
            nn.init.constant_(m.bias,1)
        except:
            pass
def flip(T,ax):
    idx = [i for i in range(T.size(ax)-1,-1,-1)]

    return T.index_select(ax,torch.LongTensor(idx))

class CIFAR_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 256, 1)
        self.conv9 = nn.Conv2d(256, 128, 1)
        self.fc = nn.Linear(128, 10)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.LRelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x = self.LRelu(self.conv1(x))
        x = self.LRelu(self.conv2(x))
        x = self.LRelu(self.conv3(x))
        x = F.dropout(self.maxpool(x))
        x = self.LRelu(self.conv4(x))
        x = self.LRelu(self.conv5(x))
        x = self.LRelu(self.conv6(x))
        x = F.dropout(self.maxpool(x))
        x = self.LRelu(self.conv7(x))
        x = self.LRelu(self.conv8(x))
        x = self.LRelu(self.conv9(x))
        x = self.softmax(self.fc(torch.squeeze(x)))
        return x

