import torch
import scipy.io as sio
import numpy as np
from utils.create_synthetic_WBC_subtype_dataset import Synthetic_WBC_subtype_Dataset
from utils.architecture import Encoder, Decoder, Classifier_synthetic, vectorize_detections

def main():
    #settings
    dataset_fold = 'test'#'train','test' or 'val'  
    data_directory = '/data/'#modify accordingly
    classifier_loss = 'VP_CE'#VP_CE or VP_L2
    lam = 75    
    kernel_size = 8
    soft = 'True'
    hard = 'True'
    m = 17
    #load dataset
    dataset = Synthetic_WBC_subtype_Dataset(data_folder = data_directory+'data_'+dataset_fold+'/')
    print('Number of samples in '+dataset_fold+' set:'+str(len(dataset)))
    #create and load trained encoder/decoder
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    path = '/models/Encoder_Decoder_synthetic_WBC.pt'
    print('pretrained model:'+path)
    encoder = Encoder([lam,m,kernel_size]).double()
    encoder.to(device)
    state = encoder.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k.replace('encoder.',''): v for k, v in pretrained_dict.items() if k.replace('encoder.','') in state}
    state.update(pretrained_dict)
    encoder.load_state_dict(state)
    decoder = Decoder([kernel_size]).double()
    decoder.to(device)
    state = decoder.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k.replace('decoder.',''): v for k, v in pretrained_dict.items() if k.replace('decoder.','') in state}
    state.update(pretrained_dict)
    decoder.load_state_dict(state)
    #create and load trained classifier 
    classifier_model = eval('Classifier_synthetic'+'([kernel_size,True]).double()')
    classifier_model.to(device)
    path_classifier = '/models/Classifier_synthetic_'+classifier_loss+'.pt'
    state_classifier = classifier_model.state_dict()
    pretrained_dict = torch.load(path_classifier)
    pretrained_keys = {k:v for k, v in pretrained_dict.items() if (k in state_classifier)}
    print('pretrained classifier:'+path_classifier)
    state_classifier.update(pretrained_keys)
    classifier_model.load_state_dict(state_classifier)
    yprop = np.zeros((len(dataset),3))#store GT proportions
    yprop_hat = np.zeros((len(dataset),3))#store predicted proportions
    with torch.no_grad():
        #loop through dataset
        for idx_donor in range(len(dataset)):
            H, X, y = dataset[idx_donor]
            x_emb, idxs = encoder(X.to(device),soft,hard)#x_emb: embeddings, idxs: locations
            rec = decoder(x_emb)#reconstructed images
            features = vectorize_detections(x_emb,idxs)#features for classifier
            yhat = classifier_model(features.to(device))#predicted classes
            yaux = torch.zeros_like(yhat)
            yprop_hat[idx_donor,:] = torch.mean(yaux.scatter(1,yhat.argmax(dim=1).repeat(3,1).transpose(1,0),1),dim=0).data.cpu().numpy()#predicted proportions
            yprop[idx_donor,:] = np.array(y)
            #note that in this loop only proportions are saved

if __name__=='__main__':
    main() 


