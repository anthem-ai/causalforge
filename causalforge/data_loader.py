from abc import ABC, abstractmethod
import os, os.path
import numpy as np 
import requests , zipfile



def download_url(url, save_path, chunk_size=128):
    print(">>> downloading ",url," into ",save_path,"...")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


class DataLoader(ABC):
    
    def __init__(self):
        self.loaded = False  

    @staticmethod
    def get_loader(dataset_name='IHDP'):
        if dataset_name=='IHDP':
            return IHDPLoader()
        else:
            raise Exception('dataset not supported::'+str(dataset_name))
            
    @abstractmethod
    def load(self):
        pass 


    
    def split(self,X_df,W_df,Y_df,test_size=None,random_state=None):
        #split
        assert test_size is not None and test_size >0 and test_size < 1
        if random_state is not None:
            np.random.seed(random_state)
        msk = np.random.rand(len(X_df)) > test_size
        #
        X_df_tr = X_df[msk]
        W_df_tr = W_df[msk]
        Y_df_tr = Y_df[msk]
        #
        X_df_te = X_df[~msk]
        W_df_te = W_df[~msk]
        Y_df_te = Y_df[~msk]
        #
        return X_df_tr,W_df_tr,Y_df_tr, X_df_te,W_df_te,Y_df_te




class IHDPLoader(DataLoader):
    
    def __init__(self):
        super(IHDPLoader, self).__init__()

    def load(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(*[my_path, "data"])
        path_train_zip = os.path.join(*[my_path, "data" , "ihdp_npci_1-1000.train.npz.zip"])
        path_train = os.path.join(*[my_path, "data" , "ihdp_npci_1-1000.train.npz"])
        path_test_zip = os.path.join(*[my_path, "data" , "ihdp_npci_1-1000.test.npz.zip"])
        path_test = os.path.join(*[my_path, "data" , "ihdp_npci_1-1000.test.npz"])
        if not os.path.exists(path):
            os.makedirs(path)
            # download trainset 
            download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip", path_train_zip) 
            with zipfile.ZipFile(path_train_zip, 'r') as zip_ref:
                zip_ref.extractall(path)
            # download testset 
            download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip", path_test_zip) 
            with zipfile.ZipFile(path_test_zip, 'r') as zip_ref:
                zip_ref.extractall(path)
        # load 
        train_cv = np.load(path_train)
        test = np.load(path_test)
    
        self.X_tr    = train_cv.f.x.copy()
        self.T_tr    = train_cv.f.t.copy()
        self.YF_tr   = train_cv.f.yf.copy()
        self.YCF_tr  = train_cv.f.ycf.copy()
        self.mu_0_tr = train_cv.f.mu0.copy()
        self.mu_1_tr = train_cv.f.mu1.copy()
        
        self.X_te    = test.f.x.copy()
        self.T_te    = test.f.t.copy()
        self.YF_te   = test.f.yf.copy()
        self.YCF_te  = test.f.ycf.copy()
        self.mu_0_te = test.f.mu0.copy()
        self.mu_1_te = test.f.mu1.copy()
        
        self.loaded = True
        
        return self.X_tr,self.T_tr, self.YF_tr, self.YCF_tr, self.mu_0_tr, self.mu_1_tr, \
            self.X_te, self.T_te, self.YF_te, self.YCF_te, self.mu_0_te, self.mu_1_te
    
    
    def __len__(self):
        if not self.loaded:
            self.load()
        return self.X_tr.shape[-1]
    
    def __getitem__(self, idx):
        if not self.loaded:
            self.load()
        return self.X_tr[:,:,idx], self.T_tr[:,idx], self.YF_tr[:,idx], self.YCF_tr[:,idx],self.mu_0_tr[:,idx], self.mu_1_tr[:,idx], \
            self.X_te[:,:,idx], self.T_te[:,idx], self.YF_te[:,idx], self.YCF_te[:,idx], self.mu_0_te[:,idx], self.mu_1_te[:,idx]
        
        

            
            
            
