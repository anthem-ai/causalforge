from causalforge.data_loader import DataLoader 

def test_IHDP_load():
    
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r 
    

def test_IHDP_iter():
    
    loader = DataLoader.get_loader('IHDP')
    len(loader) == 1000
    
    i = 0 
    while i < len(loader):
        loader[i]
        print(i)
        i += 1 
    
