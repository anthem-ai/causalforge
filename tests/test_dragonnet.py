from causalforge.model import Model , PROBLEM_TYPE
from causalforge.data_loader import DataLoader 

def test_factory():
     
    # load IHDP dataset 
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r
    
    # model 
    params={}
    params['input_dim'] = X_tr.shape[1] 
    params['neurons_per_layer'] = 200 
    params['reg_l2'] = 0.01
    params['targeted_reg'] = True 
    params['verbose'] = True 
    params['val_split'] = 0.2
    params['ratio'] = 1.0
    params['batch_size'] = 64
    params['epochs'] = 100
    params['learning_rate'] = 1e-5
    params['momentum'] = 0.9
    params['use_adam'] = True
    params['adam_epochs'] = 30
    params['adam_learning_rate'] = 1e-3
    
    dragonnet = Model.create_model("dragonnet", 
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)
    
    assert dragonnet.model is not None 
    
    # train
    idx = 1 
    X,t,y = X_tr[:,:,idx],T_tr[:,idx],YF_tr[:,idx]
    dragonnet.fit(X,t,y)
    
    # test 
    Xte =  X_te[:,:,idx]
    pred_test_ite = dragonnet.predict_ite(Xte)
    
    # error 
    
    
    
    
    
    