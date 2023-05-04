from causalforge.model import Model , PROBLEM_TYPE

def test_factory():
    
    params={}
    params['input_dim'] = 100 
    params['neurons_per_layer'] = 200 
    params['reg_l2'] = 0.01
    
    dragonnet = Model.create_model("dragonnet", 
                                   params,
                                   problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION, 
                                   multiple_treatments=False)
    
    assert dragonnet.model is not None 