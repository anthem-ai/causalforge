from causalflow import problem_factory 
from causalflow import data_loader
from causalflow import model_factory 
from causalflow.metrics import sqrt_PEHE_with_diff

# 1. define the problem  
bcaus_problem = problem_factory.create_problem(problme_type='propensity_estimation',multiple_treatments=False,outcome='binary')
bcauss_problem = problem_factory.create_problem(problme_type='treatment_effect_estimation',granularity='ATE/ITE', multiple_treatments=False,outcome='binary')
bcauss_ext = problem_factory.create_problem(problme_type='treatment_effect_estimation',multiple_treatments=True,outcome='binary')


# 2. load proper dataset to study the problem defined 

# you can browse them 
bcaus_dataset_list = data_loader.dataset_list(bcaus_problem)
for dataset in bcaus_dataset_list:
    print(dataset)

# or if you know the name ..
X_train, W_train, Y_train, X_test, W_test, Y_test = data_loader.load(name='IHDP')
# patient_0 , 1, 1 
# patient_0 , 0, 0 
# how we respresent conterfactuals? obervational data? 


# 3. create proper dataset to study the problem defined 
# you can browse them 
bcaus_like_models = model_factory.model_list(bcaus_problem)
for model in bcaus_like_models:
    print(model)

# or if you know the name ..
bcaus_model = model_factory.create_model(name='bcaus',params={'w1': 0.009, ...}) 
# or if the same model is defined for >1 problems (e.g. outcome continous vs. discrete)
bcaus_model = model_factory.create_model(name='bcaus',params={'w1': 0.009, ...},problem=bcaus_problem) 


# 4. train 
bcaus_model.train(X_train, W_train, Y_train,params={'lr': 0.0001})
# if the library runs on LHPS we can support mlflow integration and if the model has been already trained on such dataset 
# the stub just upload the trained model with expected eval metrics 

# 5. predict treatment effect 
pred = bcaus_model.predict(X_test)

# 6. eval 
pehe = sqrt_PEHE_with_diff(Y_test, pred)
print(f"PEHE score for {model} on {dataset} = {pehe}")


