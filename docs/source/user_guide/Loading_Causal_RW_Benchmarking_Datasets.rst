Loading Real-World & Benchmarking Datasets with CausalForge
===========================================================

A dataset used for causal studies is, in general, different from
datasets used for associational studies. In causal inference we have a
fundamental problem which is, indeed, referred as fundamental problem of
causal inference.

**Fundamental problem of causal inference (FPCI)**: *we do not observe
both potential outcomes (control & treated), but we only observe one.*

Indeed, this formulation of FPCI holds only for binary exposures, when
we have two cohorts: treated & control (or untreated). In case of T
exposures, we observe only one potential outcome and we do not observe
the remaining T-1. An unobserved potential outcome is generally referred
as **counterfactual**.

In causal inference we have

-  **real-world datasets**: these datasets are real datasets; they are
   typically observational and obey to the FPCI, hence, they don’t come
   with counterfactuals; as a conseguence, it should not be possible to
   do causal model validation on this kind of daatsets, although they
   are people who adopt associational metrics (e.g. accuracy or auROC)
   pretending they are proxies of causal metrics like PEHE (Precision in
   Estimation of Heterogeneous Effect) or ATE (Average Treatment Effect)
   MAE (Mean Absolute Error), which, on the contrary, requires
   counterfactuals for computation;

-  **benchmarking datasets**: these datasets can be either simulations
   or combinations of real-world datasets and RCTs, and **they come with
   counterfactuals**; they can be used to do causal model validation.

With CausalForge it is very easy to load a dataset. First, you want to
load a proper **data loader** given the name of the dataset, and then
you want to load all the typical ingredients of a causal inference
dataset, i.e., **covariates**, **teatment assigments**, **factuals**
and, if available, **counterfactuals**.

Let’s see this in action with a dataset very popular in the causal
inference community: **IHDP**.

The IHDP Dataset
----------------

The Infant Health and Development Program (IHDP) is a randomized
controlled study designed to evaluate the effect of home visit from
specialist doctors on the cognitive test scores of premature infants.
The datasets is first used for benchmarking treatment effect estimation
algorithms in Hill [1], where selection bias is induced by removing
non-random subsets of the treated individuals to create an observational
dataset, and the outcomes are generated using the original covariates
and treatments. It contains 747 subjects and 25 variables. In order to
compare our results with the literature and make our results
reproducible, we use the simulated outcome implemented as setting “A” in
[2], and downloaded the data at https://www.fredjo.com/, which is
composed of 1000 repetitions of the experiment.

.. code:: ipython3

    from causalforge.data_loader import DataLoader 
    
    r = DataLoader.get_loader('IHDP').load()
    X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr, X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te = r 
    
    X_tr.shape, T_tr.shape, YF_tr.shape, YCF_tr.shape, mu_0_tr.shape , mu_1_tr.shape




.. parsed-literal::

    ((672, 25, 1000),
     (672, 1000),
     (672, 1000),
     (672, 1000),
     (672, 1000),
     (672, 1000))



.. code:: ipython3

    X_te.shape, T_te.shape, YF_te.shape , YCF_te.shape, mu_0_te.shape , mu_1_te.shape 




.. parsed-literal::

    ((75, 25, 1000), (75, 1000), (75, 1000), (75, 1000), (75, 1000), (75, 1000))



x, t, yf, ycf, mu0, mu1 are covariates, treatment, factual outcome,
counterfactual outcome, and noiseless potential outcomes respectively.

Hence,

-  **for the trainset** X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr are
   covariates, treatment, factual outcome, counterfactual outcome, and
   noiseless potential outcomes respectively;
-  **for the testset** X_te, T_te, YF_te, YCF_te, mu_0_te, mu_1_te are
   covariates, treatment, factual outcome, counterfactual outcome, and
   noiseless potential outcomes respectively;

References
----------

1. `Hill J.L., Bayesian nonparametric modeling for causal inference, J.
   Comput. Graph. Statist., 20 (1) (2011), pp. 217-240,
   10.1198/jcgs.2010.08162 <https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162>`__

2. Shi C., Blei D.M., Veitch V., Adapting neural networks for the
   estimation of treatment effects Wallach H.M., Larochelle H.,
   Beygelzimer A., d’Alché Buc F., Fox E.B., Garnett R. (Eds.), NeurIPS
   (2019), pp. 2503-2513
