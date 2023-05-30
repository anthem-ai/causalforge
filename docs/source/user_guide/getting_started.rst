Getting Started
===============

The sections below provide a high level overview of the ``CausalForge`` package. This page takes you through installation, dependencies, main features, imputation methods supported, and basic usage of the package. It also provides links to get in touch with the authors, review our lisence, and review how to contribute.

Installation
------------

* Requires tensorflow and pytorch installed. For example, for a local Mac enviroment without GPUs, you can create one with ``conda env create -f env_mac.yml``, and you activate it with ``conda activate causalforge``. 
* Download ``CausalForge`` with ``pip install causalforge``. 
* If ``pip`` cached an older version, try ``pip install --no-cache-dir --upgrade causalforge``.
* If you want to work with the development branch, use the script below:

*Development*

.. code-block:: sh

   git clone -b dev --single-branch https://github.com/anthem-ai/causalforge
   cd causalforge
   python setup.py install

Versions and Dependencies
-------------------------


* Python 3.8+
* Dependencies:

  * ``numpy>=1.18.5``
  * ``scipy>=1.4.1``
  * ``pandas``
  * ``cython``
  * ``statsmodels``
  * ``scikit-learn``
  * ``matplotlib``
  * ``pymc``
  * ``seaborn``
  * ``tqdm``
  * ``tensorflow``
  * ``keras``
  * ``torch``


Methods Supported
----------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Paper [Link]
     - Journal/Conference
   * - BCAUSS
     - `Gino Tesei et al, Learning end-to-end patient representations through self-supervised covariate balancing for causal treatment effect estimation <https://www.sciencedirect.com/science/article/pii/S1532046423000606/pdfft?md5=923768a5e1b27765e9da9ac13c0477aa&pid=1-s2.0-S1532046423000606-main.pdf>`_ 
     - Journal of Biomedical Informatics 2023 
   * - Dragonnet
     - `Claudia Shi et al, Adapting Neural Networks for the Estimation of Treatment Effects <https://arxiv.org/pdf/1906.02120v2.pdf>`_
     - NeurIPS 2019   
   * - BCAUS
     - `Belthangady et al, Minimizing bias in massive multi-arm observational studies with BCAUS: balancing covariates automatically using supervision <https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-021-01383-x>`_
     - BMC Medical Research Methodology 2021  
   * - GANITE
     - `Jinsung Yoon et al, GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets <https://openreview.net/pdf?id=ByKWUeWA->`_
     - ICLR 2018  

License
-------

Distributed under the MIT license. See `LICENSE <https://github.com/anthem-ai/causalforge/blob/main/LICENSE>`_ for more information.

