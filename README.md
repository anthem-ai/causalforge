<img alt="causalforge-logo" class="causalforge-logo"  height="250" width="300"  src="https://raw.githubusercontent.com/anthem-ai/causalforge/main/logo.png">

# CausalForge

[![PyPI version](https://badge.fury.io/py/causalforge.svg)](https://badge.fury.io/py/causalforge)
[![Documentation Status](https://readthedocs.org/projects/causalforge/badge/?version=latest)](https://causalforge.readthedocs.io/en/latest/?badge=latest)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)


CausalForge is a Python package that provides a suite of modeling & causal inference methods using machine learning algorithms based on Elevence Health recent research. It provides convenient APIs that allow to estimate Propensity Score, Average Treatment Effect (ATE), Conditional Average Treatment Effect (CATE) or Individual Treatment Effect (ITE) 
from experimental or observational data. Methods have been redesigned for production. [Check out the documentation.](https://causalforge.readthedocs.io/en/latest/?badge=latest)

<details>
  <summary> <H3>Installing Python Package</H3>  </summary>

We recommend to create a proper enviroment with tensorflow and pytorch 
installed. For example, for a local Mac enviroment without GPUs: 

```sh
conda env create -f env_mac.yml
conda activate causalforge
```

You can install it after cloning this repository, i.e.

```sh
git clone https://github.com/anthem-ai/causalforge
cd causalforge
[sudo] pip install -e . [--trusted-host pypi.org --trusted-host files.pythonhosted.org]
```

or directly from the repository (development), i.e.

```sh
pip install --upgrade git+https://github.com/anthem-ai/causalforge [--trusted-host pypi.org --trusted-host files.pythonhosted.org]
```

or directly from PyPI, i.e.

```sh
pip install causalforge
```

After installing you can import classes and methods, e.g.

```python
import causalforge
causalforge.__version__
```
</details>

<details>
  <summary> <H3>Testing</H3>  </summary>
  
```bash
cd tests
pytest --disable-warnings 
```

</details>

## Citation

```bibtex
@article{tesei2023learning,
  title={Learning end-to-end patient representations through self-supervised covariate balancing for causal treatment effect estimation},
  author={Tesei, Gino and Giampanis, Stefanos and Shi, Jingpu and Norgeot, Beau},
  journal={Journal of Biomedical Informatics},
  volume={140},
  pages={104339},
  year={2023},
  publisher={Elsevier}
}
```

