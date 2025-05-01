# Real-time interpretation of neutron vibrational spectra with symmetry-equivariant Hessian matrix prediction

Predict the Hessian and INS of organic molecules, implemented with JAX.

### Installation
To use the package, conda environment is recommended. The suggested environment file, env_ins_mol.yml, is included in the package. To create the environment, you can use the following command:

```bash
git clone https://github.com/maplewen4/INS_molecule.git
cd INS_molecule
conda env create -f env_ins_mol.yml
```

### Usage

To train and test the machine learning model, you may use the associated dataset, https://doi.org/10.5281/zenodo.14796532. To build the training dataset, you can run `load_data.py` with proper path to the raw files. To train a model, you could run `train_hessian.py`. The architecture of the model can be modified within `Nequip.py`. To use the pre-trained model to predict the Hessian matrices and inelastic neutron scattering spectra, please use `predict_hessian_and_ins_from_strucure.py`. The pretrained model is also included in the Zenodo link.

## Citation

```
@misc{han_real-time_2025,
	title = {Real-time interpretation of neutron vibrational spectra with symmetry-equivariant {Hessian} matrix prediction},
	url = {http://arxiv.org/abs/2502.13070},
	doi = {10.48550/arXiv.2502.13070},
	publisher = {arXiv},
	author = {Han, Bowen and Zhang, Pei and Mehta, Kshitij and Pasini, Massimiliano Lupo and Li, Mingda and Cheng, Yongqiang},
	month = feb,
	year = {2025},
	note = {arXiv:2502.13070 [physics]},
	keywords = {Physics - Chemical Physics, Physics - Computational Physics},
}
```
