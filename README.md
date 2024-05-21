# **CalibrationPINN**: Physics-informed neural networks for model calibration"

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6913329.svg)](https://doi.org/10.5281/zenodo.6913329) -->

The research code **CalibrationPINN** provides a framework for the calibration of constitutive models from full-field displacement data. In particular, this code repository also provides the software for the corresponding scientific publications:

1. [*Deterministic and Statistical Calibration of Constitutive Models from Full-Field Data with Parametric Physics-Informed Neural Networks*](#deterministic-and-statistical-calibration-of-constitutive-models-from-full-field-data-with-parametric-physics-informed-neural-networks)
<!-- Add citation including the DOI -->

The code is supposed to be executed in a Singularity container. Alternatively, the code can also be executed in a virtual environment created with *miniconda*, for example. See [here](#installation) for installation instructions.



## Related scientic publications

### Deterministic and Statistical Calibration of Constitutive Models from Full-Field Data with Parametric Physics-Informed Neural Networks
**ABSTRACT**: The calibration of constitutive models from full-field data has recently gained increasing interest due to improvements in full-field measurement capabilities. In addition to the experimental characterization of novel materials, continuous structural health monitoring is another application that has recently emerged. However, monitoring is usually associated with severe time constraints, difficult to meet with standard numerical approaches. Therefore, parametric physics- informed neural networks (PINNs) for constitutive model calibration from full-field displacement data are investigated. In an offline stage, a parametric PINN can be trained to learn a parameterized solution of the underlying partial differential equation. In the subsequent online stage, the parametric PINN then acts as a surrogate for the parameters-to-state map in calibration. We test the proposed approach for the calibration of the linear elastic as well as a hyperelas- tic constitutive model from noisy synthetic displacement data. For this purpose, a deterministic nonlinear least-squares calibration and a Markov chain Monte Carlo-based Bayesian inference are carried out to quantify the uncertainty. A proper statistical evaluation of the results underlines the high accuracy of the deterministic calibration and that the estimated uncertainty is well-calibrated. Finally, we consider experimental data and show that the results are in good agreement with a Finite Element Method-based calibration. Due to the fast eval- uation of PINNs, calibration can be performed in near real-time. This advantage is particularly evident in many-query applications such as Markov chain Monte Carlo-based Bayesian inference.

The results from the paper can be reproduced with the following scripts:
- *parametric_pinns_calibration_paper_synthetic_linearelasticity.py* 
- *parametric_pinns_calibration_paper_synthetic_neohooke.py*
- *parametric_pinns_calibration_paper_experimental_linearelasticity.py*
- *parametric_pinns_calibration_paper_bcscomparison.py*



## Installation
1. For strict separation of input/output data and the source code, the project requires the following file structure:

|- project directory
|   |- app
|   |- input
|   |- output

Please note that the output folder is always created automatically. If you are not using experimental data that needs to be stored in the input folder, the input folder is also created automatically.

2. Clone the repository into the app folder.

3. Install the software dependencies. The code is supposed to be executed in a [*Singularity*](#singularity) container. The code can also be executed in a virtual environment created with [*miniconda*](#miniconda), for example.

4. Run the code. Due to the high computing costs, we recommend running the simulations on a GPU.


### Singularity
You can find the singularity definition file in the *.devcontainer* directory. To build the image, navigate to your projectdirectory (see directory tree above) and execute:

'''
singularity build --force calibrationpinn.sif app/.devcontainer/container.def
'''

Once the image is built, you can run the scripts via:

'''
singularity run \
 --nv \
 --nvccli \
 calibrationpinn.sif \
 python3 <full-path-to-script>/<script-name>
'''

### miniconda
Alternatively, you can create a *miniconda* environment and install the software dependencies in the envronment via:

'''
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f requirements-conda.yml
conda activate venv-calibrationpinn
'''

After installing the dependencies, you can activate the virtual environment via:

'''
conda activate venv-calibrationpinn
'''

Please note that you need *gmsh* as an additional dependency, which cannot be installed in the virtual environment. On a UNIX system, *gmsh* can be installed via:

'''
apt-get update
apt-get install gmsh
'''
