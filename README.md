# **CalibrationPINN**: Physics-informed neural networks for model calibration

[![DOI](https://zenodo.org/badge/803810465.svg)](https://zenodo.org/doi/10.5281/zenodo.11368998)

The research code **CalibrationPINN** provides a framework for the calibration of constitutive models from full-field displacement data. In particular, this code repository provides the software for the related scientific publications:

1. [*"Deterministic and statistical calibration of constitutive models from full-field data with parametric physics-informed neural networks"*](#deterministic-and-statistical-calibration-of-constitutive-models-from-full-field-data-with-parametric-physics-informed-neural-networks)

This code is supposed to be executed in a [*Singularity container*](https://sylabs.io). You can find the [installation instructions](#installation) below.



## Related scientific publications


### Deterministic and statistical calibration of constitutive models from full-field data with parametric physics-informed neural networks

<!-- The [full paper]() is available open source. -->

**Citing**:

    @article{anton_parametricPINNsCalibration,
        title={Deterministic and statistical calibration of constitutive models from full-field data with parametric physics-informed neural networks},
        author={Anton, David and Tröger, Jendrik-Alexander and Wessels, Henning and Römer, Ulrich and Henkes, Alexander and Hartmann, Stefan},
        year={2024},
        journal={arXiv preprint},
        doi={https://doi.org/10.48550/arXiv.2405.18311}
    }

<!-- @article{anton_parametricPINNsCalibration,
    title={Deterministic and statistical calibration of constitutive models from full-field data with parametric physics-informed neural networks},
    author={Anton, David and Tröger, Jendrik-Alexander and Wessels, Henning and Römer, Ulrich and Henkes, Alexander and Hartmann, Stefan},
    year={2024},
    journal={arXiv preprint},
    volume={},
    number={}
    pages={},
    doi={}
} -->

The results in this publication can be reproduced with the following **scripts**, which can be found at the top level of this repository:
- *parametric_pinns_calibration_paper_synthetic_linearelasticity.py* 
- *parametric_pinns_calibration_paper_synthetic_neohooke.py*
- *parametric_pinns_calibration_paper_experimental_linearelasticity.py*
    -   Before simulation, the experimental data must be copied to the directory *input/parametric_pinns_calibration_paper*, see project file structure below. The experimental data set is published on [Zenodo](https://zenodo.org) with the DOI: [10.5281/zenodo.11257192](https://doi.org/10.5281/zenodo.11257192). Please note that the input directories may need to be created first if they do not already exist.
    -   `use_interpolated_calibration_data`: Determines whether the raw or interpolated measurement data is used for calibration.
- *parametric_pinns_calibration_paper_bcscomparison.py*
    -   `use_stress_symmetry_bcs`: Determines whether the stress symmetry boundary conditions are used.

> [!IMPORTANT]
> Some other flags are defined at the beginning of the scripts, which control, for example, whether the parametric PINN is retrained or whether the data is regenerated. In principle, the parametric PINN does not have to be retrained for each calibration and the data can also be reused as long as the setup does not change and the correct paths are specified.



## Installation


1. For strict separation of input/output data and the source code, the project requires the following file structure:

project_directory \
├── app \
├── input \
└── output

> [!NOTE]
> The output folder is normally created automatically, if it does not already exist. If you are not using any experimental data that needs to be saved in the input folder before the simulation, the input folder is also created automatically.

2. Clone the repository into the *app* folder via:

        git clone https://github.com/david-anton/CalibrationPINN.git .

3. Install the software dependencies. This code is supposed to be executed in a [*Singularity container*](#singularity). In addition, due to the high computational costs, we recommend running the simulations on a GPU. 

4. Run the code.


### Singularity

You can find the singularity definition file in the *.devcontainer* directory. To build the image, navigate to your *project_directory* (see directory tree above) and run:

    singularity build calibrationpinn.sif app/.devcontainer/container.def

Once the image is built, you can run the scripts via:

    singularity run --nv calibrationpinn.sif python3 <full-path-to-script>/<script-name>

Please replace `<full-path-to-script>` and `<script-name>` in the above command according to your file structure and the script you want to execute.

> [!IMPORTANT]
> You may have to use the *fakreroot* option of singularity if you do not have root rights on your system. In this case, you can try building the image by running the command `singularity build --fakeroot calibrationpinn.sif app/.devcontainer/container.def`. However, the fakeroot option must be enabled by your system administrator. For further information, please refer to the [Singularity documentation](https://sylabs.io/docs/).



## Citing


If you use this research code, please cite the [related scientific publications](#related-scientic-publications) and the code as follows:

    @misc{anton_calibrationPINN,
        title={CalibrationPINN: Physics-informed neural networks for model calibration},
        author={Anton, David},
        year={2024},
        publisher={Zenodo},
        doi={https://doi.org/10.5281/zenodo.11368998},
        note={Code available from https://github.com/david-anton/CalibrationPINN}
    }