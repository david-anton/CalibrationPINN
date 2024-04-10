import torch

from parametricpinn.training.loss_2d.momentum_linearelasticity_K_G import (
    calculate_E_from_K_and_G_factory,
    calculate_nu_from_K_and_G_factory,
)

material_model = "plane stress"
calculate_E_from_K_and_G = calculate_E_from_K_and_G_factory(material_model)
calculate_nu_from_K_and_G = calculate_nu_from_K_and_G_factory(material_model)


min_bulk_modulus = 100000
max_bulk_modulus = 200000
min_shear_modulus = 60000
max_shear_modulus = 100000

bulk_moduli = torch.linspace(min_bulk_modulus, max_bulk_modulus, 1000)
shear_moduli = torch.linspace(min_shear_modulus, max_shear_modulus, 1000)
bulk_shear_moduli_combinations = torch.cartesian_prod(bulk_moduli, shear_moduli)


poisson_ratios = calculate_nu_from_K_and_G(
    K=bulk_shear_moduli_combinations[:, 0], G=bulk_shear_moduli_combinations[:, 1]
)
youngs_moduli = calculate_E_from_K_and_G(
    K=bulk_shear_moduli_combinations[:, 0], G=bulk_shear_moduli_combinations[:, 1]
)


print(f"Min Poissons ratio: {torch.min(poisson_ratios)}")
print(f"Max Poissons ratio: {torch.max(poisson_ratios)}")

print(f"Min Youngs moduli: {torch.min(youngs_moduli)}")
print(f"Max Youngs moduli: {torch.max(youngs_moduli)}")

print(calculate_nu_from_K_and_G(max_bulk_modulus, min_shear_modulus))
