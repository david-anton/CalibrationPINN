from dataclasses import dataclass

import pytest

from parametricpinn.training.loss_2d.momentumbase_linearelasticity import (
    calculate_E_from_K_and_G_factory,
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    calculate_nu_from_K_and_G_factory,
)


@dataclass
class SetupPlaneStrain:
    model = "plane strain"
    youngs_modulus_E = 1.0
    poissons_ratio_nu = 1 / 4
    bulk_modulus_K = 4 / 5
    shear_modulus_G = 2 / 5


@dataclass
class SetupPlaneStress:
    model = "plane stress"
    youngs_modulus_E = 1.0
    poissons_ratio_nu = 1 / 4
    bulk_modulus_K = 2 / 3
    shear_modulus_G = 2 / 5


setup_plane_strain = SetupPlaneStrain()
setup_plane_stress = SetupPlaneStress()


@pytest.mark.parametrize("setup", [setup_plane_strain, setup_plane_stress])
def test_calculate_K_from_E_and_nu(setup: SetupPlaneStrain) -> None:
    sut = calculate_K_from_E_and_nu_factory(setup.model)

    actual = sut(E=setup.youngs_modulus_E, nu=setup.poissons_ratio_nu)

    expected = setup.bulk_modulus_K
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("setup", [setup_plane_strain, setup_plane_stress])
def test_calculate_G_from_E_and_nu(setup: SetupPlaneStrain) -> None:
    actual = calculate_G_from_E_and_nu(
        E=setup.youngs_modulus_E, nu=setup.poissons_ratio_nu
    )

    expected = setup.shear_modulus_G
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("setup", [setup_plane_strain, setup_plane_stress])
def test_calculate_E_from_K_and_G(setup: SetupPlaneStrain) -> None:
    sut = calculate_E_from_K_and_G_factory(setup.model)

    actual = sut(K=setup.bulk_modulus_K, G=setup.shear_modulus_G)

    expected = setup.youngs_modulus_E
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("setup", [setup_plane_strain, setup_plane_stress])
def test_calculate_nu_from_K_and_G(setup: SetupPlaneStrain) -> None:
    sut = calculate_nu_from_K_and_G_factory(setup.model)

    actual = sut(K=setup.bulk_modulus_K, G=setup.shear_modulus_G)

    expected = setup.poissons_ratio_nu
    assert actual == pytest.approx(expected)
