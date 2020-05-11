import pytest
import minplascalc as mpc


@pytest.fixture
def mixture():
    c = mpc.Mixture(species=[mpc.species_from_name(sp) for sp in 
                             ['O2','O2+','O','O+','O++']],
                    x0=[1,0,0,0,0],
                    T=1000, P=101325)
    return c


def test_solver(mixture):
    mixture.T = 1000

    assert mixture.calculate_density() == pytest.approx(0.3899566)


LOW_T = 1000
HIGH_T = 25000
LOW_P = 10132.5
HIGH_P = 1013250

@pytest.mark.parametrize("T, P, result, tol", [
    (LOW_T, LOW_P, 1081.252, 1e-3),
    (HIGH_T, LOW_P, 23193.92, 1e-2),
    (LOW_T, HIGH_P, 1081.252, 1e-3),
    (HIGH_T, HIGH_P, 5829.868, 1e-3),
])
def test_heat_capacity(mixture, T, P, result, tol):
    mixture.T = T
    mixture.P = P

    thisresult = mixture.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("T, P, result, tol", [
    (LOW_T, LOW_P, -1.455147e7, 1e1),
    (HIGH_T, LOW_P, 1.893144e8, 1e2),
    (LOW_T, HIGH_P, -1.455147e7, 1e1),
    (HIGH_T, HIGH_P, 1.483885e8, 1e2),
])
def test_enthalpy(mixture, T, P, result, tol):
    mixture.T = T
    mixture.P = P

    assert mixture.calculate_enthalpy() == pytest.approx(result, abs=tol)


def test_calculate_viscosity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_viscosity()


def test_calculate_thermal_conductivity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_thermal_conductivity()


def test_calculate_electrical_conductivity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_electrical_conductivity()


def test_calculate_total_emission_coefficient(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_total_emission_coefficient()

