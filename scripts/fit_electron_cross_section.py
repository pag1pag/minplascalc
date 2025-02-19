"""Fit the electron cross section from LXCat to the formula used in minplascalc.

WORK IN PROGRESS.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from minplascalc.units import Units

u = Units()


def fit_function(tau, D1, D2, D3, D4):
    return D1 + D2 * tau**D3 * np.exp(-D4 * tau**2)


def jac_fit_function(tau, D1, D2, D3, D4):
    exp_factor = np.exp(-D4 * tau**2)
    return np.vstack(
        (
            np.ones_like(tau),
            tau**D3 * exp_factor,
            np.concatenate(
                ([0], D2 * np.log(tau[1:]) * tau[1:] ** D3 * exp_factor[1:])
            ),
            -D2 * tau ** (2.0 + D3) * exp_factor,
        )
    ).T


def fit_from_lxcat_to_minplascalc(energy, cross_section):
    """Fit the electron cross section from LXCat to the formula used in minplascalc.

    Parameters
    ----------
    energy : np.ndarray
        Energy values in eV.
    cross_section : np.ndarray
        Cross section values in m^2.

    Returns
    -------
    np.ndarray
        The values of D1, D2, D3, D4.
    """
    # Fit the cross section from LXCat to the formula used in minplascalc.
    # The formula used in minplascalc is:
    # Omega = D1 + D2 * tau**D3 * np.exp(-D4 * tau**2)
    # where tau = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar
    # and D1, D2, D3, D4 are the parameters to fit.
    # The values of D1, D2, D3, D4 are taken from the `./data/species/O.json` file.
    # These are the default values for the electron cross section of the oxygen atom.
    # The values of D1, D2, D3, D4 are fitted to the cross section from LXCat.

    temperatures = 2 / 3 * energy * u.eV_to_K  # eV to K
    taus = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar

    p0 = [
        6e-20,
        1e-32,
        1.38,
        0.27e-19,
    ]
    curve_fit_parameters, _ = curve_fit(
        fit_function,
        xdata=taus,
        ydata=cross_section,
        p0=p0,
        jac=jac_fit_function,
        method="dogbox",
        xtol=1e-15,
        gtol=1e-15,
        x_scale=[1e-20, 1e-40, 0.01, 1e-18],
        loss="huber",
        f_scale=1e-20,
        tr_solver="exact",
        bounds=(
            [1e-22, 0, 1, 1e-22],
            [1e-18, 1e-30, 10, 1e-18],
        ),
    )
    print(curve_fit_parameters)

    return list(curve_fit_parameters)


if __name__ == "__main__":
    # LXCAT
    # DATABASE:         Morgan (Kinema Research  Software)
    # PERMLINK:         www.lxcat.net/Morgan
    # DESCRIPTION:      Assembled over the course of 30 years WL Morgan and suitable for use
    # with 2-term Boltzmann solvers.
    # CONTACT:          W. Lowell Morgan, Kinema Research  Software
    # e / C
    # Effective E + C → E + C (m/M = 0.000045683, complete set) | (p. 1664, Vol. IV). Updated: 6 June 2011.
    energy_cross_section = np.array(
        [
            [0.000000e0, 1.050000e-19],
            [5.000000e-2, 1.070000e-19],
            [1.340000e-1, 1.250000e-19],
            [2.710000e-1, 1.620000e-19],
            [3.890000e-1, 2.230000e-19],
            [4.400000e-1, 2.930000e-19],
            [4.720000e-1, 3.960000e-19],
            [5.590000e-1, 4.060000e-19],
            [6.280000e-1, 3.970000e-19],
            [7.640000e-1, 3.420000e-19],
            [1.080000e0, 2.790000e-19],
            [1.410000e0, 2.370000e-19],
            [2.180000e0, 2.030000e-19],
            [3.020000e0, 1.970000e-19],
            [3.880000e0, 1.900000e-19],
            [4.820000e0, 1.880000e-19],
        ]
    )

    # LXCAT
    # DATABASE:         Morgan (Kinema Research  Software)
    # PERMLINK:         www.lxcat.net/Morgan
    # DESCRIPTION:      Assembled over the course of 30 years WL Morgan and suitable for use
    # with 2-term Boltzmann solvers.
    # CONTACT:          W. Lowell Morgan, Kinema Research  Software
    # e / H
    # Effective E + H2 -> E + H2 (m/M = 0.000272, complete set) | Updated: 6 June 2011.
    energy_cross_section = np.array(
        [
            [0.000000e0, 6.400000e-20],
            [1.000000e-3, 6.400000e-20],
            [2.000000e-3, 6.499999e-20],
            [3.000000e-3, 6.600000e-20],
            [5.000000e-3, 6.800000e-20],
            [7.000000e-3, 7.099999e-20],
            [8.500000e-3, 7.200000e-20],
            [1.000000e-2, 7.300000e-20],
            [1.500000e-2, 7.699999e-20],
            [2.000000e-2, 8.000000e-20],
            [3.000000e-2, 8.499999e-20],
            [4.000000e-2, 8.960000e-20],
            [5.000000e-2, 9.280000e-20],
            [7.000000e-2, 9.850000e-20],
            [1.000000e-1, 1.050000e-19],
            [1.200000e-1, 1.085000e-19],
            [1.500000e-1, 1.140000e-19],
            [1.700000e-1, 1.160000e-19],
            [2.000000e-1, 1.200000e-19],
            [2.500000e-1, 1.250000e-19],
            [3.000000e-1, 1.300000e-19],
            [3.500000e-1, 1.345000e-19],
            [4.000000e-1, 1.390000e-19],
            [5.000000e-1, 1.470000e-19],
            [7.000000e-1, 1.630000e-19],
            [1.000000e0, 1.740000e-19],
            [1.200000e0, 1.780000e-19],
            [1.300000e0, 1.800000e-19],
            [1.500000e0, 1.825000e-19],
            [1.700000e0, 1.825000e-19],
            [1.900000e0, 1.810000e-19],
            [2.100000e0, 1.790000e-19],
            [2.200000e0, 1.770000e-19],
            [2.500000e0, 1.700000e-19],
            [2.800000e0, 1.640000e-19],
            [3.000000e0, 1.600000e-19],
            [3.300000e0, 1.560000e-19],
            [3.600000e0, 1.480000e-19],
            [4.000000e0, 1.400000e-19],
            [4.500000e0, 1.310000e-19],
            [5.000000e0, 1.220000e-19],
            [6.000000e0, 1.040000e-19],
            [7.000000e0, 8.899999e-20],
            [8.000000e0, 7.850000e-20],
            [1.000000e1, 6.000000e-20],
            [1.200000e1, 5.200000e-20],
            [1.500000e1, 4.500000e-20],
            [1.700000e1, 4.200000e-20],
            [2.000000e1, 3.900000e-20],
            [2.500000e1, 3.600000e-20],
            [3.000000e1, 3.400000e-20],
            [5.000000e1, 2.900000e-20],
            [7.500000e1, 2.600000e-20],
            [1.000000e2, 2.300000e-20],
            [1.500000e2, 1.900000e-20],
            [2.000000e2, 1.620000e-20],
            [3.000000e2, 1.280000e-20],
            [5.000000e2, 9.200000e-21],
            [7.000000e2, 7.200000e-21],
            [1.000000e3, 5.400000e-21],
        ]
    )

    energy = energy_cross_section[:, 0]
    cross_section = energy_cross_section[:, 1]

    max_energy = 1e1  # eV
    mask = energy <= max_energy
    energy = energy[mask]
    cross_section = cross_section[mask]

    curve_fit_parameters = fit_from_lxcat_to_minplascalc(energy, cross_section)

    print(curve_fit_parameters)

    # Plot the cross section from LXCat and the fitted curve.
    temperatures = 2 / 3 * energy * u.eV_to_K  # eV to K
    taus = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar

    # %%
    plt.plot(energy, cross_section, label="LXCat")
    plt.plot(energy, fit_function(taus, *curve_fit_parameters), label="Fit")
    # electron_cross_section = [
    #     2.7716909516406733e-20,
    #     1.2091239912854124e-85,
    #     7.039441779968125,
    #     2.3051925774489085e-19,
    # ]
    # plt.plot(
    #     energy,
    #     fit_function(taus, *electron_cross_section),
    #     "--",
    #     label="Minplascalc default",
    # )

    electron_cross_section = [
        6e-20,
        1e-32,
        1.38,
        0.27e-19,
    ]
    plt.plot(
        energy,
        fit_function(taus, *electron_cross_section),
        "-.",
        label="Manual fit",
    )
    plt.title("Electron cross section of atomic carbon (C)")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Cross section [m^2]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

# %%
