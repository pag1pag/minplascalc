r"""
Example: Calculating the properties of an :math:`H_2` plasma.
=============================================================

Compute and compare the properties of an :math:`H_2` plasma using minplascalc and reference
data from [Boulos2023]_.

Properties calculated include:

- Density
- Enthalpy
- Heat capacity
- Viscosity
- Thermal conductivity
- Electrical conductivity
"""  # noqa: D205

# %%
# Import the required libraries.
# ------------------------------
#
# We start by importing the modules we need:
#
# - matplotlib for drawing graphs,
# - numpy for array functions,
# - and of course minplascalc.

import matplotlib.pyplot as plt
import numpy as np

import minplascalc as mpc
from minplascalc.utils import get_path_to_data

# %%
# Create mixture object for the species we're interested in.
# ----------------------------------------------------------
#
# Next, we create a minplascalc LTE mixture object. Here we use a helper function
# in minplascalc which creates the object directly from a list of the species names.

hydrogen_mixture = mpc.mixture.lte_from_names(
    ["H2", "H2+", "H", "H+"], [1, 0, 0, 0], 1000, 101325
)
# %%
# Set a range of temperatures to calculate the equilibrium compositions at.
# -------------------------------------------------------------------------
#
# Next, set a range of temperatures to calculate the equilibrium compositions at - in this case
# we're going from 1000 to 25000 K in 50 K steps.
# Also initialise a list to store the property values at each temperature.

temperatures = np.arange(500, 25000, 100)
density = []
h = []
cp = []
viscosity = []
thermal_conductivity = []
electrical_conductivity = []

# %%
# Perform the composition calculations.
# --------------------------------------
#
# Now we can perform the property calculations.
#
# Note that execution of this calculation is fairly compute intensive and the following code
# snippet may take several seconds to complete.

for T in temperatures:
    hydrogen_mixture.T = T
    density.append(hydrogen_mixture.calculate_density())
    h.append(hydrogen_mixture.calculate_enthalpy())
    cp.append(hydrogen_mixture.calculate_heat_capacity())
    viscosity.append(hydrogen_mixture.calculate_viscosity())
    thermal_conductivity.append(hydrogen_mixture.calculate_thermal_conductivity())
    electrical_conductivity.append(hydrogen_mixture.calculate_electrical_conductivity())

# %%
# Load reference data.
# --------------------

# Load reference data from Boulos et al. (2023) for comparison.
# The data is stored in a CSV file in the `.\data\papers\Boulos2023` directory.

data_path = get_path_to_data("papers", "Boulos2023", "H2.csv")
data = np.genfromtxt(data_path, delimiter=",", skip_header=2)

# Extract the temperature, density, enthalpy and heat capacity data.
temperatures_ref = data[:, 0]
density_ref = data[:, 1]
enthalpy_ref = data[:, 2]
cp_ref = data[:, 3]
viscosity_ref = data[:, 4]
thermal_conductivity_ref = data[:, 5]
electrical_conductivity_ref = data[:, 6]


# %%
# Plot the results.
# -----------------
#
# Now we can visualise the properties by plotting them against temperature, to see how they vary.

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

ax = axs[0, 0]
ax.set_title(r"$\mathregular{H_2}$ plasma density at 1 atm")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\rho [kg/m^3]}$")
ax.semilogy(temperatures, density, "k", label="minplascalc")
ax.semilogy(temperatures_ref, density_ref, "k--", label="Boulos et al. (2023)")
ax.legend()

ax = axs[0, 1]
ax.set_title(r"$\mathregular{H_2}$ plasma heat capacity at 1 atm")
ax.set_xlabel("T [K]")
ax.set_ylabel(r"$\mathregular{C_P [J/(kg.K)]}$")
ax.plot(temperatures, cp, "k", label="minplascalc")
ax.plot(temperatures_ref, cp_ref, "k--", label="Boulos et al. (2023)")
ax.legend()

ax = axs[0, 2]
ax.set_title(r"$\mathregular{H_2}$ plasma enthalpy at 1 atm")
ax.set_xlabel("T [K]")
ax.set_ylabel(r"$\mathregular{H [J/kg]}$")
ax.plot(temperatures, h, "k", label="minplascalc")
ax.plot(temperatures_ref, enthalpy_ref, "k--", label="Boulos et al. (2023)")
ax.legend()

ax = axs[1, 0]
ax.set_title(r"$\mathregular{H_2}$ plasma viscosity")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\mu [Pa.s]}$")
ax.plot(temperatures, viscosity, "k", label="minplascalc")
ax.plot(temperatures_ref, viscosity_ref, "k--", label="Boulos et al. (2023)")
ax.legend()

ax = axs[1, 1]
ax.set_title(r"$\mathregular{H_2}$ plasma thermal conductivity")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\kappa [W/(m.K)]}$")
ax.plot(temperatures, thermal_conductivity, "k", label="minplascalc")
ax.plot(temperatures_ref, thermal_conductivity_ref, "k--", label="Boulos et al. (2023)")
ax.legend()

ax = axs[1, 2]
ax.set_title(r"$\mathregular{H_2}$ plasma electrical conductivity")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\sigma [S/m]}$")
ax.plot(temperatures, electrical_conductivity, "k", label="minplascalc")
ax.plot(
    temperatures_ref, electrical_conductivity_ref, "k--", label="Boulos et al. (2023)"
)
ax.legend()


plt.tight_layout()

# %%
# Conclusion
# ----------
#
# The results obtained using minplascalc are comparable to other data for hydrogen plasmas in literature,
# for example [Boulos2023]_. In particular the position and size of the peaks in :math:`C_p`,
# which are caused by the highly nonlinear dissociation and first ionisation reactions
# of :math:`H_2` and :math:`O` respectively, are accurately captured.

# %%
