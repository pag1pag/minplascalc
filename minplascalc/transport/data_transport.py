r"""Fitting parameters for the classical transport collision integrals.

This module contains the fitting parameters for the classical transport
collision integrals of neutral-neutral and neutral-ion interactions.

The data is taken from [Laricchiuta2007]_.

Notes
-----
The fitting parameters are :math:`c_j`.
They are used in in eq. (16) of [Laricchiuta2007]_ to compute the polynomial
functions :math:`a_i(\beta)`.

.. math::

    a_i(\beta)=\sum_{j=0}^2 c_j \beta^j

where :math:`\beta` is a parameter to estimate the hardness of interacting electronic
distribution densities, and it is estimated in eq. 5 of [Laricchiuta2007]_.

Then, the reduced collision integral \Omega^{(\ell, s) \star} is computed, using
eq. 16 of [Laricchiuta2007]_, as:

.. math::

    \begin{aligned}
        \ln \Omega^{(\ell, s) \star} =
        & {\left[a_1(\beta)+a_2(\beta) x\right]
            \frac{e^{\left(x-a_3(\beta)\right) / a_4(\beta)}}
            {e^{\left(x-a_3(\beta)\right) / a_4(\beta)}+e^{\left(a_3(\beta)-x\right) / a_4(\beta)}} } \\
        & +a_5(\beta)
            \frac{e^{\left(x-a_6(\beta)\right) / a_7(\beta)}}
            {e^{\left(x-a_6(\beta)\right) / a_7(\beta)}+e^{\left(a_6(\beta)-x\right) / a_7(\beta)}}
    \end{aligned}

where :math:`x=\ln T^{\star}`

The reduced temperature is defined as :math:`T^{\star}=\frac{k_b T}{\epsilon}` in eq. 12
of [Laricchiuta2007]_, where :math:`\epsilon` is the binding energy, defined in eq. 7 of
[Laricchiuta2007]_.
"""

import numpy as np

#############################################################################
#############################################################################
#############################################################################
# Table 2 of [Laricchiuta2007]_
# "Fitting parameters, entering in Eq. (16), of classical transport collision
# integrals \Omega^{(l,s), \star} for neutral–neutral interactions (m = 6)."

c0_nn_11 = [
    7.884756e-1,
    -2.952759e-1,
    5.020892e-1,
    -9.042460e-1,
    -3.373058,
    4.161981,
    2.462523,
]
c1_nn_11 = [
    -2.438494e-2,
    -1.744149e-3,
    4.316985e-2,
    -4.017103e-2,
    2.458538e-1,
    2.202737e-1,
    3.231308e-1,
]
c2_nn_11 = [0, 0, 0, 0, -4.850047e-3, -1.718010e-2, -2.281072e-2]
c0_nn_12 = [
    7.123565e-1,
    -2.910530e-1,
    4.187065e-2,
    -9.287685e-1,
    -3.598542,
    3.934824,
    2.578084,
]
c1_nn_12 = [
    -2.688875e-2,
    -2.065175e-3,
    4.060236e-2,
    -2.342270e-2,
    2.545120e-1,
    2.699944e-1,
    3.449024e-1,
]
c2_nn_12 = [0, 0, 0, 0, -4.685966e-3, -2.009886e-2, -2.292710e-2]
c0_nn_13 = [
    6.606022e-1,
    -2.870900e-1,
    -2.519690e-1,
    -9.173046e-1,
    -3.776812,
    3.768103,
    2.695440,
]
c1_nn_13 = [
    -2.831448e-2,
    -2.232827e-3,
    3.778211e-2,
    -1.864476e-2,
    2.552528e-1,
    3.155025e-1,
    3.597998e-1,
]
c2_nn_13 = [0, 0, 0, 0, -4.237220e-3, -2.218849e-2, -2.267102e-2]
c0_nn_14 = [
    6.268016e-1,
    -2.830834e-1,
    -4.559927e-1,
    -9.334638e-1,
    -3.947019,
    3.629926,
    2.824905,
]
c1_nn_14 = [
    -2.945078e-2,
    -2.361273e-3,
    3.705640e-2,
    -1.797329e-2,
    2.446843e-1,
    3.761272e-1,
    3.781709e-1,
]
c2_nn_14 = [0, 0, 0, 0, -3.176374e-3, -2.451016e-2, -2.251978e-2]
c0_nn_15 = [
    5.956859e-1,
    -2.804989e-1,
    -5.965551e-1,
    -8.946001e-1,
    -4.076798,
    3.458362,
    2.982260,
]
c1_nn_15 = [
    -2.915893e-2,
    -2.298968e-3,
    3.724395e-2,
    -2.550731e-2,
    1.983892e-1,
    4.770695e-1,
    4.014572e-1,
]
c2_nn_15 = [0, 0, 0, 0, -5.014065e-4, -2.678054e-2, -2.142580e-2]
c0_nn_22 = [
    7.898524e-1,
    -2.998325e-1,
    7.077103e-1,
    -8.946857e-1,
    -2.958969,
    4.348412,
    2.205440,
]
c1_nn_22 = [
    -2.114115e-2,
    -1.243977e-3,
    3.583907e-2,
    -2.473947e-2,
    2.303358e-1,
    1.920321e-1,
    2.567027e-1,
]
c2_nn_22 = [0, 0, 0, 0, -5.226562e-3, -1.496557e-2, -1.861359e-2]
c0_nn_23 = [
    7.269006e-1,
    -2.972304e-1,
    3.904230e-1,
    -9.442201e-1,
    -3.137828,
    4.190370,
    2.319751,
]
c1_nn_23 = [
    -2.233866e-2,
    -1.392888e-3,
    3.231655e-2,
    -1.494805e-2,
    2.347767e-1,
    2.346004e-1,
    2.700236e-1,
]
c2_nn_23 = [0, 0, 0, 0, -4.963979e-3, -1.718963e-2, -1.854217e-2]
c0_nn_24 = [
    6.829159e-1,
    -2.943232e-1,
    1.414623e-1,
    -9.720228e-1,
    -3.284219,
    4.011692,
    2.401249,
]
c1_nn_24 = [
    -2.332763e-2,
    -1.514322e-3,
    3.075351e-2,
    -1.038869e-2,
    2.243767e-1,
    3.005083e-1,
    2.943600e-1,
]
c2_nn_24 = [0, 0, 0, 0, -3.913041e-3, -2.012373e-2, -1.884503e-2]
c0_nn_33 = [
    7.468781e-1,
    -2.947438e-1,
    2.234096e-1,
    -9.974591e-1,
    -3.381787,
    4.094540,
    2.476087,
]
c1_nn_33 = [
    -2.518134e-2,
    -1.811571e-3,
    3.681114e-2,
    -2.670805e-2,
    2.372932e-1,
    2.756466e-1,
    3.300898e-1,
]
c2_nn_33 = [0, 0, 0, 0, -4.239629e-3, -2.009227e-2, -2.223317e-2]
c0_nn_44 = [
    7.365470e-1,
    -2.968650e-1,
    3.747555e-1,
    -9.944036e-1,
    -3.136655,
    4.145871,
    2.315532,
]
c1_nn_44 = [
    -2.242357e-2,
    -1.396696e-3,
    2.847063e-2,
    -1.378926e-2,
    2.176409e-1,
    2.855836e-1,
    2.842981e-1,
]
c2_nn_44 = [0, 0, 0, 0, -3.899247e-3, -1.939452e-2, -1.874462e-2]
c_nn_11 = np.array([c0_nn_11, c1_nn_11, c2_nn_11], dtype=np.float64).transpose()
c_nn_12 = np.array([c0_nn_12, c1_nn_12, c2_nn_12], dtype=np.float64).transpose()
c_nn_13 = np.array([c0_nn_13, c1_nn_13, c2_nn_13], dtype=np.float64).transpose()
c_nn_14 = np.array([c0_nn_14, c1_nn_14, c2_nn_14], dtype=np.float64).transpose()
c_nn_15 = np.array([c0_nn_15, c1_nn_15, c2_nn_15], dtype=np.float64).transpose()
c_nn_22 = np.array([c0_nn_22, c1_nn_22, c2_nn_22], dtype=np.float64).transpose()
c_nn_23 = np.array([c0_nn_23, c1_nn_23, c2_nn_23], dtype=np.float64).transpose()
c_nn_24 = np.array([c0_nn_24, c1_nn_24, c2_nn_24], dtype=np.float64).transpose()
c_nn_33 = np.array([c0_nn_33, c1_nn_33, c2_nn_33], dtype=np.float64).transpose()
c_nn_44 = np.array([c0_nn_44, c1_nn_44, c2_nn_44], dtype=np.float64).transpose()
fill_nan_c_nn = np.full(c_nn_11.shape, np.nan, dtype=np.float64)

c_nn = np.array(
    [
        [c_nn_11, c_nn_12, c_nn_13, c_nn_14, c_nn_15],
        [fill_nan_c_nn, c_nn_22, c_nn_23, c_nn_24, fill_nan_c_nn],
        [fill_nan_c_nn, fill_nan_c_nn, c_nn_33, fill_nan_c_nn, fill_nan_c_nn],
        [fill_nan_c_nn, fill_nan_c_nn, fill_nan_c_nn, c_nn_44, fill_nan_c_nn],
    ],
    dtype=np.float64,
)
"""Parameters for the classical transport collision integrals for neutral-neutral interactions."""


#############################################################################
#############################################################################
#############################################################################
# Table 1 of [Laricchiuta2007]_
# "Fitting parameters, entering in Eq. (16), of classical transport collision
# integrals \Omega^{(l,s), \star} for neutral–ion interactions (m = 4)."

c0_in_11 = [
    9.851755e-1,
    -4.737800e-1,
    7.080799e-1,
    -1.239439,
    -4.638467,
    3.841835,
    2.317342,
]
c1_in_11 = [
    -2.870704e-2,
    -1.370344e-3,
    4.575312e-3,
    -3.683605e-2,
    4.418904e-1,
    3.277658e-1,
    3.912768e-1,
]
c2_in_11 = [0, 0, 0, 0, -1.220292e-2, -2.660275e-2, -3.136223e-2]
c0_in_12 = [
    8.361751e-1,
    -4.707355e-1,
    1.771157e-1,
    -1.094937,
    -4.976384,
    3.645873,
    2.428864,
]
c1_in_12 = [
    -3.201292e-2,
    -1.783284e-3,
    1.172773e-2,
    -3.115598e-2,
    4.708074e-1,
    3.699452e-1,
    4.267351e-1,
]
c2_in_12 = [0, 0, 0, 0, -1.283818e-2, -2.988684e-2, -3.278874e-2]
c0_in_13 = [
    7.440562e-1,
    -4.656306e-1,
    -1.465752e-1,
    -1.080410,
    -5.233907,
    3.489814,
    2.529678,
]
c1_in_13 = [
    -3.453851e-2,
    -2.097901e-3,
    1.446209e-2,
    -2.712029e-2,
    4.846691e-1,
    4.140270e-1,
    4.515088e-1,
]
c2_in_13 = [0, 0, 0, 0, -1.280346e-2, -3.250138e-2, -3.339293e-2]
c0_in_14 = [
    6.684360e-1,
    -4.622014e-1,
    -3.464990e-1,
    -1.054374,
    -5.465789,
    3.374614,
    2.648622,
]
c1_in_14 = [
    -3.515695e-2,
    -2.135808e-3,
    1.336362e-2,
    -3.149321e-2,
    4.888443e-1,
    4.602468e-1,
    4.677409e-1,
]
c2_in_14 = [0, 0, 0, 0, -1.228090e-2, -3.463073e-2, -3.339297e-2]
c0_in_15 = [
    6.299083e-1,
    -4.560921e-1,
    -5.228598e-1,
    -1.124725,
    -5.687354,
    3.267709,
    2.784725,
]
c1_in_15 = [
    -3.720000e-2,
    -2.395779e-3,
    1.594610e-2,
    -2.862354e-2,
    4.714789e-1,
    5.281419e-1,
    4.840700e-1,
]
c2_in_15 = [0, 0, 0, 0, -1.056602e-2, -3.678869e-2, -3.265127e-2]
c0_in_22 = [
    9.124518e-1,
    -4.697184e-1,
    1.031053,
    -1.090782,
    -4.127243,
    4.059078,
    2.086906,
]
c1_in_22 = [
    -2.398461e-2,
    -7.809681e-4,
    4.069668e-3,
    -2.413508e-2,
    4.302667e-1,
    2.597379e-1,
    2.920310e-1,
]
c2_in_22 = [0, 0, 0, 0, -1.352874e-2, -2.169951e-2, -2.560437e-2]
c0_in_23 = [
    8.073459e-1,
    -4.663682e-1,
    6.256342e-1,
    -1.063437,
    -4.365989,
    3.854346,
    2.146207,
]
c1_in_23 = [
    -2.581232e-2,
    -1.030271e-3,
    4.086881e-3,
    -1.235489e-2,
    4.391454e-1,
    3.219224e-1,
    3.325620e-1,
]
c2_in_23 = [0, 0, 0, 0, -1.314615e-2, -2.587493e-2, -2.686959e-2]
c0_in_24 = [
    7.324117e-1,
    -4.625614e-1,
    3.315871e-1,
    -1.055706,
    -4.571022,
    3.686006,
    2.217893,
]
c1_in_24 = [
    -2.727580e-2,
    -1.224292e-3,
    7.216776e-3,
    -8.585500e-3,
    4.373660e-1,
    3.854493e-1,
    3.641196e-1,
]
c2_in_24 = [0, 0, 0, 0, -1.221457e-2, -2.937568e-2, -2.763824e-2]
c0_in_33 = [
    8.402943e-1,
    -4.727437e-1,
    4.724228e-1,
    -1.213660,
    -4.655574,
    3.817178,
    2.313186,
]
c1_in_33 = [
    -2.851694e-2,
    -1.328784e-3,
    7.706027e-3,
    -3.456656e-2,
    4.467685e-1,
    3.503180e-1,
    3.889828e-1,
]
c2_in_33 = [0, 0, 0, 0, -1.237864e-2, -2.806506e-2, -3.120619e-2]
c0_in_44 = [
    8.088842e-1,
    -4.659483e-1,
    6.092981e-1,
    -1.113323,
    -4.349145,
    3.828467,
    2.138075,
]
c1_in_44 = [
    -2.592379e-2,
    -1.041599e-3,
    1.428402e-3,
    -1.031574e-2,
    4.236246e-1,
    3.573461e-1,
    3.388072e-1,
]
c2_in_44 = [0, 0, 0, 0, -1.210668e-2, -2.759622e-2, -2.669344e-2]
c_in_11 = np.array([c0_in_11, c1_in_11, c2_in_11], dtype=np.float64).transpose()
c_in_12 = np.array([c0_in_12, c1_in_12, c2_in_12], dtype=np.float64).transpose()
c_in_13 = np.array([c0_in_13, c1_in_13, c2_in_13], dtype=np.float64).transpose()
c_in_14 = np.array([c0_in_14, c1_in_14, c2_in_14], dtype=np.float64).transpose()
c_in_15 = np.array([c0_in_15, c1_in_15, c2_in_15], dtype=np.float64).transpose()
c_in_22 = np.array([c0_in_22, c1_in_22, c2_in_22], dtype=np.float64).transpose()
c_in_23 = np.array([c0_in_23, c1_in_23, c2_in_23], dtype=np.float64).transpose()
c_in_24 = np.array([c0_in_24, c1_in_24, c2_in_24], dtype=np.float64).transpose()
c_in_33 = np.array([c0_in_33, c1_in_33, c2_in_33], dtype=np.float64).transpose()
c_in_44 = np.array([c0_in_44, c1_in_44, c2_in_44], dtype=np.float64).transpose()
fill_nan_c_in = np.full(c_in_11.shape, np.nan, dtype=np.float64)
c_in = np.array(
    [
        [c_in_11, c_in_12, c_in_13, c_in_14, c_in_15],
        [fill_nan_c_in, c_in_22, c_in_23, c_in_24, fill_nan_c_in],
        [fill_nan_c_in, fill_nan_c_in, c_in_33, fill_nan_c_in, fill_nan_c_in],
        [fill_nan_c_in, fill_nan_c_in, fill_nan_c_in, c_in_44, fill_nan_c_in],
    ],
    dtype=np.float64,
)
"""Parameters for the classical transport collision integrals for neutral-ion interactions."""
