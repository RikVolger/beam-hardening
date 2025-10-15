import numpy as np
from scipy import integrate as scint
from matplotlib import pyplot as plt
from pathlib import Path


# define dIdx function
def make_dIdx(attenuation1, attenuation2=None, holdup_profile=None):
    def dIdx(t, intensity):
        if holdup_profile is not None:
            assert attenuation2 is not None
            if isinstance(holdup_profile, float):
                holdup = holdup_profile
            else:
                holdup = np.interp(t, holdup_profile[:, 0], holdup_profile[:, 1])
            attenuation = holdup * attenuation1 + (1 - holdup) * attenuation2
        else:
            attenuation = attenuation1
        return -intensity * attenuation
    return dIdx


# Energy spectrum ends
E_min = 18      # keV
E_max = 120     # keV
# Water column thickness (pathlength)
D = 20          # cm

E_range = np.arange(E_min, E_max + 1)

XCOM_folder = Path("C:/Users/rikvolger/Documents/Codebase/"
                   "ASTRA-reconstructions/attenuation/XCOM_output")

# load energy spectrum (I0)
energy_spectrum = np.genfromtxt("./data/TASMIP-spectrum_120kV_3.8mm-Al.csv",
                                delimiter=",",
                                skip_header=1)

# Interpolate to nice 1 kV wells (18-120 kV)
intensity_0 = np.interp(E_range,
                        energy_spectrum[:, 0],
                        energy_spectrum[:, 1])

fig, ax = plt.subplots(1, 1)
ax.plot(energy_spectrum[:, 0], energy_spectrum[:, 1], label="original")
ax.plot(E_range,
        intensity_0,
        label="interpolated",
        linestyle="--")
ax.set_title("X-ray energy spectrum")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("X-ray flux $[photons / cm2 / mA / s / keV]$")
ax.legend()

fig.savefig("./img/png/01_energy-spectrum.png", dpi=300)
fig.savefig("./img/pdf/01_energy-spectrum.pdf", dpi=900)

# load attenuation coefficients (water, c071) (MeV)
attenuation_water = np.genfromtxt(
    XCOM_folder / "data_exp-001.csv",
    delimiter="\t",
    skip_header=1
)
density_water = 1.0     # g/cm3
# Interpolate to nice 1 kV wells (18-120 kV)
attenuation_water = np.interp(E_range,
                              attenuation_water[:, 0] * 1e3,
                              attenuation_water[:, 2]) * density_water

attenuation_salt = np.genfromtxt(
    XCOM_folder / "data_exp-075.csv",
    delimiter="\t",
    skip_header=1
)
density_salt = 1.02
# Interpolate to nice 1 kV wells (18-120 kV)
attenuation_salt = np.interp(E_range,
                             attenuation_salt[:, 0] * 1e3,
                             attenuation_salt[:, 2]) * density_water

attenuation_air = np.genfromtxt(
    "./data/attenuation_air.csv",
    delimiter=",",
    skip_header=1
)

density_air = 1.2e-3    # g / cm3

attenuation_air = np.interp(E_range,
                            attenuation_air[0] * 1e3,
                            attenuation_air[1]) * density_air

# Estimation of linear attenuation coefficient for water.
# Assumes energy-agnostic attenuation.
constant_attenuation = np.ones((E_max - E_min + 1)) * .15

# The case for single medium
titles = ["Air", "Water", "Salty", "Constant"]
linestyles = ["-", "-", "--", ":"]
attenuations = [
    attenuation_air,
    attenuation_water,
    attenuation_salt,
    constant_attenuation
]

fig = plt.figure(figsize=(4, 2.5), layout="tight")
ax = fig.subplots(1, 1)

att_fig = plt.figure(figsize=(4, 2.5), layout="tight")
ax_att = att_fig.subplots(1, 1)

for i in range(len(titles)):
    # solve ivp for each attenuation coefficient
    intensity = scint.solve_ivp(make_dIdx(attenuations[i]),
                                t_span=(0, D),
                                y0=intensity_0,
                                t_eval=np.linspace(0, D, 100))

    total_flux = intensity.y.sum(axis=0)

    print(titles[i])
    print("=" * 40)
    print(f"Initial flux:\t{total_flux[0]:.5g}")
    print(f"Final flux:\t{total_flux[-1]:.5g}\n")
    # do some plotting
    ax.plot(intensity.t, total_flux, label=titles[i], linestyle=linestyles[i])
    ax_att.plot(E_range, attenuations[i], label=titles[i])
ax.set_title("X-ray intensity")
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Distance (cm)")
ax.legend()

ax_att.set_title("Specific attenuations")
ax_att.set_ylabel("$\mu$ ($cm^{-1}$)")
ax_att.set_xlabel("Energy (keV)")
ax_att.legend()

att_fig.savefig("./img/png/02_attenuation-profile.png", dpi=300)
att_fig.savefig("./img/pdf/02_attenuation-profile.pdf", dpi=900)

fig.savefig("./img/png/03_1D-intensity.png", dpi=300)
fig.savefig("./img/pdf/03_1D-intensity.pdf", dpi=900)

d_vec = np.linspace(0, D, 100)
holdup = np.ones(d_vec.shape) * .3
holdup = np.vstack((d_vec, holdup))

titles = ["Water", "Constant"]
attenuations = [
    attenuation_water,
    constant_attenuation
]

fig = plt.figure(figsize=(4, 2.5), layout="tight")
ax = fig.subplots(1, 1)

for i in range(len(titles)):
    # solve ivp for each attenuation coefficient
    intensity = scint.solve_ivp(make_dIdx(attenuations[i], attenuation_air, holdup),
                                t_span=(0, D),
                                y0=intensity_0,
                                t_eval=np.linspace(0, D, 100))

    total_flux = intensity.y.sum(axis=0)

    print(titles[i])
    print("=" * 40)
    print(f"Initial flux:\t{total_flux[0]:.5g}")
    print(f"Final flux:\t{total_flux[-1]:.5g}\n")
    # do some plotting
    ax.plot(intensity.t, total_flux, label=titles[i], linestyle=linestyles[i])
ax.set_title("X-ray intensity ($\\varepsilon = 30\%$)")
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Distance (cm)")
ax.legend()

fig.savefig("./img/png/04_intensity_holdup-.30.png", dpi=300)
fig.savefig("./img/pdf/04_intensity_holdup-.30.pdf", dpi=900)

# at different constant holdups, calculate signal after 20 cm for constant / water
# Calculate I_full (0% holdup) and I_empty (100% holdup)
# Use our standard procedure to calculate holdups for those
int_full = scint.solve_ivp(make_dIdx(attenuation_water, attenuation_air, 0.),
                           t_span=(0, D),
                           y0=intensity_0,
                           t_eval=np.linspace(0, D, 100))
I_full = int_full.y.sum(axis=0)[-1]

int_empty = scint.solve_ivp(make_dIdx(attenuation_water, attenuation_air, 1.),
                            t_span=(0, D),
                            y0=intensity_0,
                            t_eval=np.linspace(0, D, 100))
I_empty = int_empty.y.sum(axis=0)[-1]

holdup_range = np.linspace(0, 1, 100)
holdup_calc = np.zeros_like(holdup_range)

for i, holdup in enumerate(holdup_range):
    # calculate intensity with attenuation water
    intensity = scint.solve_ivp(make_dIdx(attenuation_water, attenuation_air, holdup),
                                t_span=(0, D),
                                y0=intensity_0,
                                t_eval=np.linspace(0, D, 100))
    I_eval = intensity.y.sum(axis=0)[-1]
    holdup_calc[i] = np.log(I_eval / I_full) / np.log(I_empty / I_full)

fig = plt.figure(figsize=(4, 2.5), layout="tight")
ax = fig.subplots(1, 1)

ax.plot(holdup_range, holdup_range, label="Original")
ax.plot(holdup_range, holdup_calc, label="Recalculated")
ax.set_ylabel("Holdup (recalculated)")
ax.set_xlabel("Holdup (original)")
ax.legend()

fig.savefig("./img/png/05_holdups.png", dpi=300)
fig.savefig("./img/pdf/05_holdups.pdf", dpi=900)

D_range = np.linspace(1, D, 100)
holdup_calc = np.zeros_like(D_range)
holdup = .3

for i, D in enumerate(D_range):
    int_full = scint.solve_ivp(make_dIdx(attenuation_water, attenuation_air, 0.),
                               t_span=(0, D),
                               y0=intensity_0)
    I_full = int_full.y.sum(axis=0)[-1]

    int_empty = scint.solve_ivp(make_dIdx(attenuation_water, attenuation_air, 1.),
                                t_span=(0, D),
                                y0=intensity_0)
    I_empty = int_empty.y.sum(axis=0)[-1]
    # calculate intensity with attenuation water
    intensity = scint.solve_ivp(make_dIdx(attenuation_water, attenuation_air, holdup),
                                t_span=(0, D),
                                y0=intensity_0)
    I_eval = intensity.y.sum(axis=0)[-1]
    holdup_calc[i] = np.log(I_eval / I_full) / np.log(I_empty / I_full)

fig = plt.figure(figsize=(4, 2.5), layout="tight")
ax = fig.subplots(1, 1)

ax.plot([D_range[0], D_range[-1]], [holdup, holdup], label="Original")
ax.plot(D_range, holdup_calc, label="Recalculated")
ax.set_ylabel("Holdup")
ax.set_xlabel("Pathlength")
ax.legend()

fig.savefig("./img/png/06_recalculated-holdups.png", dpi=300)
fig.savefig("./img/pdf/06_recalculated-holdups.pdf", dpi=900)

plt.show()
