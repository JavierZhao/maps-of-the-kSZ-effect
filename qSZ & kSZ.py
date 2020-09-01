import numpy as np
import healpy as hp
import scipy as sp
from scipy.special import comb
np.random.seed(0)

import astropaint as ap
from astropaint import Catalog, Canvas, Painter
from astropaint import utils, transform
from astropaint.profiles import NFW

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from astropy import units as u
from astropy.constants import sigma_T, m_p
from astropy.cosmology import Planck18_arXiv_v2 as cosmo

from astropaint.lib.utils import interpolate, LOS_integrate
from astropaint.lib import transform

from astropaint.profiles import Battaglia16

# ---------------Caching----------------
# To cache templates use
#  the @memory.cache decorator
from joblib import Memory
cachedir = 'cache'
memory = Memory(cachedir, verbose=False)

# ---------------Constants---------------
sigma_T = sigma_T.to(u.Mpc**2).value # [Mpc^2]
m_p = m_p.to(u.M_sun).value # [M_sun]
f_b = cosmo.Ob0/cosmo.Om0
c = 299792. #km/s
h = cosmo.h
T_cmb = 2.7251
Gcm2 = 4.785E-20 # G/c^2 (Mpc/M_sun)

# ---------------Mathematical utility functions---------------
def cot(x):
    return pow(np.tan(x), -1)

def coth(x):
    return pow(np.tanh(x), -1)

def sYlm(s, l, m, theta, phi):
    """
    returns the sYlm coefficients calculated based on the Goldberg, et al., formula from Wikipedia:
    https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#:~:text=Spin%2Dweighted%20harmonics,of%20spin%2Dweight%20s%20functions.
    """
    fact1 = np.math.factorial(l + m)
    fact2 = np.math.factorial(l - m)
    fact3 = np.math.factorial(l + s)
    fact4 = np.math.factorial(l - s)

    result = pow(-1, m)
    result *= np.sqrt((fact1 * fact2 * (2 * l + 1)) / (4 * np.pi * fact3 * fact4))
    result *= pow(np.sin(theta / 2), 2 * l)
    summation = sum([comb(l - s, r) * comb(l + s, r + s - m) * pow(-1, l - r - s) *
                     np.exp(complex(0, m * phi)) * pow(cot(theta / 2), 2 * r + s - m) for r in range(l - s + 1)])
    result *= summation
    return result

def rho_2D_bartlemann(R, rho_s, R_s):
    """
    projected NFW mass profile
    Eq. 7 in Bartlemann 1996: https://arxiv.org/abs/astro-ph/9602053

    Returns
    -------
    surface mass density: [M_sun/Mpc^2]
    """

    x = np.asarray(R / R_s, dtype=np.complex)
    f = 1 - 2 / np.sqrt(1 - x ** 2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
    f = f.real
    f = np.true_divide(f, x ** 2 - 1)
    Sigma = 8 * rho_s * R_s * f
    return Sigma

def tau_2D(R, rho_s, R_s):
    """
    projected NFW tau profile
    Eq. 7 in Battaglia 2016 :

    Returns
    -------
    tau: [NA]
    """
    X_H = 0.76
    x_e = (X_H + 1) / 2 * X_H
    f_s = 0.02
    mu = 4 / (2 * X_H + 1 + X_H * x_e)

    Sigma = rho_2D_bartlemann(R, rho_s, R_s)
    tau = sigma_T * x_e * X_H * (1 - f_s) * f_b * Sigma / mu / m_p
    return tau

# ---------------Setting up the catalog---------------
catalog = Catalog('WebSky_lite')

catalog.cut_M_200c(mass_min=5E14)
# print(catalog.size)

catalog.cut_redshift(redshift_max=0.5)
# print(catalog.size)

nside = 512 #use this for tests
#nside = 4096 (final simulation)
canvas = Canvas(catalog, nside, R_times=4)

# ---------------qSZ polarization---------------
def qSZ_pol(R, rho_s, R_s, v_r, theta, phi, niu=143):
    """
    Q±iU profile for non moving clusters
    Units: K
    143 GHz and 217 GHz and 315 GHz
    """
    # frequency dependence
    T0 = 2.72548
    h = 6.626 * pow(10, -34)
    KB = 1.3806 * pow(10, -23)
    x = (pow(10, 9) * h * niu) / (KB * T0)  # dimensionless frequency

    F = pow(x, 4) * np.exp(x) / pow(np.exp(x) - 1, 2)  # Blackbody spectrum

    # a2m coefficients from Planck 2018 data: https://pla.esac.esa.int/#maps
    a_2_m_list = [-1.2315e-05 + 1.5049e-05j,
                  -(-2.6998e-06 - 9.1367e-06j),
                  1.0702e-05 + 0j,
                  -2.6998e-06 + 9.1367e-06j,
                  -1.2315e-05 - 1.5049e-05j]

    s_w_Y_2_m_list = [sYlm(2, 2, m, theta, phi) for m in [-2, -1, 0, 1, 2]]
    summation = sum([a_2_m_list[i] * s_w_Y_2_m_list[i] for i in range(5)])

    tau = tau_2D(R, rho_s, R_s)

    pol = -(np.sqrt(6) / 10) * tau * summation * F

    result = abs(pol)
    #     result = pol.imag
    #     result = pol.real
    #     result = pol

    return result

# ---------------plot qSZ polarization---------------
painter = Painter(qSZ_pol)
canvas.clean()
painter.spray(canvas, cache = True)

canvas.show_map("cartesian", lonra=[0,20], latra=[0,20], min=0,max=9E-10, title = "qSZ @217 GHz")
plt.savefig('qSZ @217 GHz.png')

# ---------------kSZ polarization---------------
def kSZ_pol(R, rho_s, R_s, v_r, v_th, v_ph, niu=315):
    """
    Q+iU profile for moving clusters
    uses the convention c = 1
    input unit of frequency niu: GHz
    Units: K
    143 GHz and 217 GHz and 315 GHz
    """
    T0 = 2.72548
    h = 6.626 * pow(10, -34)
    KB = 1.3806 * pow(10, -23)

    # dimensionless frequency
    x = (pow(10, 9) * h * niu) / (KB * T0)

    f = (x / 2) * coth(x / 2)  # frequency dependence f

    F = pow(x, 4) * np.exp(x) / pow(np.exp(x) - 1, 2)  # Blackbody spectrum

    tau = tau_2D(R, rho_s, R_s)

    const = 2 * pow(KB, 3) * pow(T0, 4) * pow(h, -2)

    pol = -(tau / 10) * const * f * F * complex(v_th, -v_ph) * complex(v_th, -v_ph)
    # need to divided by c^2 (v_th and v_ph are in km/s)
    pol = pol / pow(c,2)

    result = abs(pol)
    #     result = pol.real  #Q
    #     result = pol.imag  #U
    return result

# ---------------plot kSZ polarization---------------
painter = Painter(kSZ_pol)
canvas.clean()
painter.spray(canvas, cache = True)

canvas.show_map("cartesian", lonra=[0,20], latra=[0,20], min=0,max=15, title = "kSZ @ 315 GHz")
plt.savefig('kSZ @315 GHz.png')