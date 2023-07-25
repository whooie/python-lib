"""
Physical constants. All quantities taken from NIST:
https://physics.nist.gov/cuu/Constants/index.html
"""

import math as m
import numpy as np
import scipy.special as sp
from periodictable import elements
import re
from itertools import product

# planck constant [kg m^2 s^-1]
h = 6.626070040e-34         # +/- 0 (exact)

# reduced planck constant [kg m^2 s^-1]
hbar = h/2/np.pi            # +/- 0 (exact)

# speed of light in vacuum [m s^-1]
c = 2.99792458e+8           # +/- 0 (exact)

# Avogadro's number
NA = 6.022140857e+23        # +/- 0 (exact)

# Boltzmann's constant [J K^-1]
kB = 1.38064852e-23         # +/- 0 (exact)

# electric permittivity in vacuum [F m^-1]
e0 = 8.8541878128e-12       # +/- 0.0000000013e-12

# magnetic permeability in vacuum [N A^-2]
u0 = 1.25663706212e-6       # +/- 0.00000000019e-6

# Newtonian gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11             # +/- 0.00015e-11

# gravitational acceleration near Earth's surface [m s^-2]
g = 9.80665                 # +/- 0 (exact)

# elementary charge [C]
e = 1.602176634e-19         # +/- 0 (exact)

# electron mass [kg]
me = 9.1093837015e-31       # +/- 0.0000000028e-31

# proton mass [kg]
mp = 1.67262192369e-27      # +/- 0.00000000051e-27

# neutron mass [kg]
mn = 1.67492749804e-27      # +/- 0.00000000095e-27

# unified atomic mass unit [kg]
mu = 1.66053906660e-27      # +/- 0.0000000005e-27

# Rydberg constant [m^-1]
Rinf = 10973731.568160   # +/- 0.000021

# fine structure constant
alpha = 7.2973525693e-3     # +/- 0.0000000011e-3

# molar gas constant
R = 8.314462618             # +/- 0 (exact)

# Stefan-Boltzmann constant
SB = (np.pi**2*kB**4)/(60*hbar**3*c**2) # +/- 0 (exact)

# Bohr radius [m]
a0 = 5.29177210903e-11      # +/- 0.00000000080e-11

# Bohr magneton [J T^-1]
uB = 9.2740100783e-24       # +/- 0.0000000028e-24

# nuclear magneton [J T^-1]
uN = 5.050783699e-27        # +/- 0.000000031e-27

# Hartree energy [J] = 2*Rinf*h*c
Eh = 4.3597447222071e-18    # +/- 0.0000000000085e-18

# unified atomic mass unit [kg] = 1/NA/1000
amu = 1.66053906660e-27     # +/- 0.00000000050e-27

amu2kg = amu
kg2amu = 1/amu2kg

a02m = a0
m2a0 = 1/a02m

cm2J = 100*h*c
J2cm = 1/cm2J

Hz2J = h
J2Hz = 1/Hz2J

K2J = kB
J2K = 1/K2J

Eh2J = Eh
J2Eh = 1/Eh2J

eV2J = e
J2eV = 1/eV2J

def rescale(X, u1, u2, inv=False) -> (float, np.ndarray):
    return (u1/u2)**(1 - 2*inv) * X

def hund_S(n, l) -> (float, np.ndarray):
    return (n >= 0)*(n <= 2*(2*l+1)) \
                * ((2*l+1)/2 - abs(n - (2*l+1))/2)

def hund_L(n, l) -> (float, np.ndarray):
    return (n >= 0)*(n <= 2*l+1) \
                * (-n*(n - (2*l+1))/2) \
            + (n > 2*l+1)*(n <= 2*(2*l+1)) \
                * (-(n - (2*l+1))*(n - 2*(2*l+1))/2)

def hund_J(n, l) -> (float, np.ndarray):
    return (n >= 0)*(n <= 2*l+1) \
                * abs(hund_L(n, l) - hund_S(n, l)) \
            + (n > 2*l+1)*(n <= 2*(2*l+1)) \
                * (hund_L(n, l) + hund_S(n, l))

def term_symbol(S, L, J) -> str:
    assert L >= 0 and L <= 23
    s = str(int(2*S+1))
    l = (L == 0)*"S" + (L == 1)*"P" + (L == 2)*"D" + (L == 3)*"F" \
            + (L >= 4)*chr(int(L) + 3)
    j = str(int(J)) if J%1 == 0 else str(int(2*J))+"/2"
    return s+l+j

def hund(n, l, as_term=False) -> (tuple, str):
    SLJ = (hund_S(n, l), hund_L(n, l), hund_J(n, l))
    return term_symbol(*SLJ) if as_term else SLJ

def hund_econf(econf: str, as_term=False) -> (tuple, str):
    orbitals = econf.split(" ")
    S = 0; L = 0; J = 0
    for oi in orbitals:
        match = re.match(r'([1-9])([spdf])([0-9]{,2})', oi)
        if match is None:
            raise Exception(f"Invalid orbital '{oi}'")
        n = int(match.group(3))
        l = {"s": 0, "p": 1, "d": 2, "f": 3}[match.group(2)]
        assert n <= 2*(2*l+1)
        S += hund_S(n, l)
        L += hund_L(n, l)
        J += hund_J(n, l)
    return term_symbol(S, L, J) if as_term else (S, L, J)

def molecule_mass(M: str, as_amu=False) -> float:
    atoms = M.split(" ")
    m = 0
    for ai in atoms:
        match = re.match(r'([0-9]*)([A-Z][a-z]?)([0-9]*)', ai)
        if match is None:
            raise Exception(f"Invalid atom '{ai}'")
        element = elements.isotope(match.group(2))
        if match.group(1) == "":
            mi = element.mass
        else:
            mi = element[int(match.group(1))].mass
        m += mi*(int(match.group(3)) if match.group(3) != "" else 1)
    return m if as_amu else m*amu2kg

def hydrogen_R(n, l, r) -> (float, np.ndarray):
    assert n >= 1
    assert l >= 0
    assert abs(l) <= n-1
    return np.sqrt((2/n/a0)**3 * m.factorial(n-l-1)/(2*n*m.factorial(n+l))) \
            * sp.assoc_laguerre(2*r/a0/n, n-l-1, 2*l+1) \
            * (2*r/a0/n)**l * np.exp(-r/a0/n)

def hydrogen_Y(l, m, theta, phi) -> (float, np.ndarray):
    assert l >= 0
    assert abs(m) <= l
    Y = sp.sph_harm(m, l, phi, theta)
    return Y

def hydrogen_wf(n, l, m, r, theta, phi) -> (float, np.ndarray):
    return hydrogen_Y(l, m, theta, phi)*hydrogen_R(n, l, r)

def qho_wf(n, m, w, x) -> (float, np.ndarray):
    return 1/np.sqrt(2**n * m.factorial(n)) \
            * ((m*w)/(np.pi*hbar))**(1/4) \
            * sp.eval_hermite(n, np.sqrt(m*w/hbar)*x) \
            * np.exp(-(m*w)/(2*hbar)*x**2)

def q_raise(J, mj) -> (float, np.ndarray):
    assert J >= 0
    return np.sqrt((J - mj)*(J + mj + 1)) if mj >= -J and mj < J else 0.0

def q_lower(J, mj) -> (float, np.ndarray):
    assert J >= 0
    return np.sqrt((J + mj)*(J - mj + 1)) if mj > -J and mj <= J else 0.0

def lande_gJ(S, L, J, gS=2.0, gL=1.0) -> float:
    return (
        gL * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1))
        + gS * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
    ) if J != 0.0 else 0.0

def lande_gF(S, L, J, I, F, gS=2.0, gL=1.0, gJ=None, gI=5.585) -> float:
    _gJ = lande_gJ(S, L, J, gS, gL) if gJ is None else gJ
    return (
        _gJ * (F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1))
        + gI * uN / uB * (F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))
    ) if F != 0.0 else 0.0

def contract(expr, D=4, zerostart=True):
    z = 0 if zerostart else 1
    indices = product(
        *[list(range(z, z + D)) for k in range(expr.__code__.co_argcount)])
    return sum(expr(*a) for a in indices)

def nm2rgb(wavelength: float) -> tuple[int, int, int]:
    gamma = 0.80
    intensity_max = 255
    if wavelength >= 380.0 and wavelength < 440.0:
        red = -(wavelength - 440.0) / (440.0 - 380.0)
        green = 0.0
        blue = 1.0
    elif wavelength >= 440.0 and wavelength < 490.0:
        red = 0.0
        green = (wavelength - 440.0) / (490.0 - 440.0)
        blue = 1.0
    elif wavelength >= 490.0 and wavelength < 510.0:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510.0) / (510.0 - 490.0)
    elif wavelength >= 510.0 and wavelength < 580.0:
        red = (wavelength - 510.0) / (580.0 - 510.0)
        green = 1.0
        blue = 0.0
    elif wavelength >= 580.0 and wavelength < 645.0:
        red = 1.0
        green = -(wavelength - 645.0) / (645.0 - 580.0)
        blue = 0.0
    elif wavelength >= 645.0 and wavelength < 781.0:
        red = 1.0
        green = 0.0
        blue = 0.0
    else:
        red = 0.0
        green = 0.0
        blue = 0.0

    # intensity drop near vision limits
    if wavelength >= 380.0 and wavelength < 420.0:
        factor = 0.3 + 0.7 * (wavelength - 380.0) / (420.0 - 380.0)
    elif wavelength >= 420.0 and wavelength < 701.0:
        factor = 1.0
    elif wavelength >= 701.0 and wavelength < 781.0:
        factor = 0.3 + 0.7 * (781.0 - wavelength) / (781.0 - 701.0)
    else: factor = 0.0

    red = int(round(intensity_max * pow(red * factor, gamma)))
    green = int(round(intensity_max * pow(green * factor, gamma)))
    blue = int(round(intensity_max * pow(blue * factor, gamma)))
    return (red, green, blue)

def _rgb2hex(rgb: tuple[int, int, int]) -> str:
    hex_vals = [("0" if len(h := hex(c)[2:]) < 2 else "") + h for c in rgb]
    return "#" + "".join(hex_vals)

def nm2hex(wavelength: float) -> str:
    return _rgb2hex(nm2rgb(wavelength))

