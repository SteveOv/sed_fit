"""
Low level utility functions for light curve ingest, pre-processing, estimation and fitting.
"""
#pylint: disable=no-member, invalid-name
from typing import Union
import warnings
import re

import numpy as np
from uncertainties import UFloat, ufloat

from deblib.vmath import arccos, arcsin, degrees, log10

_TRIG_MIN = ufloat(-1, 0)
_TRIG_MAX = ufloat(1, 0)

_spt_to_teff_map = {
    "M": ufloat(3100, 800),
    "K": ufloat(4600, 700),
    "G": ufloat(5650, 350),
    "F": ufloat(6700, 500),
    "A": ufloat(8600, 1300),
    "B": ufloat(20000, 10000),
    "O": ufloat(35000, 10000)
}

def append_calculated_inc_predictions(preds: np.ndarray[UFloat],
                                      field_name: str="inc") -> np.ndarray[UFloat]:
    """
    Calculate the predictions' inclination value(s) (in degrees) and append/overwrite to the array.

    :predictions: the predictions structured array to which inclination should be appended
    :field_name: the name of the inclination field to write to
    :returns: the revised array
    """
    with warnings.catch_warnings(category=[FutureWarning]):
        # Deprecation warning caused by the use of np.clip on ufloats
        warnings.filterwarnings("ignore", r"AffineScalarFunc.(__le__|__ge__)\(\) is deprecated.")
        names = list(preds.dtype.names)
        if "bP" in names:
            # From primary impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
            r1 = preds["rA_plus_rB"] / (1+preds["k"])
            e_squared = preds["ecosw"]**2 + preds["esinw"]**2
            cosi = np.clip(preds["bP"]*r1*(1+preds["esinw"]) / (1-e_squared), _TRIG_MIN, _TRIG_MAX)
            inc = degrees(arccos(cosi))
        elif "cosi" in names:
            cosi = np.clip(preds["cosi"], _TRIG_MIN, _TRIG_MAX)
            inc = degrees(arccos(cosi))
        elif "sini" in names:
            sini = np.clip(preds["sini"], _TRIG_MIN, _TRIG_MAX)
            inc = degrees(arcsin(sini))
        else:
            raise KeyError("Missing bP, cosi or sini in predictions required to calc inc.")

        if field_name not in names:
            # It's difficult to append a field to structured array or recarray so copy to new inst.
            # The numpy recfunctions module has merge and append_field funcs but they're slower.
            new = np.empty_like(preds, np.dtype(preds.dtype.descr + [(field_name, UFloat.dtype)]))
            new[names] = preds[names]
            new[field_name] = inc
        else:
            new = preds
        new[field_name] = inc
        return new


def get_teff_from_spt(target_spt):
    """
    Estimates a stellar T_eff [K] from the passed spectral type.

    :target_spt: the spectral type string
    :returns: the estimated teff in K
    """
    teff = None

    # Also add the whole spt in case it's just a single char (i.e.: V889 Aql is set to "A")
    if target_spt is not None \
            and (spts := re.findall(r"([A-Z][0-9])", target_spt) + [target_spt.upper()]):
        for spt in spts:
            if spt and len(spt) and (tp := spt.strip()[0]) in _spt_to_teff_map \
                and _spt_to_teff_map[tp].n > (teff.n if teff is not None else 0):
                teff = _spt_to_teff_map[tp]
    return teff


def dist_by_brightness_and_teff(Teff1: float, Teff2: float,
                                R1: float, R2: float,
                                band: Union[int, str],
                                mag: float,
                                component: int=None):
    """
    Calculate a distance using the Kervella (2004A&A...426..297K)
    surface brightness-effective temperature relations.

    Based on the absdim 15 (Southworth) KERV function (ln 1471-1512)

    :Teff1: effective temperature of star 1 (in units of K)
    :Teff2: effective temperature of star 2 (in units of K)
    :R1: the radius of star 1 (in units of R_Sun)
    :R2: the radisu of star 2 (in units of R_Sun)
    :band: on of the supported photometric bands; U, B, V, R, I, J, H, K or L
    :mag: the de-reddened apparent magnitude in the chosen band (see deredden)
    :component: which component's distance; the system as a whole [None], or star 1 or 2
    :returns: the distance (in units of pc)
    """
    RSUN = 6.9599e8     # Sol radius in m
    AU = 1.495979e11    # Astronomical unit (AU) in m

    if component is None: # System as a whole
        factor = 1000.0 * RSUN / AU
        zmld1 = 10**get_log_zmld(Teff1, band)
        zmld2 = 10**get_log_zmld(Teff2, band)
        helper = (R1 * factor / zmld1)**2 + (R2 * factor / zmld2)**2
        dist = 10**(0.2 * mag) * 2 * helper**0.5
    elif component == 1:
        dist = 9.3048e0 * R1 / 10**(get_log_zmld(Teff1, band) - 0.2 * mag)
    elif component == 2:
        dist = 9.3048e0 * R2 / 10**(get_log_zmld(Teff2, band) - 0.2 * mag)
    else:
        raise ValueError("components must be one of None, 1 or 2")
    return dist

def get_log_zmld(Teff, band: Union[int, str]) -> float:
    """
    Calculate the zero-magnitude angular diameter in mas,
    with the Kervella (2004A&A...426..297K) calibrations.

    Based on the absdim 15 (Southworth) GETLOGSZMLD function (ln 1514-1530)

    :Teff: the star's effective temperature (in units of K)
    :band: on of the supported photometric bands; U, B, V, R, I, J, H, K or L
    :returns: the angular diameter
    """
    # Coefficient lookups (Table 5, last page) for bands "U", "B", "V", "R", "I", "J", "H", "K", "L"
    A = (5.6391e0, 3.6753e0, 3.0415e0, 2.1394e0, 0.9847e0,
         0.9598e0, 1.1684e0, 0.8470e0, 0.6662e0)
    B = (-46.4505e0, -30.9671e0, -25.4696e0, -18.0221e0,
         -8.7985e0, -8.3451e0, -9.6156e0, -7.0790e0, -5.6609)
    C = (96.0513e0, 65.5421e0, 53.7010e0, 38.3497e0, 19.9281e0,
         18.5204e0, 20.2779e0, 15.2731e0, 12.4902)

    band_ix = _band_index(band)
    return A[band_ix] * log10(Teff)**2 \
            + B[band_ix] * log10(Teff) \
            + C[band_ix]

def deredden(ebv: float, mag: float, band: Union[int, str],
             flux_rat=None, component: int=None):
    """
    Dereddens the apparent magnitudes

    Based on the absdim 15 (Southworth) DEREDDEN function (ln 1532-1558)

    :ebv: the extinction E(B-V)
    :mag: the apparent magnitude of the dEB as a whole
    :band: on of the supported photometric bands; U, B, V, R, I, J, H, K or L
    :flux_rat: the flux ratio of star2/star1 - required if component magnitude required
    :component: which component's magnitude; the system as a whole [None], or star 1 or 2    
    :returns: the chosen dereddened magnitude
    """
    # For bands "U", "B", "V", "R", "I", "J", "H", "K", "L"
    av = (5.0e0, 4.2e0, 3.2e0, 2.5e0, 2.0e0, 1.0e0, 0.6e0, 0.5e0, 0.4e0)
    band_ix = _band_index(band)

    mag -= av[band_ix] * ebv
    if component is not None:
        if flux_rat is None:
            raise ValueError("flux_rat required if component is not None")
        if component == 1:
            mag += 2.5 * log10(1.0 + flux_rat)
        elif component == 2:
            mag += 2.5 * log10((1.0 + flux_rat) / flux_rat)
        else:
            raise ValueError("only component values of None, 1 or 2 supported")
    return mag

def _band_index(band: Union[str, int]) -> int:
    if isinstance(band, str):
        return ["U", "B", "V", "R", "I", "J", "H", "K", "L"].index(band)
    return band
