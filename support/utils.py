""" General purpose utility/helper functions """
from typing import Union as _Union
from numbers import Number as _Number
import re as _re

import astropy.units as _u
from uncertainties import nominal_value as _nom_val, UFloat as _UFloat, ufloat as _ufloat

_to_file_safe_sub_pattern = _re.compile(r"[^\w\d._-]", _re.IGNORECASE)

_spt_to_teff_map = {
    "M": _ufloat(3100, 800),
    "K": _ufloat(4600, 700),
    "G": _ufloat(5650, 350),
    "F": _ufloat(6700, 500),
    "A": _ufloat(8600, 1300),
    "B": _ufloat(20000, 10000),
    "O": _ufloat(35000, 10000)
}

_spt_find_pattern = _re.compile(r"([A-Z]{1}[0-9]*)")

def to_file_safe_str(text: str, replacement: str="-", lower: bool=True) -> str:
    """
    Parse the text and replace any potentially troublesome characters when used as a file name.
    Do no pass in a full path as / and \\ are among the characters which will be replaced.

    :text: the original text
    :replacement: the character to substitute for any troublesome characters
    :lower: whether or not to force the revised text to lower case [True]
    :returns: the revised text
    """
    retval = _to_file_safe_sub_pattern.sub(replacement, text)
    return retval.lower() if lower else retval


def format_value(value: _Union[_Number, _UFloat],
                 unit: _Union[str, _u.UnitBase, _u.FunctionUnitBase]=None,
                 reference_value: _Union[_Number, _UFloat]=None,
                 num_format: str="9.3f",
                 large_num_format: str="6.3e",
                 large_num_threshold: float=1e6) -> str:
    """
    Converts the requested value to text with the accompanying unit and an optional reference value.
    The number format used will depend on the magnitude of the value, switching from num_format to
    large_num_format if the value reaches or exceeds large_num_threshold.
    """
    value_fmt = f"{{0:{num_format if _nom_val(value) < large_num_threshold else large_num_format}}}"
    unit_text = ""
    if isinstance(unit, (_u.UnitBase, _u.FunctionUnitBase)):
        unit_text = f" {unit:unicode}"
    elif isinstance(unit, str):
        unit_text = " " + unit

    text = value_fmt.format(value) + unit_text
    if reference_value:
        text += " " * max(1, 12 - len(unit_text))
        text += "(" + value_fmt.format(reference_value) + unit_text + ")"
    return text


def estimate_teff_from_spt(target_spt: str) -> _UFloat:
    """
    Estimates a stellar T_eff [K] from the passed spectral type.

    :target_spt: the spectral type string
    :returns: the estimated teff in implied units of K
    """
    teff = None
    if target_spt is not None:
        spts = _spt_find_pattern.findall(target_spt.strip().upper())
        for spt in (s for s in spts if len(s) > 0):
            tp = spt.strip()[0]
            if (tp in _spt_to_teff_map) and (teff is None or _spt_to_teff_map[tp].n > teff.n):
                teff = _spt_to_teff_map[tp]
    return teff
