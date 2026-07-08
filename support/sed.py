
""" Low level utility functions for SED ingest, pre-processing, estimation and fitting. """
# pylint: disable=no-member, multiple-statements
from typing import Tuple
from pathlib import Path
import re
from urllib.parse import quote_plus

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique
from astropy.io.votable import parse_single_table
import numpy as np


def get_sed_for_target(target: str,
                       search_term: str=None,
                       radius: float=0.1,
                       missing_uncertainty_ratio: float=0.1,
                       remove_duplicates: bool=False,
                       flux_unit=u.W / u.m**2 / u.Hz,
                       freq_unit=u.Hz,
                       wl_unit=u.micron,
                       verbose: bool=False) -> Table:
    """
    Gets spectral energy distribution (SED) observations for the target. These data are found and
    downloaded from the VizieR photometry tool (see https://vizier.cds.unistra.fr/vizier/sed/doc/).
    
    The VizieR photometry tool is developed by Anne-Camille Simon and Thomas Boch.

    The data are sorted and errorbars based on missing_uncertainty_ratio are set where none given
    (sed_eflux is either zero or NaN). The sed_flux, sed_eflux and sed_freq fields will be converted
    to the requested unit if necessary.

    Calculated fields are added for sed_wl (wavelength), sed_vfv and sed_evfv (freq * flux) to aid
    plotting, where x and y axes of wavelength and nu*F(nu) are often used.

    Tables will be locally cached within the `.cache/.sed/` directory for future requests.

    :target: the name of the target object
    :search_term: optional search term, or leave as None to use the target value
    :radius: the search radius in arcsec
    :missing_uncertainty_rate: uncertainty, as a ratio of the fluxes, to apply where none recorded
    :remove_duplicates: if True, only the first row for each combination of sed_filter, sed_freq,
    sed_flux, sed_eflux and coordinates will be included in the returned table
    :flux_unit: the unit of the returned sed_flux field (must support conversion from u.Jy)
    :freq_unit: the unit of the returned sed_freq field
    :wl_unit: the unit of the returned sed_wl field
    :verbose: whether to output diagnostics messages
    :returns: an astropy Table containing the chosen data, sorted by descending frequency
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    sed_cache_dir = Path(".cache/.sed/")
    sed_cache_dir.mkdir(parents=True, exist_ok=True)

    # Read in the SED for this target via the cache (filename includes both search criteria)
    sed_fname = sed_cache_dir / (re.sub(r"[^\w\d-]", "-", target.lower()) + f"-{radius}.vot")
    if not sed_fname.exists():
        if verbose: print(f"Table {sed_fname.name} not cached so will query the VizieR SED service")
        try:
            targ = quote_plus(search_term or target)
            sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={targ}&-c.rs={radius}")
            sed.write(sed_fname, format="votable") # votable matches that published in link above
        except ValueError as err:
            raise ValueError(f"No SED for target={target} and search_term={search_term}") from err

    # Read first/only table in votable & parse into a stock astropy Table (more consistent to use)
    sed = parse_single_table(sed_fname).to_table()
    sed.sort(["sed_freq"], reverse=True)
    rcount = len(sed)
    if verbose: print(f"Opened SED table {sed_fname.name} containing {rcount} row(s).")

    # Add wavelength which will be useful downstream
    sed["sed_wl"] = sed["sed_freq"].to(wl_unit, equivalencies=u.spectral())

    # Set flux uncertainties where none given
    mask_no_err = (sed["sed_eflux"].value == 0) | np.isnan(sed["sed_eflux"])
    sed["sed_eflux"][mask_no_err] = sed["sed_flux"][mask_no_err] * missing_uncertainty_ratio

    # Get the data into desired units
    if sed["sed_flux"].unit != flux_unit: # It's actually flux density, usually received in Jy
        sed["sed_flux"].convert_unit_to(flux_unit, equivalencies=u.spectral_density(sed["sed_wl"]))
        sed["sed_eflux"].convert_unit_to(flux_unit, equivalencies=u.spectral_density(sed["sed_wl"]))
    if sed["sed_freq"].unit != freq_unit:
        sed["sed_freq"].convert_unit_to(freq_unit, equivalencies=u.spectral())

    if remove_duplicates:
        sed = unique(sed, keep="first",
                     keys=["sed_filter","sed_freq","sed_flux","sed_eflux","_RAJ2000","_DEJ2000"])
        ucount = len(sed)
        if verbose: print(f"Dropped {rcount-ucount} duplicate(s) leaving {ucount} unique row(s).")
        sed.sort(["sed_freq"], reverse=True)
    return sed


def retain_only_closest_observations(sed: Table, target_coords: SkyCoord) -> Table:
    """
    Will parse the passed SED table and add a dist_r column for the angular distance of each
    observation from the target's coordinates. With this, the SED table will be updated to leave
    only the closest observation to the target for each sed_filter within the table.

    :sed: the SED table to filter
    :target_coordinates: the target's coordinates
    :returns: the revised SED table, sorted on the sed_wl field
    """
    sed["dist_r"] = np.sqrt((target_coords.ra.to(u.deg).value - sed['_RAJ2000'])**2
                            + (target_coords.dec.to(u.deg).value - sed['_DEJ2000'])**2)
    sed.sort(["sed_filter", "dist_r"])
    sed = unique(sed, keys=["sed_filter"], keep="first")
    sed.sort(["sed_wl"])
    return sed


def calculate_vfv(sed: Table,
                  freq_colname: str="sed_freq",
                  flux_colname: str="sed_flux",
                  flux_err_colname: str="sed_eflux",
                  unit=None) -> Tuple[u.Quantity, u.Quantity]:
    """
    Calculate the nu*F(nu) values from the passed SED Table. These are often plotted in place of
    raw flux/flux err values. New columns are not added directly to the table but may be added
    by client code, if required. For example:
    ```python
    sed["sed_vfv"], sed["sed_evfv"] = calculate_vfv(sed)
    ```

    :sed: the SED table which is the source of the fluxes
    :freq_colname: the name of the frequency column to use
    :flux_colname: the name of the flux column to use
    :flex_err_colname: the name of the flux uncertainty column to use
    :unit: optional unit to transform the result to - must be equivalent to the natural unit
    :returns: a tuple of astropy Quanities with values (sed_freq * sed_flux, sed_freq * sed_eflux)
    """
    freqs, fluxes, flux_errs = sed.columns[freq_colname, flux_colname, flux_err_colname].values()
    vfv = freqs.quantity * fluxes.quantity
    evfv = freqs.quantity * flux_errs.quantity
    if unit is not None:
        return vfv.to(unit, equivalencies=u.spectral_density(freqs.quantity)), \
                evfv.to(unit, equivalencies=u.spectral_density(freqs.quantity))
    return vfv, evfv
