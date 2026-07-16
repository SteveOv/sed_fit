""" A class for handling the generation of model fluxes for filters sourced from bt-settl data """
# pylint: disable=no-member, multiple-statements
from abc import ABC as _AbstractBaseClass, abstractmethod as _abstractmethod
from typing import Union as _Union, Tuple as _Tuple, Iterable as _Iterable, List as _List
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile, getfullargspec as _getfullargspec
from warnings import filterwarnings as _filterwarnings
import re as _re
from json import load as _json_load
from urllib.parse import quote_plus as _quote_plus
from itertools import product as _product
from datetime import datetime as _datetime, timezone as _timezone
from functools import lru_cache as _lru_cache

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike

from scipy.stats import binned_statistic as _binned_statistic
from scipy.interpolate import RegularGridInterpolator as _RegularGridInterpolator
from scipy.interpolate import RBFInterpolator as _RBFInterpolator

import astropy.units as _u
from astropy.table import Table as _Table
from astropy.io.votable import parse_single_table as _parse_single_table

from dust_extinction.baseclasses import BaseExtModel as _BaseExtModel

# We parse units as text from votables & text files. Stop us getting swamped format with warnings.
_filterwarnings("ignore", category=_u.UnitsWarning)

class StellarGrid(_AbstractBaseClass):
    """ Base for classes which generate model stellar fluxes """
    # pylint: disable=too-many-arguments, too-many-positional-arguments

    _this_dir = _Path(_getsourcefile(lambda:0)).parent
    _CACHE_DIR = _this_dir / "../.cache"
    _DEF_FILTER_MAP_FILE = _this_dir / "data/stellar_grids/sed-filter-mappings.json"

    # Default output units
    _DEF_LAM_UNIT = _u.um
    _DEF_TEFF_UNIT = _u.K
    _DEF_LOGG_UNIT = _u.dex
    _DEF_FLUX_DENSITY_UNIT = _u.W / _u.m**2 / _u.Hz
    _DEF_FLUX_UNIT = _u.W / _u.m**2

    # For calculating fluxes for stars with given radius in R_sun and distance in pc
    _pc = (1 * _u.pc).to(_u.m).value
    _R_sun = (1 * _u.R_sun).to(_u.m).value

    def __init__(self,
                 wavelength_range: _Tuple[float],
                 teff_range: _Tuple[float],
                 logg_range: _Tuple[float],
                 metal_range: _Tuple[float],
                 extinction_model: _BaseExtModel=None,
                 wavelength_unit: _u.Unit=_DEF_LAM_UNIT,
                 teff_unit: _u.Unit=_DEF_TEFF_UNIT,
                 logg_unit: _u.Unit=_DEF_LOGG_UNIT,
                 flux_unit: _u.Unit=_DEF_FLUX_UNIT):
        """
        Initializes a new instance of this class.
        Sets up any extinction model and the details of the supported filters.
        
        :wavelength_range: the range of supported wavelengths
        :teff_range: the range of supported effective temperatures
        :logg_range: the range of supported logg values
        :metal_range: the range of supported metallicities
        :extinction_model: optional extinction model to use if applying extinction to model fluxes
        :wavelength_unit: the unit for the wavelength values
        :teff_unit: the unit for the effective temperature values
        :logg_unit: the unit for the logg values
        :flux_unit: the unit for the flux values
        :verbose: whether or not to output verbose status messages
        """
        super().__init__()
        self._wavelength_range = wavelength_range
        self._teff_range = teff_range
        self._logg_range = logg_range
        self._metal_range = metal_range
        self._extinction_model = extinction_model
        self._wavelength_unit = wavelength_unit
        self._teff_unit = teff_unit
        self._logg_unit = logg_unit
        self._flux_unit = flux_unit

        # The json has maps betweeen name of supported Vizier SED filters the corresponding SVO name
        with open(StellarGrid._DEF_FILTER_MAP_FILE, "r", encoding="utf8") as j:
            self._filters = { viz: self.get_filter(svo, wavelength_unit, extinction_model)
                                                            for viz, svo in _json_load(j).items() }
            self._filter_names_list = list(self._filters.keys())

    @property
    def extinction_model(self) -> _BaseExtModel:
        """ Get the model used to apply extinction to fluxes """
        return self._extinction_model

    @property
    def wavelength_range(self) -> _Tuple[float]:
        """ Gets the range of wavelength covered by this model (units of wavelength_unit)"""
        return self._wavelength_range

    @property
    def teff_range(self) -> _Tuple[float]:
        """ Gets the range of effective temperatures covered by this model (units of teff_unit) """
        return self._teff_range

    @property
    def logg_range(self) -> _Tuple[float]:
        """ Gets the range of logg covered by this model (units of logg_unit) """
        return self._logg_range

    @property
    def metal_range(self) -> _Tuple[float]:
        """ Gets the range of metallicities covered by this model """
        return self._metal_range

    @property
    def wavelength_unit(self) -> _u.Unit:
        """ Gets the unit of the flux wavelengths """
        return self._wavelength_unit

    @property
    def teff_unit(self) -> _u.Unit:
        """ Gets the temperature units """
        return self._teff_unit

    @property
    def logg_unit(self) -> _u.Unit:
        """ Gets the logg units """
        return self._logg_unit

    @property
    def flux_unit(self) -> _u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._flux_unit

    def has_filter(self, filter_name: _Union[str, _Iterable]) -> _np.ndarray[bool]:
        """ Gets whether this model knows of the requested filter(s) """
        return _np.isin(filter_name, self._filter_names_list)

    def get_filter_indices(self, filter_names: _Union[str, _Iterable]) -> _np.ndarray[int]:
        """
        Get the indices of the given filters. Useful in optimizing filter access when iterating as
        the indices can be used in place of the names. Raises a ValueError if a filter is unknown.

        :filter_names: a list of filters for which we want the indices
        :returns: an array of the equivalent indices
        """
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        return _np.array([self._filter_names_list.index(n) for n in filter_names], dtype=int)

    def get_filter_fluxes(self,
                          filters: _ArrayLike,
                          teff: float,
                          logg: float,
                          metal: float=0.,
                          radius: float=None,
                          distance: float=None,
                          av: float=None) -> _np.ndarray[float]:
        """
        Will return flux values for a target with the requested filters, teff, logg & metal values,
        optionally modified by stellar radius/distance and extinction values.

        Will raise a ValueError if a named filter is unknown.
        Will raise IndexError if an indexed filter is out of range.

        :filters: a list of filter names or indices for which we are generating fluxes
        :teff: the effective temperature value for the fluxes (in K)
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :radius: optional stellar radius value in R_sun
        :distance: optional stellar distance value in pc
        :av: optional A_v value with which to redden fluxes, if we also have an extinction model
        :returns: the resulting flux values (in implied flux_units)
        """
        # Find the unique filters and the map onto the request/response (a filter can appear > once)
        # filters may be specified as either names or as indices (after call to get_filter_indices).
        if isinstance(filters, (str|int)):
            unique_filters, flux_mappings = _np.array([filters]), _np.array([0])
        else:
            unique_filters, flux_mappings = _np.unique(filters, return_inverse=True)

        if unique_filters.dtype not in (_np.int64, _np.int32): # Need the filters' column index
            unique_filters = self.get_filter_indices(unique_filters)

        fluxes = _np.array([self.get_filter_flux(f, teff, logg, metal, radius, distance, av)
                                                                        for f in unique_filters])

        # Map these fluxes onto the response, where a filter/flux may appear more than once
        return _np.array([fluxes[m] for m in flux_mappings], dtype=float)

    def get_filter_flux(self,
                        the_filter: _Union[str, int],
                        teff: float,
                        logg: float,
                        metal: float,
                        radius: float=None,
                        distance: float=None,
                        av: float=None) -> float:
        """
        Will return flux values for a target with the requested filter, teff, logg & metal values,
        optionally modified by stellar radius/distance and extinction values.

        Will raise a ValueError if a named filter is unknown.
        Will raise IndexError if an indexed filter is out of range.

        :the_filter: the chosen filter by name or index
        :teff: the effective temperature value for the fluxes (in K)
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :radius: optional stellar radius value in R_sun
        :distance: optional stellar distance value in pc
        :av: optional A_v value with which to redden fluxes, if we also have an extinction model
        :returns: the resulting flux value (in implied flux_units)
        """
        flux = 0
        if isinstance(the_filter, str):
            filter_table = self._filters[the_filter]
        else:
            filter_table = self._filters[self._filter_names_list[the_filter]]

        # Work out the lambda range where the filter and grid data overlap
        ol_lam_short = max(min(self.wavelength_range), filter_table.meta["filter_short"].value)
        ol_lam_long = min(max(self.wavelength_range), filter_table.meta["filter_long"].value)
        if ol_lam_short < ol_lam_long: # No overlap; no flux
            # Filter's wavelengths & transmission coeffs in the region it overlaps the fluxes
            filter_lam = filter_table["Wavelength"].quantity.value
            ol_mask = (ol_lam_short <= filter_lam) & (filter_lam <= ol_lam_long)

            fluxes = self.get_fluxes(filter_lam[ol_mask], teff, logg, metal, radius, distance, av=0)
            if av:
                # Optimisation for filters, use cached Ax/Av ratios for the filter's set of lams
                if self.extinction_model is None:
                    raise ValueError("av specified but cannot redden without an extinction_model")
                fluxes *= _np.power(10, -0.4 * filter_table["axav"][ol_mask] * av)
            flux = _np.sum(_np.multiply(fluxes, filter_table["Norm-Transmission"][ol_mask].value))
        return flux

    def get_fluxes(self,
                   wavelengths: _ArrayLike,
                   teff: float,
                   logg: float,
                   metal: float=0,
                   radius: float=None,
                   distance: float=None,
                   av: float=None) -> _np.ndarray[float]:
        """
        Will return flux values for a target at the requested  wavelengths for the requested teff,
        logg, metal values, optionally modified by stellar radius/distance and extinction values.

        :wavelengths: array of wavelengths to get fluxes for
        :teff: the effective temperature value for the fluxes (in K)
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :radius: optional stellar radius value in R_sun
        :distance: optional stellar distance value in pc
        :av: optional A_v value with which to redden fluxes, if we also have an extinction model
        :returns: the resulting flux values (in implied flux_units)
        """
        # Get the subclass to generate the fluxes in their own special way.
        fluxes = self._get_surface_fluxes(wavelengths, teff, logg, metal)

        if radius and distance:
            fluxes *= ((radius * self._R_sun) / (distance * self._pc))**2

        if av:
            if self.extinction_model is None:
                raise ValueError("av specified but cannot redden without an extinction_model")
            wavenumbers = (1 / (wavelengths << self.wavelength_unit)).to(1 / _u.um)
            fluxes *= self.extinction_model.extinguish(wavenumbers, Av=av)
        return fluxes

    @_abstractmethod
    def _get_surface_fluxes(self, wavelengths: _ArrayLike, teff: float, logg: float, metal: float) \
            -> _np.ndarray[float]:
        """
        Called by get_fluxes() to generate the surface fluxes for the passed wavelengths and params.

        :wavelengths: array of wavelengths to get fluxes for (in units of self.wavelength_unit)
        :teff: the effective temperature value for the fluxes (in units of K)
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :returns: the resulting flux values (in implied self.flux_units)
        """

    @classmethod
    def get_filter(cls,
                   svo_name: str,
                   wavelength_unit: _u.Unit,
                   extinction_model: _BaseExtModel=None) -> _Table:
        """
        Downloads and caches the requested filter from the SVO. Returns a table of the filter's
        Wavelength and Transmission fields, and adds a Norm-Transmission column.
        Will also add meta entries for filter_short, filter_long and filter_mid to record
        the wavelength range covered by the filter and, if an extinction model is supplied, also
        axav and mid_axav items with extinction curve Ax/Av ratios at the filter's wavelengths.

        :svo_name: the unique name of the filter given by the SVO
        :wavelength_unit: the wavelength unit for the Wavelength column
        :extinction_model: the extinction model being used with this filter
        :returns: and astropy Table with Wavelength, Transmission and Norm-Transmission columns
        """
        filter_cache_dir = cls._CACHE_DIR / ".filters/"
        filter_cache_dir.mkdir(parents=True, exist_ok=True)

        filter_fname = (filter_cache_dir / (_re.sub(r"[^\w\d.-]", "-", svo_name) + ".xml"))
        if not filter_fname.exists():
            try:
                fid = _quote_plus(svo_name)
                table = _Table.read(f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={fid}")
                table.write(filter_fname, format="votable")
            except ValueError as err:
                raise ValueError(f"No filter table in SVO for filter={svo_name}") from err

        table = _parse_single_table(filter_fname).to_table()
        ftrans = table["Transmission"]
        table["Norm-Transmission"] = ftrans / _np.sum(ftrans) # so total trans == 1

        # Add metadata on the filter coverage
        if table["Wavelength"].unit != wavelength_unit:
            table["Wavelength"]=table["Wavelength"].to(wavelength_unit, equivalencies=_u.spectral())
        table.sort("Wavelength")

        table.meta["filter_short"] = _np.min(table["Wavelength"].quantity)
        table.meta["filter_long"] = _np.max(table["Wavelength"].quantity)
        table.meta["filter_mid"] = _np.median(table["Wavelength"].quantity)

        # These are cached Ax/Av ratios for the chosen extinction model. They're surprisingly
        # expensive to get but remain fixed across the filter so we'll cache them here.
        if extinction_model:
            table["axav"] = extinction_model(1 / table["Wavelength"].quantity.to(_u.um))
            table.meta["axav_mid"] = extinction_model(1 / table.meta["filter_mid"].to(_u.um))

        return table

    @classmethod
    def _bin_fluxes(cls,
                    wavelengths: _u.Quantity,
                    fluxes: _u.Quantity,
                    bin_midpoints: _u.Quantity) -> _u.Quantity:
        """
        Will calculate and return the means of the fluxes within each of the requested bins.

        :wavelengths: source flux wavelengths and unit
        :fluxes: source fluxes and unit
        :bin_midpoints: the midpoint wavelength and unit of each bin to populate
        :returns: the binned fluxes in the same units as the input
        """
        if bin_midpoints.unit != wavelengths.unit:
            bin_midpoints = bin_midpoints.to(wavelengths.unit, equivalencies=_u.spectral())

        # Scipy wants bin edges so find midpoints between bins then extend by one at start & end.
        bin_mid_gaps = _np.diff(bin_midpoints) / 2
        bin_edges = _np.concatenate([[bin_midpoints[0] - (bin_mid_gaps[0])],
                                    bin_midpoints[:-1] + (bin_mid_gaps),
                                    [bin_midpoints[-1] + (bin_mid_gaps[-1])]]).value

        result = _binned_statistic(wavelengths.value, fluxes.value, statistic=_np.nanmean,
                                   bins=bin_edges, range=(bin_edges.min(), bin_edges.max()))
        return result.statistic << fluxes.unit


class SvoStellarGrid(StellarGrid, _AbstractBaseClass):
    """
    Base class, building on StellarGrid, of grids based on the ascii text format seen for at least
    BtSettl and Coelho grids published by the SVO at https://svo2.cab.inta-csic.es/theory/newov2/
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, line-too-long

    # The top of the files are expected to resemble the example below (with leading # chars)
    #
    # Coelho Synthetic stellar library (SEDs)
    # teff = 3800 K (value for the effective temperature for the model. Temperatures are given in K)
    # logg = 5 log(cm/s2) (value for Log(G) for the model.)
    # meta = -0.5  (value for the Metallicity for the model ([Fe/H]).)
    # afe = 0  (value for alpha elements over iron abundance, where the alpha-elements considered are O, Ne, Mg, Si, S, Ca and Ti.)
    #
    # column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom
    # column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A

    # Regexes for reading metadata from the SVO ascii files
    _PARAM_RE = \
        _re.compile(r"^#[\s*](?P<k>\w*)[\s]*=[\s]*(?P<val>[+-]?([0-9]*[.])?[0-9]+)", _re.MULTILINE)
    _LAMBDA_UNIT_RE = _re.compile(r"Wavelength in (?P<unit>[\w\/]*)$", _re.MULTILINE)
    _FLUX_UNIT_RE = _re.compile(r"Flux in (?P<unit>[\w\/]*)$", _re.MULTILINE)

    def __init__(self,
                 data_file: _Path,
                 extinction_model: _BaseExtModel=None,
                 use_quick_mode: bool=True,
                 interp_method: str=None,
                 verbose: bool=False):
        """
        Initializes a new instance of this class.

        If you set use_quick_mode to True, a set of pre-filtered grids will be created during
        initialization. These will take some time to set up, but once in place they greatly
        simplify the filtered flux calculations as we will be working with a total pre-filtered flux
        value per filter, to which optional radius, distance and exinction calculations are applied.

        Documentation on Scipy RegularGridInterpolator
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        :data_file: the source of the model data, in numpy npz format, or None to use default file
        :extinction_model: optional extinction model to use if applying extinction to model fluxes
        :use_quick_mode: if True, a set of pre-filtered grids are used for quicker flux calcs
        :interp_method: the method used by the interpolators (see scipy doc for options) or leave as
        None to let the class choose
        :verbose: whether or not to output verbose status messages
        """
        with _np.load(data_file, allow_pickle=True) as df:
            model_grid = df["model_grid"]
            meta = df["meta"].item()
            if verbose:
                created = meta.get("created", "unknown")
                print(f"Loading model grid from {data_file.name} created at {created}")

        if interp_method is None:
            interp_method="slinear" if min(model_grid.shape) > 1 else "linear"

        if verbose:
            print(f"{self.__class__.__name__} is initializing the fluxes interpolator with a grid",
                f"of {len(meta['teffs'])} teff, {len(meta['loggs'])} logg & {len(meta['metals'])}",
                f"metal values and {len(meta['wavelengths'])} wavelength bins",end="...",flush=True)

        # For reddening. The extinction model may restrict the wavelength range we can report on.
        # Older data files may be missing the wavelength_unit in which case we want a hard fail here
        if extinction_model is not None:
            wavenumbers = 1 / (meta['wavelengths'] << meta["wavelength_unit"]).to(_u.micron).value
            wavelength_mask = wavenumbers >= _np.min(extinction_model.x_range)
            wavelength_mask &= wavenumbers <= _np.max(extinction_model.x_range)
            self._wavelengths = meta['wavelengths'][wavelength_mask]
        else:
            self._wavelengths = meta['wavelengths']

        super().__init__(wavelength_range=(self._wavelengths.min(), self._wavelengths.max()),
                         teff_range=(meta['teffs'].min(), meta['teffs'].max()),
                         logg_range=(meta['loggs'].min(), meta['loggs'].max()),
                         metal_range=(meta['metals'].min(), meta['metals'].max()),
                         extinction_model=extinction_model,
                         wavelength_unit=meta["wavelength_unit"],
                         teff_unit=meta["teff_unit"],
                         logg_unit=meta["logg_unit"],
                         flux_unit=meta["flux_unit"])

        # Create the single interpolator over the full grid of flux data.
        # Used for the interpolation of fluxes for given teff, logg, metal & wavelengths.
        if verbose: print(f"will use {interp_method} interpolation", end="...", flush=True)
        index_points = (meta['teffs'], meta['loggs'], meta['metals'], meta['wavelengths'])
        self._model_full_interp = _RegularGridInterpolator(index_points, model_grid, interp_method)
        if verbose: print("done.")

        if use_quick_mode:
            if verbose: print("The use of quick mode requested so will set up pre-filtered fluxes.")
            self._use_quick_mode = False # We need fully worked fluxes to set up quick mode interps

            # Create a table of interpolators to optimize getting filters' fluxes for given teff,
            # logg and metal values with no extinction and radius/distance modification applied.
            nfilters = len(self._filter_names_list)
            if verbose: print(f"Initializing pre-filtered fluxes for {nfilters} filters", end="")
            self._filter_interps = _np.empty((nfilters, ),
                                        [("filter", "<U50"),("axav_mid", float),("interp", object)])
            index_points = (meta['teffs'], meta['loggs'], meta['metals'])
            fluxes_shape = model_grid.shape[:-1]  # no wavelengths
            for filter_ix, (filter_name, filter_table) in enumerate(self._filters.items()):
                if verbose: print(".", end="", flush=True)
                fluxes = _np.empty(shape=fluxes_shape, dtype=_np.float32)
                for teff, logg, metal in _product(*index_points):
                    tix = _np.where(meta['teffs'] == teff)
                    lix = _np.where(meta['loggs'] == logg)
                    mix = _np.where(meta['metals'] == metal)
                    fluxes[tix, lix, mix] = self.get_filter_flux(filter_name, teff, logg, metal)

                self._filter_interps[filter_ix] = (
                    filter_name,
                    filter_table.meta.get("axav_mid", 1),
                    _RegularGridInterpolator(index_points, fluxes, interp_method)
                )

        # Delay this, as we need get_filter_flux() in "full" mode to set up the filter interpolators
        self._use_quick_mode = use_quick_mode
        if verbose: print("done.")

    @property
    def wavelengths(self) -> _np.ndarray:
        """ Gets the wavelength values for which unfiltered fluxes are published. """
        return self._wavelengths

    def get_filter_flux(self,
                        the_filter: _Union[str, int],
                        teff: float,
                        logg: float,
                        metal: float,
                        radius: float=None,
                        distance: float=None,
                        av: float=None) -> float:
        flux = 0
        if self._use_quick_mode:
            if isinstance(the_filter, str):
                the_filter = self.get_filter_indices(the_filter)[0]
            interp_row = self._filter_interps[the_filter]

            # Approx: radius/dist & ext calcs applied to single total flux from filter interpolator
            flux = interp_row["interp"](xi=(teff, logg, metal))
            if radius and distance:
                flux *= ((radius * self._R_sun) / (distance * self._pc))**2
            if av:
                if self.extinction_model is None:
                    raise ValueError("av specified but cannot redden without an extinction_model")
                # A copy of eqn within dust_extinction BaseExtModel.extinguish() to give fractional
                # extinction, and using a cached copy of the filter's mid Ax/Av (<-- big speed up).
                flux *= _np.power(10.0, -0.4 * interp_row["axav_mid"] * av)
        else:
            # Fall back to the more expensive, full calculations. These call get_fluxes() and will
            # then apply any radius, distance and extinction calcs to all fluxes before summing.
            flux = super().get_filter_flux(the_filter, teff, logg, metal, radius, distance, av)
        return flux

    def _get_surface_fluxes(self, wavelengths: _ArrayLike, teff: float, logg: float, metal: float) \
                        -> _np.ndarray[float]:
        return self._model_full_interp(xi=(teff, logg, metal, wavelengths))

    @classmethod
    def make_grid_file(cls,
                       source_files: _Iterable,
                       out_file: _Path,
                       grid_nbins: int=None,
                       grid_lam_range: _Tuple=None,
                       grid_lam_sub_range: _Tuple=None,
                       wavelength_unit: _u.Unit=StellarGrid._DEF_LAM_UNIT,
                       teff_unit: _u.Unit=StellarGrid._DEF_TEFF_UNIT,
                       logg_unit: _u.Unit=StellarGrid._DEF_LOGG_UNIT,
                       flux_density_unit: _u.Unit=StellarGrid._DEF_FLUX_DENSITY_UNIT,
                       flux_unit: _u.Unit=StellarGrid._DEF_FLUX_UNIT):
        """
        Will ingest the chosen ascii grid files, previously downloaded from the SVO Theoretical
        Spectra service, to produce a grid file containing the grids of fluxes and associated
        metadata to act as a source for instances of this class.

        Either both grid_nbins and grid_lam_range are expected to have values, in which case they
        are used to define the wavelength bins into which fluxes are re-binned, or both should
        be None, in which case the wavelength bins & fluxes in the source files are used as is
        (provided they are consistent across all of the files). 

        :source_files: an iterator/list of the source SVO format ascii files to read
        :out_file: the model file to write (overwriting any existing file)
        :grid_nbins: the number of binned fluxes to store per row, or None for no re-binning
        :grid_lam_range: wavelength range (to, from) of the grid [micron], or None for no re-binning
        :grid_lam_sub_range: subset range (to, from) of grid_lam_range to use, or None for all
        :wavelength_unit: the unit for the output wavelength values
        :teff_unit: the unit for the output effective temperature values
        :logg_unit: the unit for the output logg values
        :flux_density_unit: the unit for working with flux density values
        :flux_unit: the unit for the output flux values
        """
        # Need the files in sorted list as we go through more than once & the order may set indices.
        source_files = sorted(source_files)
        print(f"{cls.__name__}.make_grid_file(): importing {len(source_files)} SVO ascii",
              f"grid files into a compressed model file to be written to:\n\t{out_file}\n")

        # For now restrict our working to alpha/afe == zero
        index_names = ["teff", "logg", "metal"]
        index_vals = cls._get_list_of_index_values(source_files, index_names, True)

        # We will either re-bin the fluxes at wavelengths defined by #bins and range, or we directly
        # use the fluxes at wavelengths common to all of the source files (if both args None).
        do_bin_fluxes = grid_nbins is not None and grid_lam_range is not None
        if do_bin_fluxes:
            print(f"Will bin fluxes in {grid_nbins} bins over {grid_lam_range} {wavelength_unit}")
            grid_bin_lams = _np.geomspace(*grid_lam_range, grid_nbins, True) << wavelength_unit
        else:
            print("Binning not requested so will use the published fluxes directly")
            grid_bin_lams = cls._get_common_wavelengths(source_files, wavelength_unit)
            grid_nbins = len(grid_bin_lams)

        # This is an option for creating a smaller data grid which gives the same results, within
        # its range, as the full grid as the spacing will be the same within the subset range.
        if grid_lam_sub_range:
            print(f"Selecting a subset of the bins within {grid_lam_sub_range} {wavelength_unit}.")
            grid_bin_mask = (min(grid_lam_sub_range) * wavelength_unit <= grid_bin_lams) \
                          & (grid_bin_lams <= max(grid_lam_sub_range) * wavelength_unit)
            grid_bin_lams = grid_bin_lams[grid_bin_mask]
            grid_nbins = len(grid_bin_lams)
            print(f"Will now produce a grid for only the {grid_nbins} bins within the subset.")

        grid_bin_freqs = grid_bin_lams.to(_u.Hz, equivalencies=_u.spectral())

        # Now set up the multi-D index array and the target bin fluxes grid which we will populate
        # We can't rely on sorting the files for the correct order as + & - switched for metals.
        teffs = _np.unique(index_vals["teff"])
        loggs = _np.unique(index_vals["logg"])
        metals = _np.unique(index_vals["metal"])
        folded_index_shape = (len(teffs), len(loggs), len(metals))
        index_vals = index_vals.reshape(folded_index_shape)
        model_grid = _np.full(folded_index_shape + (grid_nbins, ), _np.nan, dtype=_np.float32)

        for file_ix, source_file in enumerate(source_files):
            meta = cls._read_metadata_from_ascii_model_file(source_file)
            print(f"{file_ix+1}/{len(source_files)} {source_file.name}", end="...", flush=True)
            if meta.get("alpha", 0) != 0 or meta.get("afe", 0) != 0:
                print(f"skipped row as alpha != 0 ({meta['alpha']})")
            else:
                lams, flux_dens = _np.genfromtxt(source_file, _np.float32, "#", unpack=True)
                lams = (lams * meta["lambda_unit"]).to(wavelength_unit, equivalencies=_u.spectral())
                flux_dens = (flux_dens * meta["flux_unit"])\
                                    .to(flux_density_unit, equivalencies=_u.spectral_density(lams))

                print(f"[{len(lams):,d} rows]:",
                      ", ".join(f"{k}={meta[k]: .2f}" for k in index_names), end="...", flush=True)

                # Write the row of fluxes to the full grid.
                tix = _np.where(teffs == meta["teff"])
                lix = _np.where(loggs == meta["logg"])
                mix = _np.where(metals == meta["metal"])
                if do_bin_fluxes:
                    bin_flux_dens = cls._bin_fluxes(lams, flux_dens, grid_bin_lams)
                    model_grid[tix, lix, mix] = (bin_flux_dens * grid_bin_freqs).value
                else:
                    wix = _np.where(_np.in1d(lams, grid_bin_lams, assume_unique=True))
                    model_grid[tix, lix, mix] = (flux_dens[wix] * grid_bin_freqs).value
                print(f"added row of {grid_nbins} fluxes")

        # Interpolate any gaps in the grid. We can't interpolate on dimensions with only one choice.
        print("Interpolating missing values", end="...", flush=True)
        index_dim_multi = _np.array([d for d, size in enumerate(index_vals.shape) if size > 1])
        neighbours = 4**index_vals.ndim # limit RBF mem usage; otherwise scales as ~points^2
        for wix in range(grid_nbins):
            if wix % 100 == 0 and wix > 0: print(".", end="", flush=True)
            nans = _np.isnan(model_grid[:, :, :, wix])    # This lam across all other dims
            if _np.all(nans):
                raise ValueError(f"Ooops! Nothing to interp @ lambda {grid_bin_lams[wix]}")
            if _np.any(nans):
                # Awkward; each index is a tuple of vals & we can't mask or use index lists on them.
                # Get pts into 2-d array of shape (npoints, ndims) skipping axes with single choice.
                pts = _np.array([[ix[d] for d in index_dim_multi] for ix in index_vals[~nans]])
                vals = model_grid[~nans, wix]
                int_pts = _np.array([[ix[d] for d in index_dim_multi] for ix in index_vals[nans]])
                int_vals = _RBFInterpolator(pts, vals, neighbours, 5, "thin_plate_spline")(int_pts)
                model_grid[nans, wix] = _np.maximum(int_vals, 0.0)
        print("done.")

        # Complete the metadata; row & col indices (wavelengths) and units
        grid_meta = { "teffs": teffs, "loggs": loggs, "metals": metals,
                     "wavelengths": grid_bin_lams.value, "teff_unit": teff_unit,
                     "logg_unit": logg_unit, "wavelength_unit": wavelength_unit,
                     "flux_density_unit": flux_density_unit, "flux_unit": flux_unit,
                     "created": _datetime.now(_timezone.utc) }

        # Now we write out the model grids and metadata to a compressed npz file
        print(f"Saving model grids and metadata to {out_file}, overwriting any existing file.")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(out_file, model_grid=model_grid, meta=grid_meta)

    @classmethod
    def _read_metadata_from_ascii_model_file(cls, source_file: _Path) -> dict[str, any]:
        """
        Reads the metadata for teff/logg/metal/alpha values used to generate this model file
        and the units associated with them and the grid of wavelengths and flux densities.
        """
        # First few lines of each file has metadata on it teff/logg/meta/alpha and units
        with open(source_file, mode="r", encoding="utf8") as sf:
            text = sf.read(1000)
        metadata = {
            **{ m.group("k"): float(m.group("val")) for m in cls._PARAM_RE.finditer(text) },
            "teff_unit": _u.K,
            "logg_unit": _u.dex,
            "lambda_unit": _u.Unit(cls._LAMBDA_UNIT_RE.findall(text)[0]),
            "flux_unit": _u.Unit(cls._FLUX_UNIT_RE.findall(text)[0].replace("/A", "/Angstrom")),
        }

        if "meta" in metadata and not "metal" in metadata:
            metadata["metal"] = metadata.pop("meta")
        return metadata

    @classmethod
    def _get_list_of_index_values(cls, source_files: _ArrayLike,
                                  index_names: _List[str], dense: bool=False) -> _np.ndarray[float]:
        """ Gets a sorted structured NDArray of the index values across the source files. """
        if dense:
            index_lists = { }
            for source_file in source_files:
                metadata = cls._read_metadata_from_ascii_model_file(source_file)
                if all(n in metadata.keys() for n in index_names):
                    for k in index_names:
                        if k in index_lists:
                            index_lists[k] += [metadata[k]]
                        else:
                            index_lists[k] = [metadata[k]]
            index_list = list(_product(*(_np.unique(index_lists[k]) for k in index_names)))
        else:
            index_list = []
            for source_file in source_files:
                metadata = cls._read_metadata_from_ascii_model_file(source_file)
                if all(n in metadata.keys() for n in index_names):
                    index_list += [tuple(metadata[k] for k in index_names)]
        return _np.array(sorted(index_list), dtype=[(k, float) for k in index_names])

    @classmethod
    def _get_common_wavelengths(cls, source_files: _ArrayLike, unit: _u.um) -> _u.Quantity:
        """ Parse the source files & report the common set of wavelength values present in all. """
        ret_lams, source_lam_unit = None, None
        for source_file in source_files:
            if source_lam_unit is None:
                meta = cls._read_metadata_from_ascii_model_file(source_file)
                source_lam_unit = meta["lambda_unit"]
            lams, _ = _np.genfromtxt(source_file, _np.float32, "#", unpack=True)
            ret_lams = lams if ret_lams is None else _np.intersect1d(ret_lams, lams)
        return (ret_lams * source_lam_unit).to(unit, equivalencies=_u.spectral())


class BtSettlGrid(SvoStellarGrid):
    """ Generates model SED fluxes from pre-built grid of bt-settl-agss model fluxes. """
    DEF_DATA_FILE = SvoStellarGrid._this_dir / "data/stellar_grids/bt-settl/bt-settl-agss.npz"

    def __init__(self, data_file: _Path=DEF_DATA_FILE, extinction_model: _BaseExtModel=None,
                 use_quick_mode: bool=True, interp_method: str=None, verbose: bool=False):
        super().__init__(data_file, extinction_model, use_quick_mode, interp_method, verbose)


class KuruczGrid(SvoStellarGrid):
    """ Generates model SED fluxes from pre-built grid of Kurucz ODFNEW /NOVER fluxes. """
    DEF_DATA_FILE = SvoStellarGrid._this_dir / "data/stellar_grids/kurucz/kurucz-odfnew-nover.npz"

    def __init__(self, data_file: _Path=DEF_DATA_FILE, extinction_model: _BaseExtModel=None,
                 use_quick_mode: bool=True, interp_method: str=None, verbose: bool=False):
        super().__init__(data_file, extinction_model, use_quick_mode, interp_method, verbose)


def get_stellar_grid(grid: _Union[str, type[StellarGrid]], **kwargs) -> StellarGrid:
    """
    A factory method for creating StellarGrid subclass instances with the benefit of caching.

    :grid: the type of StellarGrid to get
    :kwargs: the arguments with which to initialize the grid (specific to the type of grid)
    :returns: the resulting instance
    """
    grid_type = None
    if isinstance(grid, str):
        def get_subclasses(superclass):
            for subclass in superclass.__subclasses__():
                yield subclass
                yield from get_subclasses(subclass)

        possible_names = [grid.casefold(), grid.casefold() + "grid"]
        for subclass in get_subclasses(StellarGrid):
            if subclass.__name__.casefold() in possible_names:
                grid_type = subclass
                break
    elif issubclass(grid, StellarGrid):
        grid_type = grid

    if grid_type is None:
        raise KeyError(f"No subclass of StellarGrid like {grid} found.")
    if _AbstractBaseClass in grid_type.__bases__:
        # Careful with this check, as we only want to check the immediate base class(es).
        # Avoid issubclass() as it will always be true because StellarGrid is abstract.
        raise ValueError(f"Cannot initialize the abstract class {grid_type.__name__}")

    # Ignore kwargs not used by the type's __init__ otherwise we may get a TypeError. This allows
    # calling code to send a superset of potential kwargs to this func & it will use what is needed.
    argspec = _getfullargspec(grid_type.__init__)
    expected_kwargs = { k: v for k, v in kwargs.items() if k in argspec.args and k not in ["self"] }
    return _init_type(grid_type, **expected_kwargs)

@_lru_cache
def _init_type(the_type: type, **kwargs):
    return the_type(**kwargs)
