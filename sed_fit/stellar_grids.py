""" A class for handling the generation of model fluxes for filters sourced from bt-settl data """
# pylint: disable=no-member
from abc import ABC as _AbstractBaseClass
from typing import Union as _Union, Tuple as _Tuple, Iterable as _Iterable, List as _List
from pathlib import Path as _Path
from inspect import getsourcefile as _getsourcefile
from warnings import filterwarnings as _filterwarnings
import re as _re
from json import load as _json_load
from urllib.parse import quote_plus as _quote_plus
from itertools import product as _product

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
    """ Base for classes which expose stellar fluxes """
    # pylint: disable=too-many-arguments, too-many-positional-arguments

    _this_dir = _Path(_getsourcefile(lambda:0)).parent
    _CACHE_DIR = _this_dir / "../.cache"
    _DEF_FILTER_MAP_FILE = _this_dir / "data/stellar_grids/sed-filter-mappings.json"

    # Default output units
    _LAM_UNIT = _u.um
    _FLUX_DENSITY_UNIT = _u.W / _u.m**2 / _u.Hz
    _FLUX_UNIT = _u.W / _u.m**2
    _TEFF_UNIT = _u.K
    _LOGG_UNIT = _u.dex

    # For calculating fluxes for stars with given radius in R_sun and distance in pc
    _pc = (1 * _u.pc).to(_u.m).value
    _R_sun = (1 * _u.R_sun).to(_u.m).value

    def __init__(self,
                 model_grid: _ArrayLike,
                 teffs: _ArrayLike,
                 loggs: _ArrayLike,
                 metals: _ArrayLike,
                 wavelengths: _ArrayLike,
                 extinction_model: _BaseExtModel=None):
        """
        Initializes a new instance of this class.

        :model_grid: a regular 4-D grid (teffs, loggs, metals, wavelengths) of flux values from
        which an interpolator can be initialized
        :teffs: the model_grid's teff index [0] values
        :loggs: the model_grid's logg index [1] values
        :metals: the model_grid's metal index [2] values
        :wavelengths: the model_grid's wavelength index [2] values
        :extinction_model: optional extinction model to use if applying extinction to model fluxes
        """
        super().__init__()

        # Create the single interpolator over the full grid of flux data. Used for the interpolation
        # of full spectrum of fluxes (over the wavelength range) for given teff, logg & metal.
        index_points = (teffs, loggs, metals)
        self._model_full_interp = _RegularGridInterpolator(index_points, model_grid, "linear")

        self._teff_range = (min(teffs), max(teffs))
        self._logg_range = (min(loggs), max(loggs))
        self._metal_range = (min(metals), max(metals))
        self._wavelengths = wavelengths

        # For reddening fluxes
        self._extinction_model = extinction_model
        self._wavenumbers = 1 / (wavelengths << self.wavelength_unit).to(_u.micron).value

        # An extinction model may restrict the wavelength range we can report on
        if extinction_model is not None:
            self._wavelength_mask = self._wavenumbers >= _np.min(extinction_model.x_range)
            self._wavelength_mask &= self._wavenumbers <= _np.max(extinction_model.x_range)
        else:
            self._wavelength_mask = _np.ones((len(wavelengths)), dtype=bool)

        # The json has maps betweeen name of supported Vizier SED filters the corresponding SVO name
        with open(StellarGrid._DEF_FILTER_MAP_FILE, "r", encoding="utf8") as j:
            self._filters = { viz: self.get_filter(svo, self._LAM_UNIT)
                                                            for viz, svo in _json_load(j).items() }
            self._filter_names_list = list(self._filters.keys())

        # Populate a grid of pre-filtered unreddened fluxes. Speed up for get_filter_fluxes if av=0
        interp_vals_shape = model_grid.shape[:-1]
        lams = wavelengths
        grid_filtered = _np.empty(interp_vals_shape + (len(self._filters),))
        for teff, logg, metal in _product(teffs, loggs, metals):
            tix = _np.where(teffs == teff)
            lix = _np.where(loggs == logg)
            mix = _np.where(metals == metal)
            fluxes = self._model_full_interp(xi=(teff, logg, metal))
            for filter_ix, (_, filter_table) in enumerate(self._filters.items()):
                filter_flux = self._get_filtered_total_flux(lams, fluxes, filter_table)
                grid_filtered[tix, lix, mix, filter_ix] = filter_flux

        # Create a table of interpolators, one per filter, for interpolating filter fluxes
        # for each filter for given teff, logg and metal values.
        index_points = (teffs, loggs, metals)
        self._model_interps = _np.empty(shape=(len(self._filters), ),
                                        dtype=[("filter", "<U50"), ("interp", object)])
        for filter_ix, filter_name in enumerate(self._filters):
            self._model_interps[filter_ix] = (
                filter_name,
                _RegularGridInterpolator(index_points, grid_filtered[:, :, :, filter_ix], "linear")
            )

    @property
    def extinction_model(self) -> _BaseExtModel:
        """ Get the model used to apply extinction to fluxes """
        return self._extinction_model

    @property
    def wavenumbers(self) -> _np.ndarray[float]:
        """ Gets the wavenumbers (1 / micron) of the flux wavelengths. """
        return self._wavenumbers[self._wavelength_mask]

    @property
    def wavelengths(self) -> _np.ndarray:
        """ Gets the wavelength values for which unfiltered fluxes are published. """
        return self._wavelengths[self._wavelength_mask]

    @property
    def wavelength_range(self) -> _Tuple[float]:
        """ Gets the range of wavelength covered by this model (units of wavelength_unit)"""
        return (self.wavelengths.min(), self.wavelengths.max())

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
    def teff_unit(self) -> _u.Unit:
        """ Gets the temperature units """
        return self._TEFF_UNIT

    @property
    def logg_unit(self) -> _u.Unit:
        """ Gets the logg units """
        return self._LOGG_UNIT

    @property
    def wavelength_unit(self) -> _u.Unit:
        """ Gets the unit of the flux wavelengths """
        return self._LAM_UNIT

    @property
    def flux_unit(self) -> _u.Unit:
        """ Gets the unit of the returned fluxes """
        return self._FLUX_UNIT

    def has_filter(self, filter_name: _Union[str, _Iterable]) -> _np.ndarray[bool]:
        """ Gets whether this model knows of the requested filter(s) """
        return _np.isin(filter_name, self._filter_names_list)

    def get_filter_indices(self, filter_names: _Union[str, _Iterable]) -> _np.ndarray[int]:
        """
        Get the indices of the given filters. Useful in optimizing filter access when iterating
        as the indices can be used in place of the names. Handles mapping filter names.

        Will raise a ValueError if a filter is unknown.

        :filter_names: a list of filters for which we want the indices
        :returns: an array of the equivalent indices
        """
        if isinstance(filter_names, str):
            filter_names = [filter_names]
        return _np.array([self._filter_names_list.index(n) for n in filter_names], dtype=int)

    def get_fluxes(self,
                   teff: float,
                   logg: float,
                   metal: float=0,
                   radius: float=None,
                   distance: float=None,
                   av: float=None) -> _np.ndarray[float]:
        """
        Will return a full spectrum of fluxes, over this model's wavelength range for the
        requested teff, logg and metal values.

        If both radius and distance are given the fluxes will be modified for a star of the given
        radius (in R_Sun) at the given distance (in pc).

        :teff: the effective temperature for the fluxes
        :logg: the logg for the fluxes
        :metal: the metallicity for the fluxes
        :radius: optional stellar radius value in R_sun
        :distance: optional stellar distance value in pc
        :av: optional A_v value with which to redden fluxes, if we also have an extinction model
        :returns: the resulting flux values (in implied flux_units)
        """
        flux = self._model_full_interp(xi=(teff, logg, metal))[self._wavelength_mask]
        if radius is not None and distance is not None:
            flux *= ((radius * self._R_sun) / (distance * self._pc))**2
        if av is not None and self.extinction_model is not None:
            flux *= self.extinction_model.extinguish(self.wavenumbers << (1 / _u.um), Av=av)
        return flux

    def get_filter_fluxes(self,
                          filters: _ArrayLike,
                          teff: float,
                          logg: float,
                          metal: float=0.,
                          radius: float=None,
                          distance: float=None,
                          av: float=None) -> _np.ndarray[float]:
        """
        Will return a ndarray of flux values calculated for requested filter names at
        the chosen effective temperature, logg and metallicity values.

        If both radius and distance are given the fluxes will be modified for a star of the given
        radius (in R_Sun) at the given distance (in pc).

        Will raise a ValueError if a named filter is unknown.
        Will raise IndexError if an indexed filter is out of range.

        :filters: a list of filter names or indices for which we are generating fluxes
        :teff: the effective temperature for the fluxes
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

        if not av: # so None or 0.0
            # As there's no av we can use the pre-calculated grid of unreddened filter fluxes
            if unique_filters.dtype not in (_np.int64, _np.int32): # Need the filters' column index
                unique_filters = self.get_filter_indices(unique_filters)
            filter_flux = _np.array([
                self._model_interps[f]["interp"]((teff, logg, metal)) for f in unique_filters])

            # Optionally adjust for stellar params
            if radius is not None and distance is not None:
                filter_flux *= ((radius * self._R_sun) / (distance * self._pc))**2
        elif self.extinction_model is not None:
            # Get the full set of reddened fluxes and will apply rad/distance as appropriate.
            lam = self.wavelengths
            flux = self.get_fluxes(teff, logg, metal, radius, distance, av)

            # Now apply the filters
            if unique_filters.dtype in (_np.int64, _np.int32):  # Need the names of the filters
                unique_filters = [self._filter_names_list[i] for i in unique_filters]
            filter_flux = _np.array([
                self._get_filtered_total_flux(lam, flux, self._filters[f]) for f in unique_filters])
        else:
            raise ValueError("av specified but unable to redden flux without an extinction_model")

        # Map these fluxes onto the response, where a filter/flux may appear >1 times
        return _np.array([filter_flux[m] for m in flux_mappings], dtype=float)

    @classmethod
    def get_filter(cls, svo_name: str, lambda_unit: _u.Unit) -> _Table:
        """
        Downloads and caches the requested filter from the SVO. Returns a table of the filter's
        Wavelength and Transmission fields, and adds a Norm-Transmission column.
        Will also add meta entries for filter_short, filter_long and filter_mid to record
        the wavelength range covered by the filter.

        :svo_name: the unique name of the filter given by the SVO
        :lambda_unit: the wavelength unit for the Wavelength column
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
        if table["Wavelength"].unit != lambda_unit:
            table["Wavelength"] = table["Wavelength"].to(lambda_unit, equivalencies=_u.spectral())
        table.meta["filter_short"] = _np.min(table["Wavelength"].quantity)
        table.meta["filter_long"] = _np.max(table["Wavelength"].quantity)
        table.meta["filter_mid"] = _np.median(table["Wavelength"].quantity)

        table.sort("Wavelength")
        return table

    @classmethod
    def _get_filtered_total_flux(cls,
                                 lambdas: _ArrayLike,
                                 fluxes: _ArrayLike,
                                 filter_table: _Table) -> _u.Quantity:
        """
        Calculate the total flux across a filter's bandpass.

        :lambdas: the wavelengths of the model fluxes
        :fluxes: the model fluxes
        :filter_grid: the grid (as returned by get_filter()) which describes the filter
        :returns: the summed flux passed through the filter
        """
        # Work out the lambda range where the filter and binned data overlap
        ol_lam_short = max(lambdas.min(), filter_table.meta["filter_short"].value)
        ol_lam_long = min(lambdas.max(), filter_table.meta["filter_long"].value)

        if ol_lam_short > ol_lam_long: # No overlap; no flux
            return 0.0

        # Get the filter's transmission coeffs in the region it overlaps the fluxes
        filter_lam = filter_table["Wavelength"].quantity.value
        filter_ol_mask = (ol_lam_short <= filter_lam) & (filter_lam <= ol_lam_long)
        filter_lam = filter_lam[filter_ol_mask]
        filter_trans = filter_table["Norm-Transmission"][filter_ol_mask].value

        # Apply the filter & calculate overall transmitted flux value
        filter_fluxes = _np.interp(filter_lam, lambdas, fluxes) * filter_trans
        return _np.sum(filter_fluxes)

    @classmethod
    def _bin_fluxes(cls,
                    lambdas: _ArrayLike,
                    fluxes: _ArrayLike,
                    lam_bin_midpoints: _ArrayLike) -> _u.Quantity:
        """
        Will calculate and return the means of the fluxes within each of the requested bins.

        :lambdas: source flux wavelengths
        :fluxes: source fluxes
        :lam_bin_midpoints: the midpoint lambda of each bin to populate
        :returns: the binned fluxes in the same units as the input
        """
        if lam_bin_midpoints.unit != lambdas.unit:
            lam_bin_midpoints = lam_bin_midpoints.to(lambdas.unit, equivalencies=_u.spectral())

        # Scipy wants bin edges so find midpoints between bins then extend by one at start & end.
        bin_mid_gaps = _np.diff(lam_bin_midpoints) / 2
        bin_edges = _np.concatenate([[lam_bin_midpoints[0] - (bin_mid_gaps[0])],
                                    lam_bin_midpoints[:-1] + (bin_mid_gaps),
                                    [lam_bin_midpoints[-1] + (bin_mid_gaps[-1])]]).value

        result = _binned_statistic(lambdas.value, fluxes.value, statistic=_np.nanmean,
                                   bins=bin_edges, range=(bin_edges.min(), bin_edges.max()))
        return result.statistic << fluxes.unit


class BtSettlGrid(StellarGrid):
    """
    Generates model SED fluxes from pre-built grids of bt-settl-agss model fluxes.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals

    _DEF_MODEL_FILE = StellarGrid._this_dir / "data/stellar_grids/bt-settl-agss/bt-settl-agss.npz"

    # Regexes for reading metadata from bt-settl ascii files
    _PARAM_RE = \
        _re.compile(r"^#[\s*](?P<k>\w*)[\s]*=[\s]*(?P<val>[+-]?([0-9]*[.])?[0-9]+)", _re.MULTILINE)
    _LAMBDA_UNIT_RE = _re.compile(r"Wavelength in (?P<unit>[\w\/]*)$", _re.MULTILINE)
    _FLUX_UNIT_RE = _re.compile(r"Flux in (?P<unit>[\w\/]*)$", _re.MULTILINE)


    def __init__(self,
                 data_file: _Path=_DEF_MODEL_FILE,
                 extinction_model: _BaseExtModel=None):
        """
        Initializes a new instance of this class.

        :data_file: the source of the model data, in numpy npz format
        :filter_map_file: json file containing mappings from VizieR to SVO filter names
        :extinction_model: optional extinction model to use if applying extinction to model fluxes
        """
        with _np.load(data_file, allow_pickle=True) as df:
            model_grid_full = df["grid_full"]
            meta = df["meta"].item()
            teffs = meta["teffs"]
            loggs = meta["loggs"]
            metals = meta["metals"]
            wavelengths = meta["wavelengths"]

        super().__init__(model_grid=model_grid_full,
                         teffs=teffs,
                         loggs=loggs,
                         metals=metals,
                         wavelengths=wavelengths,
                         extinction_model=extinction_model)

    @classmethod
    def make_grid_file(cls,
                       source_files: _Iterable,
                       out_file: _Path=_DEF_MODEL_FILE):
        """
        Will ingest the chosen bt-settl-agss ascii grid files to produce a grid file containing
        the grids of fluxes and associated metadata to act as a source for instances of this class.

        Download bt-settl-aggs ascii model grids from following url
        https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
        
        :source_files: an iterator/list of the source bt-settle ascii files to read
        :out_file: the model file to write (overwriting any existing file)
        """
        grid_full_nbins = 5000
        grid_full_bin_lams = _np.geomspace(0.05, 50, num=grid_full_nbins, endpoint=True) << _u.um
        grid_full_bin_freqs = grid_full_bin_lams.to(_u.Hz, equivalencies=_u.spectral())
        index_names = ["teff", "logg", "metal", "alpha"]

        # Need the files in sorted list as we go through them twice and the order may set indices
        source_files = sorted(source_files)
        print(f"{cls.__name__}.make_grid_file(): importing {len(source_files)} bt-settl-agss ascii",
              f"grid files into a new compressed model file written to:\n\t{out_file}\n")

        # For now restrict our working to alpha == zero
        index_vals = cls._get_list_of_index_values(source_files, index_names, True)
        index_names = index_names[:-1]
        alpha_zero_mask = index_vals["alpha"] == 0
        index_vals = index_vals[alpha_zero_mask][index_names]
        if sum(~alpha_zero_mask):
            print(f"Ignoring {sum(~alpha_zero_mask)} grid file(s) where alpha != 0")

        # Now set up the multi-D index array and the target bin fluxes grid which we will populate
        teffs = _np.unique(index_vals["teff"])
        loggs = _np.unique(index_vals["logg"])
        metals = _np.unique(index_vals["metal"])
        folded_index_shape = (len(teffs), len(loggs), len(metals))
        index_vals = index_vals.reshape(folded_index_shape)
        grid_full = _np.full(folded_index_shape + (grid_full_nbins, ), _np.nan, float)

        # Read in each source file, parse it, calculate the bin fluxes then store a row in the grid
        for file_ix, source_file in enumerate(source_files):
            meta = cls._read_metadata_from_ascii_model_file(source_file)
            print(f"{file_ix+1}/{len(source_files)} {source_file.name}", end="...")

            if meta["alpha"] != 0:
                print(f"skipped row as alpha != 0 ({meta['alpha']})")
            else:
                lams, flux_densities = _np.genfromtxt(source_file, float, comments="#", unpack=True)
                lams = (lams * meta["lambda_unit"]).to(cls._LAM_UNIT, equivalencies=_u.spectral())
                flux_densities = (flux_densities * meta["flux_unit"])\
                                .to(cls._FLUX_DENSITY_UNIT, equivalencies=_u.spectral_density(lams))

                print(f"[{len(lams):,d} rows]:",
                      ", ".join(f"{k}={meta[k]: .2f}" for k in index_names), end="...")

                # Write the row of binned fluxes to the full grid.
                bin_flux_densities = cls._bin_fluxes(lams, flux_densities, grid_full_bin_lams)
                tix = _np.where(teffs == meta["teff"])
                lix = _np.where(loggs == meta["logg"])
                mix = _np.where(metals == meta["metal"])
                grid_full[tix, lix, mix] = (bin_flux_densities *  grid_full_bin_freqs).value

                print(f"added row of {grid_full_nbins} binned fluxes")

        # Interpolate any gaps in the grid. We can't interpolate on dimensions with only one choice.
        print("Interpolating missing values", end="...")
        index_dim_has_multi = _np.array([d for d, size in enumerate(index_vals.shape) if size > 1])
        neighbours = 4**(len(index_names)) # limit RBF mem usage; otherwise scales as ~points^2
        for wix in range(grid_full_nbins):
            nans = _np.isnan(grid_full[:, :, :, wix])    # This lam across all other dims
            if _np.all(nans):
                raise ValueError("Ooops! Nothing to interp from")
            if _np.any(nans):
                # Awkward; each index is a tuple of vals & we can't mask or use index lists on them
                pts = _np.array([[ix[d] for d in index_dim_has_multi] for ix in index_vals[~nans]])
                xi = _np.array([[ix[d] for d in index_dim_has_multi] for ix in index_vals[nans]])
                grid_full[nans, wix] = _RBFInterpolator(pts, grid_full[~nans, wix], neighbours)(xi)
        print("done.")

        # Complete the metadata; row indices and col indices (filters & wavelengths)
        grid_meta = {
            "teffs": teffs,
            "loggs": loggs,
            "metals": metals,
            "wavelengths": grid_full_bin_lams.value
        }

        # Now we write out the model grids and metadata to a compressed npz file
        print(f"Saving model grids and metadata to {out_file}, overwriting any existing file.")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(out_file, meta=grid_meta, grid_full=grid_full)
        return out_file

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
    def _get_list_of_index_values(cls, source_files: _ArrayLike, index_names: _List[str],
                                  dense: bool=False) -> _np.ndarray[float]:
        """
        Gets a sorted structured NDArray of the index values across the source files.

        :source_files: the list of files to parse
        :index_names: the values to read from the files and to index on
        :dense: if True, the resulting list will be the Cartesian product of the unique values
        """
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


if __name__ == "__main__":
    # Download bt-settl-aggs ascii model grids from following url
    # https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
    # then decompress the tgz contents into the ../.cache/.modelgrids/bt-settl-agss dir

    # pylint: disable=protected-access
    in_files = (StellarGrid._CACHE_DIR / ".modelgrids/bt-settl-agss/").glob("lte*.dat.txt")
    new_file = BtSettlGrid.make_grid_file(sorted(in_files))
    bgrid = BtSettlGrid(new_file)
    print(f"\nLoaded newly created model grid from {new_file}")

    # Test what has been saved
    print("Teffs:", ",".join(f"{t:.2f}" for t in bgrid._model_full_interp.grid[0]))
    print("loggs:", ",".join(f"{l:.2f}" for l in bgrid._model_full_interp.grid[1]))
    print("metals:", ",".join(f"{m:.2f}" for m in bgrid._model_full_interp.grid[2]))

    print( "Filters:", ", ".join(bgrid._filter_names_list))

    print(f"\nRanges: teff={bgrid.teff_range} {bgrid.teff_unit:unicode},",
          f"logg={bgrid.logg_range} {bgrid.logg_unit:unicode}, metal = {bgrid.metal_range}")
    print("Test flux for 'GAIA/GAIA3:Gbp' filter, teff=2000, logg=4.0, metal=0, alpha=0:",
          ", ".join(f"{f:.3f}" for f in bgrid.get_filter_fluxes(["GAIA/GAIA3:Gbp"], 2000, 4, 0)),
          f"[{bgrid.flux_unit:unicode}]")
