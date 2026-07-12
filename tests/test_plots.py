""" Unit tests for the plots module. """
# pylint: disable=line-too-long, protected-access, no-member, too-many-locals, too-many-arguments
import unittest
import json

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

from sed_fit.stellar_grids import get_stellar_grid
from support.sed import get_sed_for_target, retain_only_closest_observations
from support.plots import plot_sed, plot_fitted_model 

class Testplots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initialize the class. """
        with open("./sed_fit/data/stellar_grids/sed-filter-mappings.json", "r", encoding="utf8") as j:
            cls.filters = json.load(j)

    @unittest.skip("Comment this out to run this interactive test")
    def test_plot_sed(self):
        """ Interactive test for easily running/testing plot_sed() """
        sed = get_sed_for_target("CW CMa", "V* CW CMa",
                                 radius=0.25, remove_duplicates=True, verbose=True)
        
        coords = SkyCoord(ra=110.4688392 * u.deg, dec=-23.7937069397 * u.deg)
        sed = retain_only_closest_observations(sed, coords)

        # Exclusions for range and filters (either unknown or known & problematic)
        model_mask = np.isin(sed["sed_filter"], list(self.filters.keys()), invert=False)
        model_mask &= np.isin(sed["_tabname"], ["B/denis/denis"], invert="True")
        model_mask &= (sed["sed_wl"] >= 0.1 * u.um) & (sed["sed_wl"] <= 22 * u.um)
        sed = sed[model_mask]

        fig = plot_sed(x=sed["sed_wl"].quantity,
                       fluxes=[sed["sed_flux"].quantity],
                       flux_errs=[sed["sed_eflux"].quantity],
                       fmts=[".k"],
                       fillstyles=["full"],
                       labels=["observation"],
                       show_grid=True,
                       figsize=(6, 4),
                       title="CW CMa SED")
        plt.show(block=True)

    @unittest.skip("Comment this out to run this interactive test")
    def test_plot_fitted_model(self):
        """ Interactive test for easily running/testing plot_fitted_model() """
        sed = get_sed_for_target("CW CMa", "V* CW CMa",
                         radius=0.25, remove_duplicates=True, verbose=True)
        
        coords = SkyCoord(ra=110.4688392 * u.deg, dec=-23.7937069397 * u.deg)
        sed = retain_only_closest_observations(sed, coords)

        stellar_grid = get_stellar_grid("BtSettlGrid", use_quick_mode=False)

        # Exclusions for range and filters (either unknown or known & problematic)
        model_mask = stellar_grid.has_filter(sed["sed_filter"])
        model_mask &= np.isin(sed["_tabname"], ["B/denis/denis"], invert="True")
        model_mask &= (sed["sed_wl"] >= min(stellar_grid.wavelength_range)) \
                    & (sed["sed_wl"] <= max(stellar_grid.wavelength_range))
        sed = sed[model_mask]

        fig = plot_fitted_model(sed=sed,
                                theta=(9881, 9600, 4.227, 4.253, 1.842, 1.739, 336.012, 0),
                                model_grid=stellar_grid,
                                sed_flux_colname="sed_flux",
                                sed_flux_err_colname="sed_eflux",
                                sed_filter_colname="sed_filter",
                                sed_lambda_colname="sed_wl",
                                show_component_spectra=False,
                                show_combined_spectrum=True,
                                show_legend=False,
                                show_grid=False,
                                figsize=(6, 4))
        plt.show(block=True)
