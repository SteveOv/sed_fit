""" Tests for the plots module. """
# pylint: disable=line-too-long, protected-access, no-member, too-many-locals, too-many-arguments, unused-variable
import unittest
import json
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from dust_extinction.parameter_averages import G23

from sed_fit.stellar_grids import get_stellar_grid
from support.sed import get_sed_for_target, retain_only_closest_observations
from support.plots import plot_sed, plot_fitted_model, plot_model_spectra
from support.utils import to_file_safe_str

class Testplots(unittest.TestCase):
    """ Tests for the plots module. """
    @classmethod
    def setUpClass(cls):
        """ Initialize the class. """
        with open("./sed_fit/data/stellar_grids/sed-filter-mappings.json", "r", encoding="utf8") as j:
            cls.filters = json.load(j)

    @unittest.skip("Comment this out to run this interactive test")
    def test_plot_sed(self):
        """ Interactive test for easily running/testing plot_sed() """
        sed = get_sed_for_target("CW CMa", "V* CW CMa", radius=0.25, remove_duplicates=True, verbose=True)

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
        # Also a handy option for reproducing the plot for fully fitted CW CMa
        sed = get_sed_for_target("CW CMa", "V* CW CMa", radius=0.25, remove_duplicates=True, verbose=True)

        coords = SkyCoord(ra=110.4688392 * u.deg, dec=-23.7937069397 * u.deg)
        sed = retain_only_closest_observations(sed, coords)

        stellar_grid = get_stellar_grid("BtSettlGrid", extinction_model=G23(Rv=3.1), use_quick_mode=False)

        # Exclusions for range and filters (either unknown or known & problematic)
        # and outliers which will have been excluded by the pruning algo.
        outliers = ["PAN-STARRS/PS1:i", "PAN-STARRS/PS1:y", "PAN-STARRS/PS1:r"]
        model_mask = stellar_grid.has_filter(sed["sed_filter"])
        model_mask &= np.isin(sed["_tabname"], ["B/denis/denis"], invert="True")
        model_mask &= np.isin(sed["sed_filter"], outliers, invert="True")
        model_mask &= (sed["sed_wl"] >= min(stellar_grid.wavelength_range)) \
                    & (sed["sed_wl"] <= max(stellar_grid.wavelength_range))
        sed = sed[model_mask]

        fig = plot_fitted_model(sed=sed,
                                # Only nominals of final fitted parameters are needed
                                theta=(9847.723, 9569.339, 4.227, 4.253, 1.842, 1.739, 336.073, 0.320),
                                model_grid=stellar_grid,
                                sed_flux_colname="sed_flux",
                                sed_flux_err_colname="sed_eflux",
                                sed_filter_colname="sed_filter",
                                sed_lambda_colname="sed_wl",
                                show_component_spectra=False,
                                show_combined_spectrum=True,
                                show_combined_fit=False,
                                show_legend=False,
                                show_grid=False,
                                figsize=(6, 4))
        # fig.savefig("./drop/cw-cma-fitted-sed.pdf")
        plt.show(block=True)


    @unittest.skip("Comment this out to run this interactive test")
    def test_plot_model_spectra_vary_params(self):
        """ Interactive test for producing plot of the effect of varying params on SED/spectra """
        plot_dir = Path("./drop/vary-spectra/")
        plot_dir.mkdir(parents=True, exist_ok=True)
        stellar_grid = get_stellar_grid("BtSettlGrid", extinction_model=G23(Rv=3.1), use_quick_mode=False)

        for ix, (vary, theta_list, label_format) in enumerate([
            ("teff",    [[t, 4, 1, 50, 0] for t in [20000, 10000, 7000, 5000, 4000, 3000]], "{0} K"),
            ("logg",    [[5000, l, 1, 50, 0] for l in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0]],      "$\\log{{g}}={0}$"),
            ("radius",  [[5000, 4, r, 50, 0] for r in [5, 4, 3, 2, 1, 0.5]],                "${0}\\,{{\\rm R_{{\\odot}}}}$"),
            ("dist",    [[5000, 4, 1, d, 0] for d in [5, 10, 20, 30, 40, 50]],              "{0} pc"),
            ("av",      [[5000, 4, 1, 50, a] for a in [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]],     "$A_V = {0}$"),
        ]):
            theta = np.array(theta_list)
            labels = [label_format.format(v) for v in theta[..., ix]]

            for suffix,             flux_unit,              x_scale,    y_scale in [
                ("log-log",         u.W / u.m**2,           "log",      "log"),
                ("log-log",         u.Jy,                   "log",      "log"),
                ("log-linear",      u.W / u.m**2,           "log",      "linear"),
                ("log-linear",      u.Jy,                   "log",      "linear"),
                ("linear",          u.W / u.m**2,           "linear",   "linear"),
                ("linear",          u.Jy,                   "linear",   "linear"),
            ]:
                fig = plot_model_spectra(theta=theta,
                                         model_grid=stellar_grid,
                                         show_component_spectra=True,
                                         show_combined_spectrum=False,
                                         show_legend=True,
                                         labels=labels,
                                         show_grid=(suffix == ""),
                                         figsize=(6, 3),
                                         num_points=2000,
                                         lam_from=0.3,
                                         lam_to=30,
                                         x_scale=x_scale,
                                         y_scale=y_scale,
                                         plot_flux_unit=flux_unit)

                flux_unit_str = to_file_safe_str(str(flux_unit).replace(" ", ""))
                fig.savefig(plot_dir / f"model-spectra-vary-{vary}-{flux_unit_str}-{suffix}.pdf")
                # plt.show(block=True)
