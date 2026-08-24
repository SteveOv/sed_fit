#!/usr/bin/env python3
""" Testing SED fitting against known targets """
# pylint: disable=no-member, invalid-name
import warnings
from pathlib import Path
import json
import argparse
from datetime import datetime
from contextlib import redirect_stdout
from inspect import getsourcefile
import traceback

import numpy as np

import corner
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning) # pylint: disable=line-too-long
from uncertainties import ufloat, nominal_value as nom_val, std_dev
from uncertainties.unumpy import nominal_values as nom_vals
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants.iau2015 import M_sun, R_sun, L_sun
from astropy.constants import sigma_sb
from astroquery.gaia import Gaia

from dust_extinction.parameter_averages import G23

from deblib.stellar import log_g

from support.extinction import get_gontcharov_av
from support.sed import get_sed_for_target
from support.plots import plot_sed, plot_fitted_model, plot_hr_diagram
from support.tee import Tee
from support.utils import to_file_safe_str, format_value, estimate_teff_from_spt

from sed_fit.fitter import create_theta, minimize_fit, mcmc_fit
from sed_fit.generic_fitter import samples_from_sampler
from sed_fit.stellar_grids import StellarGrid

THIS_STEM = Path(getsourcefile(lambda: 0)).stem

# Use a non-interactive matplotlib backend to avoid threading errors (issue #36).
mpl_use("agg")

# Affects the StellarGrid flux calculations
use_quick_mode = True

# Controls whether logg is fixed at known values (0), fitted to known values (1) or free fitted (2)
fit_logg = 0

# TODO: Controls whether we fit Av or we deredden the fluxes prior to fitting and fix Av at zero
fit_av = True

theta_plot_labels = np.array([f"$T_{{\\rm eff,{st+1}}} / {{\\rm K}}$" for st in range(2)] \
                            +[f"$\\log{{g}}_{{\\rm {st+1}}}$" for st in range(2)] \
                            +[f"$R_{{\\rm {st+1}}} / {{\\rm R_{{\\odot}}}}$" for st in range(2)] \
                            +["${\\rm dist} / {\\rm pc}$", "${\\rm A_{V}}$"])
theta_labels = np.array([(f"Teff{st+1}", u.K) for st in range(2)] \
                        +[(f"logg{st+1}", u.dex) for st in range(2)] \
                        +[(f"R{st+1}", u.Rsun) for st in range(2)] \
                        +[("dist", u.pc), ("av", u.dimensionless_unscaled)])


def print_fitted_params(theta: np.ndarray, fitted_mask: np.ndarray,
                        known_values_dict: dict, labels: np.ndarray=theta_labels):
    """ Pretty printer for fitted params in rows with known values in brackets."""
    for (param, unit), val, fitted in zip(labels, theta, fitted_mask):
        kval = None
        if param in known_values_dict:
            kval, kerr = known_values_dict[param], known_values_dict.get(f"{param}_err", None)
            if kerr is not None:
                kval = ufloat(kval, kerr)
        print(f"{param:>12s}{'*' if fitted else ' '} =", format_value(val, unit, kval))
    if "source" in known_values_dict:
        print("Source(s) of known values:", known_values_dict["source"])
    if "parallax_bibcode" in known_values_dict:
        print("Source of parallax/distance:", known_values_dict["parallax_bibcode"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The sed_fit esting module for known targets.")
    ap.add_argument("-t", "--targets", dest="targets", type=str, required=False, nargs="+",
                    help="specific target from the targets file to be fitted (overrides exclude)")
    ap.add_argument("-mo", "--mcmc-off", dest="mcmc_off", action="store_true", required=False,
                    help="suppress running of MCMC for parameters")
    ap.add_argument ("-o", "--overwrite", dest="overwrite", action="store_true", required=False,
                     help="force overwrite of existing log and csv files (otherwise append)")
    ap.set_defaults(targets=[], mcmc_off=False, overwrite=False)
    args = ap.parse_args()

    drop_dir = Path.cwd() / "drop/testing"
    drop_dir.mkdir(parents=True, exist_ok=True)
    log_file = drop_dir / f"{THIS_STEM}.log"
    if args.overwrite:
        log_file.unlink(missing_ok=True)

    with redirect_stdout(Tee(open(log_file, "a", encoding="utf8"))) as log:
        print("\n============================================================")
        print(f"Started {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"Directory for data, logs & plots: {drop_dir}\n", flush=True)

        # Set up the CSV files that will hold the results
        fit_csv, mcmc_csv = drop_dir / "min-thetas.csv", drop_dir / "mcmc-thetas.csv"
        if args.overwrite:
            fit_csv.unlink(missing_ok=True)
            mcmc_csv.unlink(missing_ok=True)
        if not fit_csv.exists():
            fit_csv.write_text("#target," +",".join(l for  l,_ in theta_labels) +"\n")
        if not mcmc_csv.exists():
            mcmc_csv.write_text("#target," +",".join(f"{l},{l}_err" for  l,_ in theta_labels) +"\n")

        # Extinction model: G23 (Gordon et al., 2023) Milky Way R(V) filter gives broadest coverage
        ext_model = G23(Rv=3.1)
        ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/um
        print(f"Using the {ext_model.__class__.__name__} extinction model covers the range from",
            f"{min(ext_wl_range):unicode} to {max(ext_wl_range):unicode}.")

        # BtSettlGrid & KuruczGrid available. The former has better coverage but is slower/larger.
        stellar_grid = StellarGrid.get_instance("BtSettlGrid", extinction_model=ext_model,
                                                use_quick_mode=use_quick_mode,
                                                interp_method="slinear", verbose=True)
        print(f"Loaded {stellar_grid.__class__.__name__} covering the ranges:")
        print(f"wavelength {stellar_grid.wavelength_range * stellar_grid.wavelength_unit:unicode},",
                f"Teff {stellar_grid.teff_range * stellar_grid.teff_unit:unicode},",
                f"logg {stellar_grid.logg_range * stellar_grid.logg_unit:unicode}",
                f"\nand metallicity {stellar_grid.metal_range * u.dimensionless_unscaled:unicode},",
                f"with fluxes returned in units of {stellar_grid.flux_unit:unicode}")

        # Fixed limit priors
        teff_limits = stellar_grid.teff_range
        logg_limits = stellar_grid.logg_range
        radius_limits = (0.1, 100)
        av_limits = (-1, 5)

        # Get the targets' configurations
        targets_config_file = Path.cwd() / "config" / "fitting-a-sed-targets.json"
        with open(targets_config_file, mode="r", encoding="utf8") as f:
            full_dict = json.load(f)
            if args.targets is not None and len(args.targets) > 0:
                targets_cfg = { k: full_dict[k] for k in args.targets if k in full_dict }
            else:
                targets_cfg = { k: c for k, c in full_dict.items() if not c.get("exclude", False) }
        targets_count = len(list(targets_cfg.keys()))

        for tix, (target, config) in enumerate(targets_cfg.items(), start=1):
            try:
                print("\n\n------------------------------------------------------------")
                print(f"Processing target {tix} of {targets_count}: {target}")
                print("------------------------------------------------------------", flush=True)


                # Create any missing config values
                config.setdefault("search_term", target)
                config.setdefault("Teff_sys",nom_val(estimate_teff_from_spt(config.get("SpT","F"))))
                config.setdefault("logg_sys", 4.0)
                for ix in [1, 2]:
                    if f"logg{ix}" not in config:
                        logg = log_g(
                            ufloat(config[f"M{ix}"], config.get(f"M{ix}_err", 0)) * M_sun.value,
                            ufloat(config[f"R{ix}"], config.get(f"R{ix}_err", 0)) * R_sun.value)
                        config[f"logg{ix}"], config[f"logg{ix}_err"] = nom_val(logg), std_dev(logg)
                for k in ["Teff", "logg"]:
                    if f"{k}R" not in config:
                        nom1, nom2 = config[f"{k}1"], config[f"{k}2"]
                        ratio = ufloat(nom2, config.get(f"{k}2_err", None) or nom2 * 0.05) \
                                / ufloat(nom1, config.get(f"{k}1_err", None) or nom1 * 0.05)
                        config[f"{k}R"], config[f"{k}R_err"] = nom_val(ratio), std_dev(ratio)

                figs_dir = drop_dir / "figs" / to_file_safe_str(target)
                figs_dir.mkdir(parents=True, exist_ok=True)

                if not all(k in config for k in ["ruwe", "ra", "dec", "parallax"]):
                    print(f"Querying Gaia DR3 for coordinates and ruwe of {target}")
                    dr3_id = int(config["gaia_dr3_id"])
                    if _tbl := Gaia.launch_job("SELECT TOP 1 * FROM gaiadr3.gaia_source_lite " \
                                            + f"WHERE source_id = {dr3_id}").get_results():
                        config["ruwe"] = _tbl["ruwe"][0]
                        config["ra"] = _tbl["ra"][0]                            # deg
                        config["dec"] = _tbl["dec"][0]                          # deg
                        config["parallax"] = _tbl["parallax"][0]
                        config["parallax_err"] = _tbl["parallax_error"][0]


                # Read the SED for target, de-duplicate then apply any range and exclusion filters.
                print(flush=True)
                sed = get_sed_for_target(target, config["search_term"], radius=0.25,
                                        remove_duplicates=True, verbose=True)

                smask = np.ones((len(sed)), dtype=bool)
                smask &= stellar_grid.has_filter(sed["sed_filter"])
                smask &= ~np.isin(sed["sed_filter"], config.get("sed_filter_exclusions",[]))
                smask &= ~np.isin(sed["_tabname"], config.get("sed_tabname_exclusions", []))
                smask &= (sed["sed_wl"] >= min(stellar_grid.wavelength_range)) \
                            & (sed["sed_wl"] <= max(stellar_grid.wavelength_range))
                sed = sed[smask]

                sed.sort(["sed_wl"])
                print(f"{len(sed)} unique SED observation(s) remain after range & exclusion",
                    "filtering. \nThe units for flux density, frequency and wavelength are:",
                    ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux", "sed_freq", "sed_wl"]))

                fig = plot_sed(sed["sed_wl"].quantity, sed["sed_flux"].quantity,
                            sed["sed_eflux"].quantity, fmts=[".r"], labels=["observed"],
                            show_grid=True, title=target + " SED data")
                fig.savefig(figs_dir / "sed-observations.pdf")
                plt.close(fig)


                # Set up the priors and the ln_prior_func callback
                TeffR_prior = ufloat(config["TeffR"], config["TeffR_err"])
                radR_prior = ufloat(config["k"], config["k_err"])
                loggR_prior = ufloat(config["loggR"], config["loggR_err"])
                logg_priors = [
                    ufloat((n := config["logg1"]), config.get("logg1_err", None) or n * 0.05),
                    ufloat((n := config["logg2"]), config.get("logg2_err", None) or n * 0.05)
                ]
                dist_prior =  1000 / ufloat(config["parallax"], config["parallax_err"])
                if "Av" in config:
                    av_prior = ufloat(config["av"], config["av_err"])
                elif "ebv" in config:
                    av_prior = ufloat(config["ebv"], config["ebv_err"]) * ext_model.Rv
                else:
                    coords = SkyCoord(ra=config["ra"] * u.deg, dec=config["dec"] * u.deg,
                                    distance=1000 / config["parallax"] * u.pc, frame="icrs")
                    av_prior = ufloat(get_gontcharov_av(coords)[0], 0.04 * ext_model.Rv)
                    print(f"\nAv from the Gontcharov extinction map & target coord: {av_prior:.3f}")

                def ln_prior_func(theta: np.ndarray[float]) -> float:
                    """ fitting prior callback function to evaluate the current candidate theta """
                    # pylint: disable=cell-var-from-loop
                    Teffs, loggs, radii = theta[0:2], theta[2:4], theta[4:6]
                    dist, av = theta[-2], theta[-1]

                    if not all(teff_limits[0] <= t <= teff_limits[1] for t in Teffs) or \
                            not all(logg_limits[0] <= l <= logg_limits[1] for l in loggs) or \
                            not all(radius_limits[0] <= r <= radius_limits[1] for r in radii) or \
                            not 0 < dist or \
                            not av_limits[0] <= av <= av_limits[1]:
                        return -np.inf

                    ret_val = 0
                    ret_val += ((Teffs[1]/Teffs[0] - TeffR_prior.n) / TeffR_prior.s)**2
                    ret_val += ((radii[1]/radii[0] - radR_prior.n) / radR_prior.s)**2
                    if fit_logg == 1:   # Fitted to known values constrained by uncertainties
                        ret_val += ((loggs[0] - logg_priors[0].n) / logg_priors[0].s)**2
                        ret_val += ((loggs[1] - logg_priors[1].n) / logg_priors[1].s)**2
                    elif fit_logg == 2: # Fully free fitted, only constrained by a ratio (as Teffs)
                        ret_val += ((loggs[1]/loggs[0] - loggR_prior.n) / loggR_prior.s)**2
                    ret_val += ((dist - dist_prior.n) / dist_prior.s)**2
                    if fit_av:
                        ret_val += ((av - av_prior.n) / av_prior.s)**2
                    return -0.5 * ret_val

                logg_msg = ""
                if fit_logg == 1:
                    logg_msg = f"logg1={logg_priors[0]:.3f}, logg2={logg_priors[1]:.3f},"
                elif fit_logg == 2:
                    logg_msg = f"loggR={loggR_prior:.3f},"
                print(f"\nPriors: TeffR={TeffR_prior:.3f}, radR={radR_prior:.3f}," + logg_msg,
                      f"dist={dist_prior:.3f} [pc], av={av_prior:.3f}.")


                # Initial Teffs, loggs & radii, modified by the ratio priors so they meet criteria
                t0_Teffs = [config["Teff_sys"]] * 2
                t0_loggs = [config["logg_sys"]] * 2
                t0_radii = [t0_Teffs[0] / 5500] * 2
                for t0, ratio in [
                    (t0_Teffs, nom_val(TeffR_prior)),
                    (t0_loggs, nom_val(loggR_prior)),
                    (t0_radii, nom_val(radR_prior)),
                ]:
                    if ratio < 1:
                        t0[1:] = [t * ratio for t in t0[1:]]
                    else:
                        t0[0] /= ratio


                # Set up the initial fitting position (theta0)
                if fit_logg < 2: # If not free fitting logg override initial loggs to known values
                    t0_loggs = [config["logg1"], config["logg2"]]
                fit_mask = np.ones(shape=(8,), dtype=bool)
                if fit_logg == 0:
                    fit_mask[2:4] = False
                theta0 = create_theta(teffs=t0_Teffs, loggs=t0_loggs, radii=t0_radii,
                                      dist=nom_val(dist_prior), av=nom_val(av_prior),
                                      nstars=2, verbose=False)


                # Prepare the SED data for fitting
                print("\nPreparing SED data for fitting with the fluxes coerced to the units",
                      f"{stellar_grid.flux_unit:unicode} used by {stellar_grid.__class__.__name__}")
                with u.set_enabled_equivalencies(u.spectral() \
                                                + u.spectral_density(sed["sed_freq"].quantity)):
                    x = stellar_grid.get_filter_indices(sed["sed_filter"])
                    y = sed["sed_flux"].quantity.to(stellar_grid.flux_unit).value
                    y_err = sed["sed_eflux"].quantity.to(stellar_grid.flux_unit).value


                # Quick minimize fit
                print(flush=True)
                theta_fit, _ = minimize_fit(x, y, y_err, theta0, fit_mask, stellar_grid,
                                            ln_prior_func, methods=["Nelder-Mead"], verbose=True)
                print("Fitted params from minimize fit. Those marked * were fitted.",
                    "Known/published values are in brackets.")
                print_fitted_params(theta_fit, fit_mask, known_values_dict=config)

                fig = plot_fitted_model(sed, theta_fit, stellar_grid, sed_flux_colname="sed_flux",
                                        show_combined_spectrum=True, show_component_spectra=False,
                                        show_grid=True, title=f"{target} SED and fitted model")
                fig.savefig(figs_dir / "sed-min-fitted.pdf")
                plt.close(fig)

                # Save the results
                with fit_csv.open("a", encoding="utf8") as f:
                    f.write(f"{target}," + ",".join(f"{t:.6e}" for t in theta_fit) + "\n")


                if args.mcmc_off:
                    print("\nSkipping MCMC sampling.")
                    continue


                # Full MCMC sampling
                print(flush=True)
                nwalkers, thin_by = 100, 1
                theta_mcmc, sampler = mcmc_fit(x, y, y_err, theta0, fit_mask, stellar_grid,
                                               ln_prior_func=ln_prior_func, thin_by=thin_by,
                                               nwalkers=nwalkers, nsteps=100000, processes=8,
                                               early_stopping=True, early_stopping_from=10000,
                                               progress=True, verbose=True)

                print("Fitted params from MCMC sampling. Those marked * were fitted.",
                    "Known/published values are in brackets.")
                print_fitted_params(theta_mcmc, fit_mask, known_values_dict=config)

                fig = plot_fitted_model(sed, theta_mcmc, stellar_grid, sed_flux_colname="sed_flux",
                                        show_combined_spectrum=True, show_component_spectra=False,
                                        show_grid=True,title=f"{target} SED and MCMC sampled model")
                fig.savefig(figs_dir / "sed-mcmc-sampled.pdf")
                plt.close(fig)

                samples = samples_from_sampler(sampler, flat=True)
                fig = corner.corner(samples, show_titles=True, plot_datapoints=True,
                                    quantiles=[0.16, 0.5, 0.84], labels=theta_plot_labels[fit_mask],
                                    truths=nom_vals(theta_mcmc[fit_mask]))
                fig.savefig(figs_dir / "sed-mcmc-corner.pdf")
                plt.close(fig)

                # Save the results
                with mcmc_csv.open("a", encoding="utf8") as f:
                    f.write(f"{target}," +",".join(f"{t.n:.6e},{t.s:.6e}" for t in theta_mcmc)+"\n")


            except Exception as exc: # pylint: disable=broad-exception-caught
                print(f"\n*** Failed to fit {target} with the following error... ***")
                traceback.print_exception(exc, file=log)

            log.flush()


        # H-R Plots.
        fit_thetas = np.genfromtxt(fit_csv, dtype=None, names=True, delimiter=",", comments="#")
        Teffs = np.array([fit_thetas["Teff1"], fit_thetas["Teff2"]])
        radii = np.array([fit_thetas["R1"], fit_thetas["R2"]])
        lums = ((4 * np.pi * (radii * R_sun)**2 * sigma_sb * Teffs**4) / L_sun).value
        fig = plot_hr_diagram(Teffs, lums, labels=["star 1", "star 2"],
                              plot_zams=True, legend_loc="best", invertx=True)
        fig.savefig(drop_dir / "figs/h-r-min.pdf")
        plt.close(fig)


        print("\n\n============================================================")
        print(f"Completed {THIS_STEM} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
