#!/usr/bin/env python3
""" Testing SED fitting against known targets """
# pylint: disable=no-member, invalid-name, no-name-in-module
from typing import List
import warnings
from pathlib import Path
from shutil import rmtree
import json
import argparse
from datetime import datetime
from contextlib import redirect_stdout
from sys import orig_argv
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
from astropy.constants.iau2015 import M_sun, R_sun
from astroquery.gaia import Gaia

from dust_extinction.parameter_averages import G23

from deblib.stellar import log_g

from support.extinction import get_gontcharov_av
from support.sed import get_sed_for_target
from support.plots import plot_sed, plot_fitted_model
from support.tee import Tee
from support.utils import to_file_safe_str, format_value, estimate_teff_from_spt

from sed_fit.fitter import minimize_fit, mcmc_fit
from sed_fit.generic_fitter import samples_from_sampler
from sed_fit.stellar_grids import StellarGrid

DELIM = ";"

# Use a non-interactive matplotlib backend to avoid threading errors.
mpl_use("agg")

def subs(num_stars: int=2):
    """ Iterate of subs for the requested number of stars """
    return ("ABCDEFGHIJKLM"[n] for n in range(num_stars))

def theta_plot_captions(num_stars: int=2) -> np.ndarray[str]:
    """ Get the theta  plot captions array for the requested number of stars """
    return np.array([f"$T_{{\\rm eff,{sub}}} / {{\\rm K}}$" for sub in subs(num_stars)] \
                    + [f"$\\log{{g}}_{{\\rm {sub}}}$" for sub in subs(num_stars)] \
                    + [f"$R_{{\\rm {sub}}} / {{\\rm R_{{\\odot}}}}$" for sub in subs(num_stars)] \
                    + ["${\\rm dist} / {\\rm pc}$", "${\\rm A_{V}}$"])

def theta_captions(num_stars: int=2) -> np.ndarray[tuple[str, u.UnitBase]]:
    """ Get the theta captions array for the requested number of stars """
    return np.array([(f"Teff{sub}", u.K) for sub in subs(num_stars)] \
                    + [(f"logg{sub}", u.dex) for sub in subs(num_stars)] \
                    + [(f"R{sub}", u.Rsun) for sub in subs(num_stars)] \
                    + [("dist", u.pc), ("av", u.dimensionless_unscaled)])

def result_colnames(num_stars: int=2, label_cols: bool=False) -> List[str]:
    """ Get the csv column names for the requested number of stars """
    ret_cols = [t for t, _ in theta_captions(num_stars)]
    if label_cols:
        return ret_cols + [f"logL{sub}" for sub in subs(num_stars)]
    return ret_cols

def print_fitted_params(theta: np.ndarray, fitted_mask: np.ndarray,
                        known_values_dict: dict, labels: np.ndarray):
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
    if "dist_bibcode" in known_values_dict:
        print("Source of distance:", known_values_dict["dist_bibcode"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The sed_fit testing module for known targets.")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of the targets to fit")
    ap.add_argument("-t", "--targets", dest="targets", type=str, required=False, nargs="+",
                    help="specific target from the targets file to be fitted (overrides exclude)")
    ap.add_argument("-mo", "--mcmc-off", dest="mcmc_off", action="store_true", required=False,
                    help="suppress running of MCMC for parameters")
    ap.add_argument ("-qo", "--quick-off", dest="use_quick_mode", action="store_false",
                     required=False, help="suppress the use of model grid's quick_mode")
    ap.add_argument("-mp", "--mcmc-processes", dest="mcmc_processes", type=int, required=False,
                    help="The number of processes on which to spread MCMC sampling,"
                            + " which defaults to the number of available cores if not set")
    # use_quick_mode affects the StellarGrid flux calculations with cached filter fluxex (True)
    ap.set_defaults(targets=[], mcmc_off=False, overwrite=False, use_quick_mode=True,
                    mcmc_processes=None, use_av_override=False)
    args = ap.parse_args()

    # Work-around for potential issue on MacOS. Try enabling this if you get failures with a
    # message similar to 'NameError: name '_fixed_theta' is not defined' when running the MCMC.
    # from multiprocessing import set_start_method
    # set_start_method("fork", force=True)

    # Get the targets' configurations
    with open(args.targets_file, mode="r", encoding="utf8") as f:
        full_dict = json.load(f)
        if args.targets is not None and len(args.targets) > 0:
            targets_cfg = { k: full_dict[k] for k in args.targets if k in full_dict }
        else:
            targets_cfg = { k: c for k, c in full_dict.items() if not c.get("exclude", False) }
    targets_count = len(list(targets_cfg.keys()))

    # Set up the output directories
    out_dir = Path.cwd() / f"drop/{args.targets_file.stem}"
    log_file = out_dir / f"{Path(ap.prog).stem}.log"
    if log_file.exists():
        response = input("\nFiles exist for this test config. Clear down files y/N? ")
        if response.strip().lower() in ["y", "yes"]:
            for file in out_dir.glob("*"):
                if file.is_dir():
                    rmtree(file, ignore_errors=True)
                else:
                    file.unlink(missing_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    lbl_csv = out_dir / "labels.csv"
    fit_csv, mcmc_csv = out_dir / "min-results.csv", out_dir / "mcmc-results.csv"

    with redirect_stdout(Tee(open(log_file, "a", encoding="utf8"))) as log:
        print("\n============================================================")
        print(f"Started {ap.prog} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"Command: {' '.join(orig_argv)}")
        print(f"Directory for data, logs & plots: {out_dir}\n", flush=True)

        # Summarise how the priors are handled for each theta item. Each flag controls whether the
        # corresponding value is fixed at known val & no prior (0), constrained by a Gaussian prior
        # for a known "truth" value, or fully free and constrained by Gaussian prior on a ratio (2).
        #                for:  Teffs        loggs        radii        dist  Av
        nstars = 2 # TODO: can only support 2 stars until ratio handling updated
        prior_flags = np.array([2]*nstars + [1]*nstars + [2]*nstars + [1] + [1], dtype=int)
        # These slices are useful shortcut to subset of fit|prior flags and theta.
        theta_slices = np.array([slice(0,nstars), slice(nstars,2*nstars), slice(2*nstars,3*nstars),
                                 slice(3*nstars,3*nstars+1), slice(3*nstars+1,3*nstars+2)])

        # Set up the CSV files that will hold the results
        run_details = f"{datetime.now():%Y-%m-%d %H:%M:%S%z %Z} $ {' '.join(orig_argv)}"
        for csv, cols, hfmt in [(lbl_csv, result_colnames(nstars, label_cols=True), r"{0}"),
                                (fit_csv, result_colnames(nstars), r"{0}"),
                                (mcmc_csv,  result_colnames(nstars), r"{0}")]:
            if not "mcmc" in csv.name or not args.mcmc_off:
                with open(csv, mode="a", encoding="utf8") as f:
                    # OK to append headers as they're comments and should be ignored if in the body
                    f.write("# target" + DELIM + DELIM.join(hfmt.format(c) for c in cols) + "\n")
                    f.write("# " + run_details + "\n")

        # Extinction model: G23 (Gordon et al., 2023) Milky Way R(V) filter gives broadest coverage
        ext_model = G23(Rv=3.1)
        ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/um
        print(f"Using the {ext_model.__class__.__name__} extinction model covers the range from",
            f"{min(ext_wl_range):unicode} to {max(ext_wl_range):unicode}.")

        # BtSettlGrid & KuruczGrid available. The former has better coverage but is slower/larger.
        stellar_grid = StellarGrid.get_instance("BtSettlGrid", extinction_model=ext_model,
                                                use_quick_mode=args.use_quick_mode,
                                                interp_method="slinear", verbose=True)
        print(f"Loaded {stellar_grid.__class__.__name__} covering the ranges:")
        print(f"wavelength {stellar_grid.wavelength_range * stellar_grid.wavelength_unit:unicode},",
                f"Teff {stellar_grid.teff_range * stellar_grid.teff_unit:unicode},",
                f"logg {stellar_grid.logg_range * stellar_grid.logg_unit:unicode}",
                f"\nand metallicity {stellar_grid.metal_range * u.dimensionless_unscaled:unicode},",
                f"with fluxes returned in units of {stellar_grid.flux_unit:unicode}")

        # Fixed limit priors
        teff_lims = stellar_grid.teff_range
        logg_lims = stellar_grid.logg_range
        rad_lims = (0.1, 100)
        av_lims = (-1, 5)

        for tix, (target, config) in enumerate(targets_cfg.items(), start=1):
            try:
                print("\n\n------------------------------------------------------------")
                print(f"Processing target {tix} of {targets_count}: {target}")
                print("------------------------------------------------------------", flush=True)
                plots_dir = figs_dir / to_file_safe_str(target)
                plots_dir.mkdir(parents=True, exist_ok=True)

                # Create any missing config values
                config.setdefault("search_term", target)
                config.setdefault("Teff_sys",nom_val(estimate_teff_from_spt(config.get("SpT","F"))))
                config.setdefault("logg_sys", 4.0)
                for sub in subs(nstars):
                    if f"logg{sub}" not in config:
                        logg = log_g(
                            ufloat(config[f"M{sub}"], config.get(f"M{sub}_err", 0)) * M_sun.value,
                            ufloat(config[f"R{sub}"], config.get(f"R{sub}_err", 0)) * R_sun.value)
                        config[f"logg{sub}"], config[f"logg{sub}_err"] = nom_val(logg),std_dev(logg)
                for c in ["Teff", "logg"]: # TODO: handle multiple ratios for >2 stars
                    if f"{c}R" not in config:
                        nomA, nomB = config[f"{c}A"], config[f"{c}B"]
                        ratio = ufloat(nomB, config.get(f"{c}B_err", None) or nomB * 0.05) \
                                / ufloat(nomA, config.get(f"{c}A_err", None) or nomA * 0.05)
                        config[f"{c}R"], config[f"{c}R_err"] = nom_val(ratio), std_dev(ratio)

                if not all(c in config for c in ["ruwe", "ra", "dec", "parallax"]):
                    print(f"Querying Gaia DR3 for coordinates and ruwe of {target}")
                    dr3_id = int(config["gaia_dr3_id"])
                    if _tbl := Gaia.launch_job("SELECT TOP 1 * FROM gaiadr3.gaia_source_lite " \
                                            + f"WHERE source_id = {dr3_id}").get_results():
                        config["ruwe"] = _tbl["ruwe"][0]
                        config["ra"] = _tbl["ra"][0]                            # deg
                        config["dec"] = _tbl["dec"][0]                          # deg
                        config["parallax"] = _tbl["parallax"][0]
                        config["parallax_err"] = _tbl["parallax_error"][0]
                        config["parallax_bibcode"] = "2022yCat.1355....0G"

                if "dist" in config:
                    dist = ufloat(config["dist"], config.get("dist_err", None) or 0)
                    print(f"Using distance from config of: {dist:.3f} pc"
                          + f" (bibcode={config.get('dist_bibcode', '')})")
                else:
                    dist = 1000 / ufloat(config["parallax"], config.get("parallax_err", None) or 0)
                    config["dist"], config["dist_err"] = dist.n, dist.s
                    config["dist_bibcode"] = config.get("parallax_bibcode", "")
                    print(f"Using distance from parallax of: {dist:.3f} pc (bibcode="
                          + f"{config.get('dist_bibcode', '')}, ruwe={config.get('ruwe', 0):.3f})")


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
                fig.savefig(plots_dir / "sed-observations.pdf")
                plt.close(fig)



                # Set up the priors and the ln_prior_func.
                # Always setup both ratios & values, reading them from config.
                ratio_priors = []
                for c in ["TeffR", "loggR", "k"]: # TODO: handle multiple ratios for >2 stars
                    ratio_priors += [None, ufloat(config[c], config[f"{c}_err"])]

                value_priors = [ufloat(config[f"{n}{sub}"], config.get(f"{n}{sub}_err", None) or 0)
                                            for n in ["Teff", "logg", "R"] for sub in subs(nstars)]
                value_priors += [dist]
                if "av" in config:
                    value_priors += [ufloat(config["av"], config["av_err"])]
                elif "ebv" in config:
                    value_priors += [ufloat(config["ebv"], config["ebv_err"]) * ext_model.Rv]
                else:
                    coords = SkyCoord(ra=config["ra"] * u.deg, dec=config["dec"] * u.deg,
                                      distance=1000 / config["parallax"] * u.pc, frame="icrs")
                    value_priors += [av := ufloat(get_gontcharov_av(coords)[0], 0.04*ext_model.Rv)]
                    print(f"\nAv from the Gontcharov extinction map & target coords: {av:.3f}")
                config.setdefault("av", value_priors[-1].n)
                config.setdefault("av_err", value_priors[-1].s)

                for ix, vp in enumerate(value_priors): # Set any missing uncertainties to 5%
                    if not std_dev(vp):
                        value_priors[ix] = ufloat(nom_val(vp), nom_val(vp) * 0.05)

                # Print out the chosen prior values. TODO: handle multiple ratios for >2 stars
                msg = ""
                for c, sl in zip(["Teff", "logg", "rad", "dist", "av"], theta_slices):
                    if all(prior_flags[sl] == 2):
                        if sl.stop - sl.start == 1:
                            msg += f"{c}=<free>, "
                        else:
                            msg += f"{c}R={ratio_priors[sl.stop-1]:.3f}, "
                    elif all(prior_flags[sl] == 1):
                        if sl.stop - sl.start == 1:
                            msg += f"{c}={value_priors[sl.start]:.3f}, "
                        else:
                            for vp, sub in zip(value_priors[sl], subs(nstars)):
                                msg += f"{c}{sub}={vp:.3f}, "
                print("\nPriors:", msg.rstrip(", "))

                nratio_priors = len(ratio_priors)
                def ln_prior_func(theta: np.ndarray[float]) -> float:
                    """ fitting prior callback function to evaluate the current candidate theta """
                    # pylint: disable=cell-var-from-loop
                    if not all(teff_lims[0]<= t <=teff_lims[1] for t in theta[theta_slices[0]]) \
                        or not all(logg_lims[0]<= l <=logg_lims[1] for l in theta[theta_slices[1]])\
                        or not all(rad_lims[0]<= r <=rad_lims[1] for r in theta[theta_slices[2]]) \
                        or not 0 < theta[-2] \
                        or not av_lims[0] <= theta[-1] <= av_lims[1]:
                        return -np.inf

                    # prior_flags:  0|False = fixed so no prior
                    #               1|True  = direct Gaussian prior on a "truth" value +/- sigma
                    #               2       = Gaussian prior if there is a ratio value +/- sigma
                    # Use 1 for "truths" with uncertainties and 2 where fitting to find values
                    # TODO: handle multiple ratios for >2 stars
                    ret_val = 0
                    for ti, prior_flag in enumerate(prior_flags):
                        if prior_flag == 1:
                            # Simple constraint, directly treating the known prior value as a truth
                            ret_val += ((theta[ti] - value_priors[ti].n) / value_priors[ti].s)**2
                        elif prior_flag == 2 \
                                and ti < nratio_priors and (ratio_prior := ratio_priors[ti]):
                            # Ratio constraint, ignoring the primary which has no ratio
                            ret_val += ((theta[ti]/theta[ti-1] - ratio_prior.n) / ratio_prior.s)**2
                    return -0.5 * ret_val



                # Save the labels to a csv.
                with lbl_csv.open("a", encoding="utf8") as f:
                    f.write(target + DELIM + DELIM.join(f"{ufloat(config[c], config.get(f'{c}_err', None) or 0):.9e}" for c in result_colnames(nstars, True)) + "\n") # pylint: disable=line-too-long



                # Initial Teffs, loggs & radii. If free fit, we use sys val modified by ratio priors
                # TODO: handle multiple ratios for >2 stars
                fit_mask = prior_flags > 0
                theta0 = nom_vals(value_priors)
                for c, sl in zip(["Teff", "logg", "R"], theta_slices[:3]):
                    if all(prior_flags[sl] == 2):
                        ratio = nom_val(ratio_priors[sl.stop-1])
                        t0 = t0 = config["Teff_sys"] / 5500 if c == "R" else config[f"{c}_sys"]
                        theta0[sl] = (t0, t0*ratio) if ratio < 1 else (t0/ratio, t0)



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
                print_fitted_params(theta_fit, fit_mask, config, theta_captions(nstars))

                fig = plot_fitted_model(sed, theta_fit, stellar_grid, sed_flux_colname="sed_flux",
                                        show_combined_spectrum=True, show_component_spectra=False,
                                        show_grid=True, title=f"{target} SED and fitted model")
                fig.savefig(plots_dir / "sed-min-fitted.pdf")
                plt.close(fig)

                # Save the results
                with fit_csv.open("a", encoding="utf8") as f:
                    f.write(target + DELIM + DELIM.join(f"{t:.9e}" for t in theta_fit) + "\n")


                if args.mcmc_off:
                    print("\nSkipping MCMC sampling.")
                    continue



                # Full MCMC sampling
                print(flush=True)
                nwalkers, thin_by = 100, 1
                theta_mcmc, sampler = mcmc_fit(x, y, y_err, theta0, fit_mask, stellar_grid,
                                               ln_prior_func=ln_prior_func, nwalkers=nwalkers,
                                               nsteps=100000, processes=args.mcmc_processes,
                                               thin_by=thin_by, early_stopping=True,
                                               progress=True, verbose=True)

                print("Fitted params from MCMC sampling. Those marked * were fitted.",
                      "Known/published values are in brackets.")
                print_fitted_params(theta_mcmc, fit_mask, config, theta_captions(nstars))

                fig = plot_fitted_model(sed, theta_mcmc, stellar_grid, sed_flux_colname="sed_flux",
                                        show_combined_spectrum=True, show_component_spectra=False,
                                        show_grid=True,title=f"{target} SED and MCMC sampled model")
                fig.savefig(plots_dir / "sed-mcmc-sampled.pdf")
                plt.close(fig)

                # The value priors were populated with the known values, so can be used as truths
                samples = samples_from_sampler(sampler, flat=True)
                truths = np.array(value_priors)
                for suffix, mask in [("corner", fit_mask), ("corner-free", prior_flags > 1)]:
                    plot_samples = samples[:, mask] # Should be a view so no copy
                    fig = corner.corner(plot_samples, show_titles=True, plot_datapoints=True,
                                        quantiles=[0.16, 0.5, 0.84],
                                        labels=theta_plot_captions(nstars)[mask],
                                        truths=nom_vals(truths[mask]))
                    fig.savefig(plots_dir / f"sed-mcmc-{suffix}.pdf")
                    plt.close(fig)
                    print(f"\nSaved a '{suffix}' plot for",
                          "*".join(f"{s:d}" for s in plot_samples.shape), "samples.")

                # Save the results
                with mcmc_csv.open("a", encoding="utf8") as f:
                    f.write(target + DELIM + DELIM.join(f"{t:.9e}" for t in theta_mcmc) + "\n")


            except Exception as exc: # pylint: disable=broad-exception-caught
                print(f"\n*** Failed to fit {target} with the following error... ***")
                traceback.print_exception(exc, file=log)

            log.flush()


        print("\n\n============================================================")
        print(f"Completed {ap.prog} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
