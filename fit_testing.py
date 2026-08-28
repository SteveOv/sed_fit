#!/usr/bin/env python3
""" Testing SED fitting against known targets """
# pylint: disable=no-member, invalid-name, no-name-in-module
import warnings
from pathlib import Path
import json
import argparse
from datetime import datetime
from contextlib import redirect_stdout
from sys import orig_argv
import traceback

import numpy as np
from numpy.core.records import fromarrays

import corner
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning) # pylint: disable=line-too-long
from uncertainties import ufloat, UFloat, nominal_value as nom_val, std_dev
from uncertainties.unumpy import nominal_values as nom_vals, uarray
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants.iau2015 import M_sun, R_sun, L_sun
from astropy.constants import sigma_sb
from astroquery.gaia import Gaia

from dust_extinction.parameter_averages import G23

from deblib.stellar import log_g

from support.extinction import get_gontcharov_av
from support.sed import get_sed_for_target
from support.plots import plot_sed, plot_fitted_model, plot_hr_diagram, plot_predictions_vs_labels
from support.tee import Tee
from support.utils import to_file_safe_str, format_value, estimate_teff_from_spt

from sed_fit.fitter import minimize_fit, mcmc_fit
from sed_fit.generic_fitter import samples_from_sampler
from sed_fit.stellar_grids import StellarGrid


# Use a non-interactive matplotlib backend to avoid threading errors.
mpl_use("agg")

subs = ["A", "B"] # Fixed at 2 stars
theta_plot_captions = np.array([f"$T_{{\\rm eff,{sub}}} / {{\\rm K}}$" for sub in subs] \
                            +[f"$\\log{{g}}_{{\\rm {sub}}}$" for sub in subs] \
                            +[f"$R_{{\\rm {sub}}} / {{\\rm R_{{\\odot}}}}$" for sub in subs] \
                            +["${\\rm dist} / {\\rm pc}$", "${\\rm A_{V}}$"])
theta_captions = np.array([(f"Teff{sub}", u.K) for sub in subs] \
                        +[(f"logg{sub}", u.dex) for sub in subs] \
                        +[(f"R{sub}", u.Rsun) for sub in subs] \
                        +[("dist", u.pc), ("av", u.dimensionless_unscaled)])
result_columns = np.array([t for t, _ in theta_captions])
label_columns = np.array([t for t, _ in theta_captions] + [f"logL{sub}" for sub in subs])

def print_fitted_params(theta: np.ndarray, fitted_mask: np.ndarray,
                        known_values_dict: dict, labels: np.ndarray=theta_captions):
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
    # use_quick_mode affects the StellarGrid flux calculations with cached filter fluxex (True)
    ap.set_defaults(targets=[], mcmc_off=False, overwrite=False, use_quick_mode=True,
                    use_av_override=False)
    args = ap.parse_args()
    run_details = f"{datetime.now():%Y-%m-%d %H:%M:%S%z %Z} $ {' '.join(orig_argv)}"

    # Summarise what is to be fitted and how. Slices are useful shortcut to subset of flags/theta.
    # fit_(Teff|loggs|radii) controls whether fixed at known vals (0), fitted to known vals (1)
    #   or free and constrained by ratio (2)
    #              fit:  Teffs   loggs   radii   dist  Av
    fit_flags = np.array([2]*2 + [0]*2 + [2]*2 + [1] + [1], dtype=int)
    fit_slices = np.array([slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 7), slice(7, 8)])

    drop_dir = Path.cwd() / "drop/testing"
    figs_dir = drop_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    log_file = drop_dir / f"{Path(ap.prog).stem}.log"
    if args.overwrite:
        log_file.unlink(missing_ok=True)
    lbl_csv = drop_dir / "labels.csv"
    fit_csv, mcmc_csv = drop_dir / "min-results.csv", drop_dir / "mcmc-results.csv"

    with redirect_stdout(Tee(open(log_file, "a", encoding="utf8"))) as log:
        print("\n============================================================")
        print(f"Started {ap.prog} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
        print(f"Command: {' '.join(orig_argv)}")
        print(f"Directory for data, logs & plots: {drop_dir}\n", flush=True)

        # Set up the CSV files that will hold the results
        for csv, cols, hd_fmt in [(lbl_csv, label_columns, r"{0},{0}_err"),
                                    (fit_csv, result_columns, r"{0}"),
                                    (mcmc_csv,  result_columns, r"{0},{0}_err")]:
            if not "mcmc" in csv.name or not args.mcmc_off:
                with open(csv, mode=("w" if args.overwrite else "a"), encoding="utf8") as f:
                    # OK to append headers as they're comments and should be ignored if in the body
                    f.write("#target," + ",".join(hd_fmt.format(c) for c in cols) + "\n")
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
                plots_dir = drop_dir / "figs" / to_file_safe_str(target)
                plots_dir.mkdir(parents=True, exist_ok=True)



                # Create any missing config values
                config.setdefault("search_term", target)
                config.setdefault("Teff_sys",nom_val(estimate_teff_from_spt(config.get("SpT","F"))))
                config.setdefault("logg_sys", 4.0)
                for sub in subs:
                    if f"logg{sub}" not in config:
                        logg = log_g(
                            ufloat(config[f"M{sub}"], config.get(f"M{sub}_err", 0)) * M_sun.value,
                            ufloat(config[f"R{sub}"], config.get(f"R{sub}_err", 0)) * R_sun.value)
                        config[f"logg{sub}"], config[f"logg{sub}_err"] = nom_val(logg),std_dev(logg)
                for k in ["Teff", "logg"]:
                    if f"{k}R" not in config:
                        nomA, nomB = config[f"{k}A"], config[f"{k}B"]
                        ratio = ufloat(nomB, config.get(f"{k}B_err", None) or nomB * 0.05) \
                                / ufloat(nomA, config.get(f"{k}A_err", None) or nomA * 0.05)
                        config[f"{k}R"], config[f"{k}R_err"] = nom_val(ratio), std_dev(ratio)

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

                # Get the distance and store it as a label
                dist = 1000 / ufloat(config["parallax"], config.get("parallax_err", None) or 0) # pc
                config.setdefault("dist", dist.n)
                config.setdefault("dist_err", dist.s)



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
                for k in ["TeffR", "loggR", "k"]:
                    ratio_priors += [None, ufloat(config[k], config[f"{k}_err"])]

                value_priors = [ufloat(config[f"{k}{sub}"], config.get(f"{k}{sub}_err", None) or 0)
                                                    for k in ["Teff", "logg", "R"] for sub in subs]
                value_priors += [dist]
                if "av_override" in config and args.use_av_override:
                    value_priors += [ufloat(config["av_override"], 0.05)]
                else:
                    if "av" in config:
                        value_priors += [ufloat(config["av"], config["av_err"])]
                    elif "ebv" in config:
                        value_priors += [ufloat(config["ebv"], config["ebv_err"]) * ext_model.Rv]
                    else:
                        coords = SkyCoord(ra=config["ra"] * u.deg, dec=config["dec"] * u.deg,
                                         distance=1000 / config["parallax"] * u.pc, frame="icrs")
                        value_priors += [Av:=ufloat(get_gontcharov_av(coords)[0],0.04*ext_model.Rv)]
                        print(f"\nAv from the Gontcharov extinction map & target coords: {Av:.3f}")
                    config.setdefault("av", value_priors[-1].n)
                    config.setdefault("av_err", value_priors[-1].s)

                for ix, vp in enumerate(value_priors): # Set any missing uncertainties to 5%
                    if not std_dev(vp):
                        value_priors[ix] = ufloat(nom_val(vp), nom_val(vp) * 0.05)

                # Print out the chosen prior values
                msg = ""
                for k, sl in zip(["Teff", "logg", "rad", "dist", "av"], fit_slices):
                    if all(fit_flags[sl] == 2):
                        msg += f"{k}R={ratio_priors[sl.stop-1]:.3f}, "
                    elif all(fit_flags[sl] == 1):
                        if sl.stop - sl.start == 1:
                            msg += f"{k}={value_priors[sl.start]:.3f}, "
                        else:
                            for i, vp in enumerate(value_priors[sl]):
                                msg += f"{k}{i+1}={vp:.3f}, "
                print("\nPriors:", msg.rstrip(", "))

                def ln_prior_func(theta: np.ndarray[float]) -> float:
                    """ fitting prior callback function to evaluate the current candidate theta """
                    # pylint: disable=cell-var-from-loop
                    if not all(teff_limits[0] <= t <= teff_limits[1] for t in theta[0:2]) \
                        or not all(logg_limits[0] <= l <= logg_limits[1] for l in theta[2:4]) \
                        or not all(radius_limits[0] <= r <= radius_limits[1] for r in theta[4:6]) \
                        or not 0 < theta[-2] \
                        or not av_limits[0] <= theta[-1] <= av_limits[1]:
                        return -np.inf

                    # fit_flags: 0|False fixed so no prior, 1|True val+/-err prior, 2 ratio prior
                    ret_val = 0
                    for ti, fit_flag in enumerate(fit_flags):
                        if fit_flag == 1:
                            # Simple sigma constraint
                            ret_val += ((theta[ti] - value_priors[ti].n) / value_priors[ti].s)**2
                        elif fit_flag == 2 and (ratio_prior := ratio_priors[ti]):
                            # Ratio constraint. Ignore primary which has a ratio prior of None
                            ret_val += ((theta[ti]/theta[ti-1] - ratio_prior.n) / ratio_prior.s)**2
                    return -0.5 * ret_val



                # Save the labels to a csv.
                with lbl_csv.open("a", encoding="utf8") as f:
                    f.write(f"{target}," + ",".join(f"{config[c]:.6e},{config.get(f'{c}_err', None) or 0:.6e}" for c in label_columns) +"\n") # pylint: disable=line-too-long



                # Initial Teffs, loggs & radii. If free fit, we use sys val modified by ratio priors
                fit_mask = fit_flags > 0
                theta0 = nom_vals(value_priors)
                for k, sl in zip(["Teff", "logg", "R"], fit_slices[:3]):
                    if all(fit_flags[sl] == 2):
                        ratio = nom_val(ratio_priors[sl.stop-1])
                        t0 = t0 = config["Teff_sys"] / 5500 if k == "R" else config[f"{k}_sys"]
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
                print_fitted_params(theta_fit, fit_mask, known_values_dict=config)

                fig = plot_fitted_model(sed, theta_fit, stellar_grid, sed_flux_colname="sed_flux",
                                        show_combined_spectrum=True, show_component_spectra=False,
                                        show_grid=True, title=f"{target} SED and fitted model")
                fig.savefig(plots_dir / "sed-min-fitted.pdf")
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
                                               ln_prior_func=ln_prior_func,
                                               nwalkers=nwalkers, nsteps=100000, processes=8,
                                               thin_by=thin_by, early_stopping=True,
                                               progress=True, verbose=True)

                print("Fitted params from MCMC sampling. Those marked * were fitted.",
                    "Known/published values are in brackets.")
                print_fitted_params(theta_mcmc, fit_mask, known_values_dict=config)

                fig = plot_fitted_model(sed, theta_mcmc, stellar_grid, sed_flux_colname="sed_flux",
                                        show_combined_spectrum=True, show_component_spectra=False,
                                        show_grid=True,title=f"{target} SED and MCMC sampled model")
                fig.savefig(plots_dir / "sed-mcmc-sampled.pdf")
                plt.close(fig)

                samples = samples_from_sampler(sampler, flat=True)
                fig = corner.corner(samples, show_titles=True, plot_datapoints=True,
                                    quantiles=[0.16, 0.5, 0.84],
                                    labels=theta_plot_captions[fit_mask],
                                    truths=nom_vals(theta_mcmc[fit_mask]))
                fig.savefig(plots_dir / "sed-mcmc-corner.pdf")
                plt.close(fig)

                # Save the results
                with mcmc_csv.open("a", encoding="utf8") as f:
                    f.write(f"{target}," +",".join(f"{t.n:.6e},{t.s:.6e}" for t in theta_mcmc)+"\n")


            except Exception as exc: # pylint: disable=broad-exception-caught
                print(f"\n*** Failed to fit {target} with the following error... ***")
                traceback.print_exception(exc, file=log)

            log.flush()


        # PLOTS
        raw = np.genfromtxt(lbl_csv, dtype=None, names=True, delimiter=",", encoding="utf8")
        lbl_vals = fromarrays(uarray(nominal_values=[raw[c] for c in label_columns],
                                     std_devs=[raw[f"{c}_err"] for c in label_columns]),
                            dtype=[(c, UFloat.dtype) for c in label_columns])

        fit_vals = np.genfromtxt(fit_csv, dtype=None, names=True, delimiter=",", encoding="utf8")

        mcmc_vals = None
        if not args.mcmc_off:
            raw = np.genfromtxt(mcmc_csv, dtype=None, names=True, delimiter=",", encoding="utf8")
            mcmc_vals = fromarrays(uarray(nominal_values=[raw[c] for c in result_columns],
                                        std_devs=[raw[f"{c}_err"] for c in result_columns]),
                                dtype=[(c, UFloat.dtype) for c in result_columns])

        print()
        for vals, name, msg in [(fit_vals, "min", "fitting results"),
                                (mcmc_vals, "mcmc", "MCMC sampling results"),
                                (lbl_vals, "labels", "label values")]:
            if vals is not None:
                print(f"Creating a H-R plot of the target's {msg}")
                lums = None
                if "LogLA" in vals.dtype.names:
                    lums = 10**np.array([vals["logLA"], vals["logLA"]])
                else:
                    teffs = np.array([vals["TeffA"], vals["TeffB"]])
                    rads = np.array([vals["RA"], vals["RB"]])
                    lums = ((4 * np.pi * (rads * R_sun)**2 * sigma_sb * teffs**4) / L_sun).value
                fig = plot_hr_diagram(teffs, lums, labels=["star A", "star B"],
                                      plot_zams=True, legend_loc="best", invertx=True,
                                      xlim=(28e3, 2.6e3), ylim=(1e-3, 2.2e4))
                fig.savefig(figs_dir / f"h-r-{name}.pdf")
                plt.close(fig)

            if vals is not None and name != "labels":
                print(f"Creating a result-vs-labels plot of the target's {msg}")
                plot_columns = ["TeffA", "TeffB", "RA", "RB"]
                plot_captions = np.array([p for c, p in zip(result_columns, theta_plot_captions)
                                                                            if c in plot_columns])
                fig = plot_predictions_vs_labels(vals[plot_columns], lbl_vals[plot_columns],
                                                 captions=plot_captions, cols=4)
                fig.savefig(figs_dir / f"results-vs-labels-{name}.pdf")
                plt.close(fig)


        print("\n\n============================================================")
        print(f"Completed {ap.prog} at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}")
        print("============================================================")
