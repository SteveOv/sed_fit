#!/usr/bin/env python3
""" Plots from fit_testing output """
# pylint: disable=no-member, invalid-name, no-name-in-module
import warnings
from pathlib import Path
import argparse

import numpy as np

from matplotlib import use as mpl_use
import matplotlib.pyplot as plt

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning) # pylint: disable=line-too-long
from uncertainties import ufloat
import astropy.units as u
from astropy.constants.iau2015 import R_sun, L_sun
from astropy.constants import sigma_sb

from support.plots import plot_hr_diagram, plot_predictions_vs_labels
from fit_testing import theta_plot_captions, theta_captions

# Use a non-interactive matplotlib backend to avoid threading errors.
mpl_use("agg")


def read_result_csv(csv_file: Path):
    """ Read the contents of the requested csv file, returning them as a structured array. """
    values = None
    if not csv_file.exists():
        print(f"The testing csv file '{csv_file.name}' was not found.")
    else:
        print(f"Loading the testing csv file '{csv_file.name}'.")
        raw = np.genfromtxt(csv_file, dtype=None, names=True, delimiter=",", encoding="utf8")
        # Yet to convince genfromtxt to use a ufloat converter so we're falling back on manual parse
        if "TeffA_err" in raw.dtype.names:
            out_dtype = [(n, object) for n in raw.dtype.names if not n.endswith("_err")]
            rows = (r for r in raw) if raw.size > 1 else [raw]
            temp = [tuple(ufloat(r[c], r[f"{c}_err"]) if f"{c}_err" in r.dtype.names else r[c]
                                                                            for c, _ in out_dtype)
                                                                                for r in rows]
            values = np.array(temp, dtype=out_dtype)
        else: # It's a result file without uncertainties. We can us it as is.
            values = raw
    return values


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Produce plots from one or more of fit_testing runs.")
    ap.add_argument(dest="drop_dirs", type=Path, nargs="*", help="The results directories to read.")
    ap.set_defaults(drop_dirs=[Path.cwd() / "drop/testing/"])
    args = ap.parse_args()

    for drop_dir in args.drop_dirs:
        print(f"\nLooking for testing results in: {drop_dir}")
        figs_dir = drop_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        fit_vals = read_result_csv(drop_dir / "min-results.csv")
        mcmc_vals = read_result_csv(drop_dir / "mcmc-results.csv")
        lbl_vals = read_result_csv(drop_dir / "labels.csv")

        nstars = (len(fit_vals.dtype) - 3) // 3 # ignoring the target, dist & av columns

        print()
        for vals, name, msg in [(fit_vals, "min", "fitting results"),
                                (mcmc_vals, "mcmc", "MCMC sampling results"),
                                (lbl_vals, "labels", "label values")]:

            # Hertzsprung-Russell diagrams
            if vals is not None:
                print(f"Creating a H-R plot of the target's {msg}")
                lums = None
                if "LogLA" in vals.dtype.names:
                    lums = 10**np.array([vals["logLA"], vals["logLA"]])
                else:
                    teffs = np.array([vals["TeffA"], vals["TeffB"]])
                    rads = np.array([vals["RA"], vals["RB"]])
                    lums = ((4 * np.pi * (rads * R_sun)**2 * sigma_sb * teffs**4) / L_sun).value
                fig = plot_hr_diagram(teffs, lums, ["star A", "star B"], True, legend_loc="best",
                                    invertx=True, xlim=(28e3, 2.6e3), ylim=(1e-3, 2.2e4))
                fig.savefig(figs_dir / f"h-r-{name}.pdf")
                plt.close(fig)

            # Plots of result vs label values.
            if vals is not None and name != "labels" and lbl_vals is not None:
                print(f"Creating a result-vs-labels plot of the target's {msg}")
                hl_mask1 = np.isin(vals["target"], ["V539 Ara", "V889 Aql"])    # square
                hl_mask2 = np.isin(vals["target"], ["MU Cas", "V596 Pup"])      # diamond
                hl_mask3 = np.isin(vals["target"], ["not used"])                # pentagon
                fill_mask = np.isin(vals["target"], ["V539 Ara", "MU Cas"])
                plot_columns = ["TeffA", "TeffB", "RA", "RB"]
                plot_captions = np.array([p for (c,_), p in zip(theta_captions(nstars),
                                                                theta_plot_captions(nstars))
                                                                            if c in plot_columns])
                fig = plot_predictions_vs_labels(vals[plot_columns], lbl_vals[plot_columns],
                                                 captions=plot_captions, cols=2,
                                                 hl_mask1=hl_mask1, hl_mask2=hl_mask2,
                                                 hl_mask3=hl_mask3, fill_mask=fill_mask)
                fig.savefig(figs_dir / f"results-vs-labels-{name}.pdf")
                plt.close(fig)
