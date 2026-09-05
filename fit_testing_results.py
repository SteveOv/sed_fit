#!/usr/bin/env python3
""" Plots from fit_testing output """
# pylint: disable=no-member, invalid-name, no-name-in-module
import warnings
from pathlib import Path
import argparse
from io import TextIOBase
from sys import stdout
from itertools import zip_longest

import numpy as np

from matplotlib import use as mpl_use
import matplotlib.pyplot as plt

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning) # pylint: disable=line-too-long
from uncertainties import ufloat_fromstr
from astropy.constants.iau2015 import R_sun, L_sun
from astropy.constants import sigma_sb

from support.plots import plot_hr_diagram, plot_predictions_vs_labels
from fit_testing import theta_plot_captions, theta_captions, subs

# Use a non-interactive matplotlib backend to avoid threading errors.
mpl_use("agg")

DELIM = ";"
def read_result_csv(csv_file: Path):
    """ Read the contents of the requested csv file, returning them as a structured array. """
    values = None
    if not csv_file.exists():
        print(f"The testing csv file '{csv_file.name}' was not found.")
    else:
        print(f"Loading the testing csv file '{csv_file.name}'.")
        # First parse to get column names, required for converters.
        values = np.genfromtxt(csv_file, dtype=None, names=True, delimiter=DELIM, encoding="utf8",
                               max_rows=1)
        cols = list(values.dtype.names)
        # Note, ufloat_fromstr will asign 1 to lsd of std_dev if parsing a number without std dev!
        values = np.genfromtxt(csv_file, dtype=None, names=cols, delimiter=DELIM, encoding="utf8",
                               converters={c: ufloat_fromstr for c in cols if c not in ["target"]})
    return values

def calculate_errors(lab_vals: np.ndarray, res_vals: np.ndarray,
                     relative_errors: bool=False) -> np.ndarray:
    """ Create an array of label-result values. Assumes labels & results same size, shape & order"""
    # Assumes that fit_vals & lbl_vals have same cols and are in the same order
    err_vals = np.zeros_like(res_vals)
    val_cols  = [c for c in res_vals.dtype.names if c not in ["target"]]
    err_vals["target"] = res_vals["target"]
    for c in val_cols:
        err_vals[c] = lab_vals[c] - res_vals[c]
        if relative_errors:
            err_vals[c] /= lab_vals[c] + 1e-21
    return err_vals

def to_results_tex(lab_vals: np.ndarray, res_vals: np.ndarray, to: TextIOBase=stdout,
                   include_errs: bool=False, relative_errs: bool=True, errs_cline: bool=True,
                   endhead: bool=True, replace_pm_with: str="&", num_stars: int=2):
    """ Write latex tabular/logtable rows to the passed TextIOBase """
    row_span, err_vals = 2, [None]
    if include_errs:
        row_span, err_vals = 3, calculate_errors(lab_vals, res_vals, relative_errs)

    out_cols = [f"{k}{sub}" for k in ["Teff", "R"] for sub in subs(num_stars)]
    out_units = ["K"] * num_stars + [r"\Rsun"] * num_stars
    out_fmts = [r"{0:.0fL}", r"{0:.0fL}", r"{0:.3fL}", r"{0:.3fL}"]
    err_fmts = [r"{0:.3fL}", r"{0:.3fL}", r"{0:.3fL}", r"{0:.3fL}"]
    to.write(r"\hline" + "\n")
    to.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\{h}}}" for h in out_cols) + " \\\\ \n")
    to.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{({u})}}" for u in out_units) +" \\\\ \n")
    to.write(r"\hline" + "\n")
    if endhead:
        to.write(r"\endhead" + "\n")
    for lrow, rrow, erow in zip_longest(lab_vals, res_vals, err_vals):
        row_head = f"\\multirow{{{row_span}}}{{*}}{{{rrow['target']}}}"
        for bl_row_num, (row, fmts) in enumerate(zip([lrow, rrow, erow],
                                                    [out_fmts, out_fmts, err_fmts]), 1):
            if row is not None:
                to.write((row_head if bl_row_num == 1 else "") + " & ")
                row = " & ".join(fmt.format(v) for v, fmt in zip(row[out_cols], fmts))
                to.write((str.replace(row, "\\pm", replace_pm_with) if replace_pm_with else row))
                to.write(" \\\\ ")
                if bl_row_num == row_span:
                    to.write("[4pt]")
                elif bl_row_num == 2 and errs_cline and include_errs:
                    to.write(f"\\cline{{2-{1 + len(out_cols)*2}}}")
                to.write("\n")
        to.flush()
    to.write(r"\hline" + "\n")


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
        print(f"\nFound the results for {nstars} star(s) in min-results.csv")

        if mcmc_vals is not None and mcmc_vals.size > 0:
            with open(drop_dir / "mcmc-table-rows.tex", mode="w", encoding="utf8") as f:
                print(f"\nWriting out LaTeX table rows for MCMC results to '{Path(f.name).name}'")
                to_results_tex(lbl_vals, mcmc_vals, to=f, include_errs=True, num_stars=nstars)

        print()
        for vals, name, msg in [(fit_vals, "min", "fitting results"),
                                (mcmc_vals, "mcmc", "MCMC sampling results"),
                                (lbl_vals, "labels", "label values")]:

            # Hertzsprung-Russell diagrams
            if vals is not None:
                print(f"Creating a H-R plot of the target's {msg}")
                lums = None
                if "LogLA" in vals.dtype.names:
                    lums = 10**np.array([vals[f"logL{sub}"] for sub in subs(nstars)])
                else:
                    teffs = np.array([vals[f"Teff{sub}"] for sub in subs(nstars)])
                    rads = np.array([vals[f"R{sub}"] for sub in subs(nstars)])
                    lums = ((4 * np.pi * (rads * R_sun)**2 * sigma_sb * teffs**4) / L_sun).value
                fig = plot_hr_diagram(teffs, lums, [f"star {sub}" for sub in subs(nstars)],
                                      plot_zams=True, legend_loc="best", invertx=True,
                                      xlim=(28e3, 2.6e3), ylim=(1e-3, 2.2e4))
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

                plot_columns = [f"{c}{sub}" for c in ["Teff", "R"] for sub in subs(nstars)]
                plot_captions = np.array([p for (c,_), p in zip(theta_captions(nstars),
                                                                theta_plot_captions(nstars))
                                                                            if c in plot_columns])
                fig = plot_predictions_vs_labels(vals[plot_columns], lbl_vals[plot_columns],
                                                 captions=plot_captions, cols=nstars,
                                                 hl_mask1=hl_mask1, hl_mask2=hl_mask2,
                                                 hl_mask3=hl_mask3, fill_mask=fill_mask)
                fig.savefig(figs_dir / f"results-vs-labels-{name}.pdf")
                plt.close(fig)
