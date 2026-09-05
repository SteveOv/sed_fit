#!/usr/bin/env python3
""" Compare distances from fit_testing with those from Kervella surface brightness-Teff relation """
# pylint: disable=no-member, invalid-name, no-name-in-module
import warnings
from pathlib import Path
import argparse
import json
from contextlib import redirect_stdout

import numpy as np

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning) # pylint: disable=line-too-long
from uncertainties import ufloat
from astroquery.simbad import Simbad
from deblib.vmath import wrap_func_for_uncertainties

from fit_testing_plots import read_result_csv
from support.tee import Tee
from support.pipeline import deredden, dist_by_brightness_and_teff

# Wrapped to support uncertainties (by perturbing arguments)
wrapped_deredden = wrap_func_for_uncertainties(deredden)
wrapped_kerv = wrap_func_for_uncertainties(dist_by_brightness_and_teff)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare distances from fit_testing with those derived"
                    + " from the Kervella+ (2004A&A...426..297K) surface brightness-Teff relation")
    ap.add_argument(dest="targets_file", type=Path, metavar="TARGETS_FILE",
                    help="json file containing the details of all the targets being testing")
    ap.add_argument("-t", "--targets", dest="targets", type=str, required=False, nargs="+",
                    help="specific targets from the targets file to report on (overrides exclude)")
    ap.add_argument("-d", "--drop-dir", dest="drop_dir", type=Path, required=False,
                    help="The directory with the fit_testing results if not ./drop/TARGETS_FILE")
    ap.set_defaults(targets=[], drop_dir=None)
    args = ap.parse_args()

    # Get the targets' configurations
    with open(args.targets_file, mode="r", encoding="utf8") as f:
        full_dict = json.load(f)
        if args.targets is not None and len(args.targets) > 0:
            targets_cfg = { k: full_dict[k] for k in args.targets if k in full_dict }
        else:
            targets_cfg = { k: c for k, c in full_dict.items() if not c.get("exclude", False) }

    if args.drop_dir is None:
        args.drop_dir = Path.cwd() / f"drop/{args.targets_file.stem}"

    log_file = args.drop_dir / f"{Path(ap.prog).stem}.log"
    with redirect_stdout(Tee(open(log_file, "a", encoding="utf8"))) as log:

        # Read the results of MCMC sampling
        mcmc_vals = read_result_csv(args.drop_dir / "mcmc-results.csv")

        # Slimey hack as the mcmc_vals structured array isn't iterating well when only 1
        targets = [str(mcmc_vals["target"][0])] if len(mcmc_vals) == 1 else mcmc_vals["target"]
        sterm_index = { targets_cfg[t].get("search_term", t): t for t in targets }
        supported_bands = ["U", "B", "V", "R", "I", "J", "H", "K", "L"]

        # Will give us a row for every combination of search term and flux/magnitude observation
        print("\nQuerying SIMBAD for the fluxes of the requested MCMC results.")
        simbad = Simbad()
        simbad.add_votable_fields("flux")
        tbl = simbad.query_objects(list(sterm_index.keys()))
        for sterm in sterm_index.keys():
            config = targets_cfg[sterm_index[sterm]]

            res_row = mcmc_vals[mcmc_vals["target"] == sterm_index[sterm]][0]
            ebv = res_row["av"] / 3.1
            TeffA, TeffB, RA, RB = res_row[["TeffA", "TeffB", "RA", "RB"]]
            print(f"\n{sterm_index[sterm]} has a MCMC derived distance of {res_row['dist']:.6f} pc",
                  f"and E(B-V) of {ebv:.6f} mag (where RV=3.1)")

            print("The distances calculated with the Kervella+ (2004A&A...426..297K)",
                  "surface brightness-Teff relation are:")
            flux_mask = np.array([sterm in r["user_specified_id"] for r in tbl])
            flux_mask &= np.isin(tbl["flux.filter"], supported_bands)
            flux_mask &= tbl["flux.qual"] == "C"
            # flux_mask &= np.isfinite(tbl["flux_err"])

            for flux_row in tbl[flux_mask]:
                band = flux_row["flux.filter"]
                mag = ufloat(flux_row["flux"], flux_row["flux_err"] or 0)
                dmag = wrapped_deredden(ebv=ebv, mag=mag, band=band)
                dist = wrapped_kerv(Teff1=TeffA, Teff2=TeffB, R1=RA, R2=RB, band=band, mag=dmag)
                print(f"dist[mag({band})={mag:.3f}] = {dist:.6f} pc", end="   ")
                print("\t( err contribs:",
                    ", ".join(f"{v.tag}={e/dist.s:.3f}" for v,e in dist.error_components().items()),
                    ")")
