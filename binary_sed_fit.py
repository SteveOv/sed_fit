""" Module fitting SED with binary star models """
# pylint: disable=no-member, line-too-long, wrong-import-position
from pathlib import Path
import json
import re
from warnings import filterwarnings

import numpy as np

filterwarnings("ignore", "Using UFloat objects with std_dev==0 may give unexpected results.", category=UserWarning)
from uncertainties import ufloat

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia

from dust_extinction.parameter_averages import G23

from deblib.constants import M_sun, R_sun
from deblib.stellar import log_g

from support.pipeline import get_teff_from_spt
from support.sed import get_sed_for_target, create_outliers_mask, group_and_average_fluxes
from support.extinction import get_gontcharov_av

from sed_fit.stellar_grids import get_stellar_grid
from sed_fit import fitter

if __name__ == "__main__":
    TARGET = "CM Dra"

    targets_config_file = Path("./config/fitting-a-sed-targets.json")
    with open(targets_config_file, mode="r", encoding="utf8") as f:
        full_dict = json.load(f)
    targets_cfg = { k: full_dict[k] for k in full_dict if full_dict[k].get("enabled", True) }
    target_config = targets_cfg[TARGET]

    target_config.setdefault("logg1", log_g(target_config["M1"] * M_sun, target_config["R1"] * R_sun).n)
    target_config.setdefault("logg2", log_g(target_config["M2"] * M_sun, target_config["R2"] * R_sun).n)

    # Additional data on the target populated with lookups
    target_data = {
        "label": target_config.get("label", TARGET),
        "search_term": target_config.get("search_term", TARGET)
    }

    simbad = Simbad()
    simbad.add_votable_fields("sp", "ids")
    if _tbl := simbad.query_object(target_data["search_term"]):
        target_data["ids"] = np.array(
            re.findall(r"(Gaia DR3|V\*|TIC|HD|HIP|2MASS)\s+(.+?(?=\||$))", _tbl["ids"][0]),
            dtype=[("type", object), ("id", object)])
        print("IDs:", ", ".join(f"{i['type']} {i['id']}" for i in target_data["ids"]))
        target_data["spt"] = _tbl["sp_type"][0]
        print("SpT:", target_data["spt"])

    # Let's get the Gaia DR3 data on this here object
    gaia_dr3_id = target_data["ids"][target_data["ids"]["type"] == "Gaia DR3"]["id"][0]
    if _job := Gaia.launch_job(f"SELECT TOP 1 * FROM gaiadr3.gaia_source WHERE source_id = {gaia_dr3_id}"):
        _tbl = _job.get_results()
        target_data["parallax_mas"] = ufloat(_tbl["parallax"][0], _tbl["parallax_error"][0])
        target_data["skycoords"] = _coords = SkyCoord(ra=_tbl["ra"][0] * u.deg,
                                                        dec=_tbl["dec"][0] * u.deg,
                                                        distance=1000 / _tbl["parallax"][0] * u.pc,
                                                        frame="icrs")
        target_data["distance_pc"] = 1000 / _tbl["parallax"][0]
        print(f"{TARGET} SkyCoords are {_coords} (or {_coords.to_string('hmsdms')})")

    # Lookup the TESS Input Catalog (8.2) for starting "system" Teff and logg values
    target_data["teff_sys"] = get_teff_from_spt(target_data["spt"]) or ufloat(5700, 0)
    target_data["logg_sys"] = ufloat(4.0, 0)
    if _tbl := Vizier(catalog="IV/39/tic82").query_object(target_data["search_term"], radius=0.1 * u.arcsec):
        if _row := _tbl[0][_tbl[0]["TIC"] in target_data["ids"][target_data["ids"]["type"] == "TIC"]["id"]]:
            # Teff may not be reliable - only use it if it's consistent with the SpT
            if target_data["teff_sys"].n-target_data["teff_sys"].s < (_row["Teff"] or 0) < target_data["teff_sys"].n+target_data["teff_sys"].s:
                target_data["teff_sys"] = ufloat(_row["Teff"], _row.get("s_Teff", None) or 0)
            if (_row["logg"] or 0) > 0:
                target_data["logg_sys"] = ufloat(_row["logg"], _row.get("s_logg", None) or 0)

    target_data["k"] = ufloat(target_config.get("k"), target_config.get("k_err", 0) or 0)
    if "light_ratio" in target_config:
        target_data["light_ratio"] = ufloat(target_config.get("light_ratio"),
                                            target_config.get("light_ratio_err", 0) or 0)
    else:
        # If from LC fit we may also need to consider 3L; l1=(1-3L)/(1+(L2/L1)) & l2=(1-3L)/(1+1/(L2/L1))
        target_data["light_ratio"] = ufloat(10**(target_config.get("logL2", 1) - target_config.get("logL1", 1)), 0)
    target_data["teff_ratio"] = (target_data["light_ratio"] / target_data["k"]**2)**0.25

    print(f"{TARGET} system values from lookup and LC fitting:")
    for p, unit in [("teff_sys", u.K), ("logg_sys", u.dex), ("k", None), ("teff_ratio", None)]:
        print(f"{p:>12s} = {target_data[p]:.3f} {unit or u.dimensionless_unscaled:unicode}")

    # The G23 (Gordon et al., 2023) Milky Way R(V) filter gives us the broadest coverage
    ext_model = G23(Rv=3.1)
    ext_wl_range = np.reciprocal(ext_model.x_range) * u.um # x_range has implicit units of 1/micron

    # BtSettlGrid & KuruczGrid available with the former having better coverage but slower/larger
    model_grid = get_stellar_grid("BtSettlGrid", extinction_model=ext_model, verbose=True)

    # Read in the SED for this target and de-duplicate (measurements may appear multiple times).
    sed = get_sed_for_target(TARGET, target_data["search_term"], radius=0.1, remove_duplicates=True)
    sed = group_and_average_fluxes(sed, verbose=True)

    # Filter SED to those covered by our models and also remove any outliers
    model_mask = np.ones((len(sed)), dtype=bool)
    model_mask &= model_grid.has_filter(sed["sed_filter"])
    model_mask &= np.isin(sed["sed_filter"], ["Cousins:I", "WISE:W4"], invert=True)
    model_mask &= (sed["sed_wl"] >= min(ext_wl_range)) \
                & (sed["sed_wl"] <= max(ext_wl_range)) \
                & (sed["sed_wl"] >= min(model_grid.wavelength_range)) \
                & (sed["sed_wl"] <= max(model_grid.wavelength_range))
    sed = sed[model_mask]

    out_mask = create_outliers_mask(sed, target_data["teff_sys"].n, [target_data["teff_ratio"].n],
                                    min_unmasked=15, verbose=True)
    sed = sed[~out_mask]

    sed.sort(["sed_wl"])
    print(f"{len(sed)} unique SED observation(s) retained after range, filter & outlier filtering",
            "\nwith the units for flux, frequency and wavelength being",
        ", ".join(f"{sed[f].unit:unicode}" for f in ["sed_flux", "sed_freq", "sed_wl"]))


    # Deredden
    fit_av = False
    if fit_av:
        sed["sed_der_flux"] = sed["sed_flux"]
    else:
        av, _ = get_gontcharov_av(target_data["skycoords"])
        print(f"\nDereddening SED observations with Av={av:.3f}")
        sed["sed_der_flux"] = sed["sed_flux"] / ext_model.extinguish(sed["sed_wl"].to(u.um), Av=av)

    # Set up the initial fit position. The fit mask indicates we're only fitting teffs & radii
    print("\nSetting up data for fitting")
    NUM_STARS = 2
    theta0 = fitter.create_theta(teffs=[target_data["teff_sys"].n] * NUM_STARS,
                                  loggs=[target_data["logg_sys"].n] * NUM_STARS,
                                  radii=[1.0] * NUM_STARS,
                                  dist=target_data["distance_pc"],
                                  av=0,
                                  nstars=NUM_STARS,
                                  verbose=True)

    fit_mask = np.array([True] * NUM_STARS      # Teff
                        + [False] * NUM_STARS   # logg
                        + [True] * NUM_STARS    # radius
                        + [False]               # dist
                        + [fit_av])             # av

    teff_limits = model_grid.teff_range
    radius_limits = (0.1, 100)

    # Ratios wrt the primary component
    teff_ratios = [1] + [ufloat(target_data["teff_ratio"].n, max(target_data["teff_ratio"].n * 0.05,
                                                                 target_data["teff_ratio"].s))] * (NUM_STARS - 1)
    rad_ratios = [1] + [ufloat(target_data["k"].n, max(target_data["k"].n * 0.05,
                                                       target_data["k"].s))] * (NUM_STARS - 1)

    # Distance & AV priors - only effect the outcome if these are fitted
    dist_prior = ufloat(theta0[-2], theta0[-2] * 0.05)
    av_prior = ufloat(theta0[-1], theta0[-1] * 0.1)

    print(f"Priors:  Teff ratios={', '.join(f'{r:.3f}' for r in teff_ratios[1:])},",
            f"radius ratios={', '.join(f'{r:.3f}' for r in rad_ratios[1:])},",
            f"dist={dist_prior:.3f},",
            f"Teff_limits={teff_limits}, radius_limits={radius_limits}")

    def ln_prior_func(theta: np.ndarray[float]) -> float:
        """
        The fitting prior callback function to evaluate the current set of candidate parameters
        (theta), returning a single negative ln(value) indicating their "goodness".

        Return a negative value as emcee will try to maximize the sum of this and the negative value
        of its ln_prob_func. The fitter component knows to flip the sign if we run a minimize fit.
        """
        teffs = theta[0 : NUM_STARS]
        radii = theta[2*NUM_STARS : 2*NUM_STARS + NUM_STARS]
        dist, av = theta[-2], theta[-1]

        # Limit criteria checks - hard pass/fail on these
        if not all(teff_limits[0] <= t <= teff_limits[1] for t in teffs) or \
            not all(radius_limits[0] <= r <= radius_limits[1] for r in radii):
            return -np.inf

        # Gaussian prior criteria: g(x) = 1/(σ*sqrt(2*pi)) * exp(-1/2 * (x-µ)^2/σ^2)
        # Omitting scaling expressions and note the implicit ln() cancelling the exp()
        ret_val = 0
        for ix in range(1, NUM_STARS): # Ratios
            ret_val += ((teffs[ix]/teffs[0] - teff_ratios[ix].n) / teff_ratios[ix].s)**2
            ret_val += ((radii[ix]/radii[0] - rad_ratios[ix].n) / rad_ratios[ix].s)**2
        # ret_val += ((dist - dist_prior.n) / dist_prior.s)**2
        # ret_val += ((av - av_prior.n) / av_prior.s)**2
        return -0.5 * ret_val

    # Get the sed data to be fitted
    x = model_grid.get_filter_indices(sed["sed_filter"])
    y = (sed["sed_der_flux"].quantity * sed["sed_freq"].quantity)\
                                    .to(model_grid.flux_unit, equivalencies=u.spectral()).value
    y_err = (sed["sed_eflux"].quantity * sed["sed_freq"].quantity)\
                                    .to(model_grid.flux_unit, equivalencies=u.spectral()).value

    # Quick initial minimize fit
    print()
    theta_min, _ = fitter.minimize_fit(x, y, y_err, theta0, fit_mask, verbose=True,
                                        ln_prior_func=ln_prior_func, stellar_grid=model_grid)

    # MCMC fit, starting from where the minimize fit finished
    print()
    theta_mcmc, _ = fitter.mcmc_fit(x, y, y_err, theta_min, fit_mask,
                                     ln_prior_func=ln_prior_func, stellar_grid=model_grid,
                                     processes=8, early_stopping=True, progress=True, verbose=True)

    # Output a comparison with known values (assuming we've fitted teffs and radii)
    print(f"\nFinal parameters for {TARGET} with nominals & 1-sigma error bars from MCMC fit")
    theta_labels = [("Teff1", model_grid.teff_unit), ("Teff2", model_grid.teff_unit),
                    ("R1", u.Rsun), ("R2", u.Rsun)]
    for (param, unit), fit in zip(theta_labels, theta_mcmc[fit_mask]):
        known = ufloat(target_config.get(param, np.NaN), target_config.get(param+"_err", None) or 0)
        print(f"{param:>12s} = {fit:.3f} {unit:unicode} (known value {known:.3f} {unit:unicode})")
