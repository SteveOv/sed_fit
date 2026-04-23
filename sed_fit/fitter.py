""" A module for fitting stellar model fluxes for multiple stars to stellar SEDs """
from typing import Tuple, List, Callable, Union
from numbers import Number
from math import floor as _floor

from warnings import filterwarnings as _filterwarnings, catch_warnings as _catch_warnings

from multiprocessing import Pool as _Pool, cpu_count as _cpu_count
from threading import Lock as _Lock

import numpy as _np

from scipy.optimize import minimize as _minimize
from scipy.optimize import OptimizeResult, OptimizeWarning

from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError

import astropy.units as _u
from astropy.constants import iau2015 as _iau2015

from uncertainties import UFloat as _UFloat
from uncertainties.unumpy import uarray as _uarray

from sed_fit.stellar_grids import StellarGrid

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, no-member
pc = (1 * _u.pc).to(_u.m).value
R_sun = _iau2015.R_sun.to(_u.m).value


# GLOBALS which will be set by (minimize|mcmc)_fit prior to fitting. Hateful things!
# Unfortunately this is how we get fast MCMC, as the way emcee works makes
# using a class or passing these between functions in args way too sloooow!
# The code expects _fixed_theta and _fit_mask to be the same size.
_fixed_theta: _np.ndarray[float]
_fit_mask: _np.ndarray[bool]
_x: _np.ndarray[float]
_y: _np.ndarray[float]
_weights: _np.ndarray[float]
_ln_prior_func: Callable[[_np.ndarray[float]], float]
_stellar_grid: StellarGrid

# Try to protect them as much as possible by wrapping writes within a critical section
_fit_mutex = _Lock()


def _ln_likelihood_func(y_model: _np.ndarray[float], degrees_of_freedom: int) -> float:
    """
    The fitting likelihood function used to evaluate the model y values against the observations,
    returning a single negative value indicating the goodness of the fit.
    
    Based on a weighted chi^2: chi^2_w = 1/(N_obs-n_param) * Σ W(y-y_model)^2

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _y: the observed y values
    - _weights: the weights to apply to each observation/model y value

    :y_model: the model y values
    :degrees_of_freedom: the #observations/#params
    :returns: the goodness of the fit
    """
    chisq = _np.sum(_weights * (_y - y_model)**2) / degrees_of_freedom
    return -0.5 * chisq


def model_func(theta: _np.ndarray[float],
               x: _np.ndarray[float]=None,
               stellar_grid: StellarGrid=None,
               combine: bool=True):
    """
    Generate the model fluxes at points x from the candidate parameters theta. Calls the
    StellarGrid instance's get_filter_fluxes() func to generate model fluxes for each star.

    :theta: the full set of parameters from which to generate model fluxes
    :x: optional filter/wavelengths to generate fluxes for - if omitted will use _x
    :stellar_grid: optional StellarGrid with which model fluxes will be generated - if ommited will
    use global _stellar_grid instance
    :combine: whether to return a single set of summed fluxes
    :returns: the model fluxes at points x, either per star if combine==False or aggregated
    """
    # These can be taken as args for external calls but fall back in the hateful globals
    if x is None:
        x = _x
    if stellar_grid is None:
        stellar_grid = _stellar_grid

    y_model = _np.array([stellar_grid.get_filter_fluxes(x, teff, logg, 0, rad, dist, av, quick=True)
                                            for teff, logg, rad, dist, av in iterate_theta(theta)])

    if combine:
        return _np.sum(y_model, axis=0)
    return y_model


def _objective_func(fit_theta: _np.ndarray[float], minimizable: bool=False) -> float:
    """
    The function to be optimized by adjusting theta so that the return value converges to zero.

    :fit_theta: current set of candidate fitted parameters only
    :minimizable: whether this function is minimizable (returns positive) or not (returns negative)
    :returns: the result of evaluating the fitted model against the observations
    """
    # Combine the fitted and fixed parameters to make a full set.
    theta = _fixed_theta.copy()
    theta[_fit_mask] = fit_theta

    if _np.isfinite(retval := _ln_prior_func(theta)):
        y_model = model_func(theta, combine=True)

        degr_freedom = y_model.shape[0] - fit_theta.shape[0]
        retval += _ln_likelihood_func(y_model, degr_freedom)

        _np.nan_to_num(retval, copy=False, nan=_np.inf)

    if minimizable != (retval >= 0):
        return -retval
    return retval


def _print_theta(theta: _np.ndarray[float],
                 fit_mask: _np.ndarray[bool],
                 prefix: str="",
                 suffix: str=""):
    """ Utility function for pretty printing theta arrays & highlighting which items are fitted. """
    print((prefix if prefix else '') +
          "[" +
          ", ".join(f"{t:.3e}{'*' if f else ''}" for t, f in zip(theta, fit_mask)) +
          "]" +
          (suffix if suffix else ''))


def minimize_fit(x: _np.ndarray[float],
                 y: _np.ndarray[float],
                 y_err: _np.ndarray[float],
                 theta0: _np.ndarray[float],
                 fit_mask: _np.ndarray[float],
                 ln_prior_func: Callable[[_np.ndarray[float]], float],
                 stellar_grid: StellarGrid,
                 methods: List[str]=None,
                 verbose: bool=False) -> Tuple[_np.ndarray[float], OptimizeResult]:
    """
    Quick fit model star(s) to the SED with scipy minimize fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.
    Will choose the best performing fit from the algorithms in methods.

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :stellar_grid: a StellarGrid instance with which model fluxes will be generated
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :returns: the final set of parameters & a scipy OptimizeResult with the details of the outcome
    """
    if verbose:
        _print_theta(theta0, fit_mask, "minimize_fit(theta0=", ")")

    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    max_iters = int(1000 * sum(fit_mask))

    with _fit_mutex, _catch_warnings(category=[RuntimeWarning, OptimizeWarning]):
        _filterwarnings("ignore", "invalid value encountered in ")
        _filterwarnings("ignore", "Desired error not necessarily achieved due to precision loss.")
        _filterwarnings("ignore", "Unknown solver options:")

        # Now we've got exclusive access, we can set the globals required for fitting
        # pylint: disable=global-statement
        global _x, _y, _weights, _fixed_theta, _fit_mask, _ln_prior_func, _stellar_grid
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _fit_mask = _np.where(fit_mask, None, theta0), fit_mask
        _ln_prior_func, _stellar_grid = ln_prior_func, stellar_grid

        best_soln, best_method = None, None
        for method in methods:
            soln = _minimize(_objective_func, x0=theta0[fit_mask], args=(True), # minimizable
                             method=method, options={ "maxiter": max_iters, "maxfev": max_iters })
            if verbose:
                print(f"({method})", "succeeded" if soln.success else f"failed [{soln.message}]",
                        f"after {soln.nit:d} iterations & {soln.nfev:d} function evaluation(s)",
                        f"[fun = {soln.fun:.6f}]")

            if best_soln is None \
                    or (soln.success and not best_soln.success) \
                    or (soln.success == best_soln.success and soln.fun < best_soln.fun):
                best_soln, best_method = soln, method

    if best_soln.success:
        theta0[fit_mask] = best_soln.x
        if verbose:
            _print_theta(theta0, fit_mask, f"The best fit with {best_method} method yielded theta=")
    else:
        _print_theta(theta0, fit_mask, "The fit failed so returning input, theta0=")

    return theta0, best_soln


def mcmc_fit(x: _np.ndarray[float],
             y: _np.ndarray[float],
             y_err: _np.ndarray[float],
             theta0: _np.ndarray[float],
             fit_mask: _np.ndarray[bool],
             ln_prior_func: Callable[[_np.ndarray[float]], float],
             stellar_grid: StellarGrid,
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=10,
             seed: int=42,
             processes: int=1,
             autocor_tol: int=50,
             early_stopping: bool=True,
             early_stopping_from: int=None,
             progress: Union[bool, str]=False,
             verbose: bool=False) -> Tuple[_np.ndarray[_UFloat], EnsembleSampler]:
    """
    Full fit model star(s) to the SED with an MCMC fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.

    Will run up to niters iterations. Every 1000 iterations will check if the fit has
    converged and will stop early if that is the case

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :stellar_grid: a StellarGrid instance with which model fluxes will be generated
    :nwalkers: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :autocor_tol: the autocorrelation tolerance
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :early_stopping_from: override the number of steps before early stopping is considered
    :progress: whether to show a progress bar (see emcee documentation for other values)
    :returns: fitted set of parameters as UFloats and an EnsembleSampler with details of the outcome
    """
    if verbose:
        _print_theta(theta0, fit_mask, "mcmc_fit(theta0=", ")")

    rng = _np.random.default_rng(seed)
    theta_fit = theta0[fit_mask]
    ndim = len(theta_fit)
    tau = [_np.inf] * ndim

    # Starting positions for the walkers clustered around theta0
    p0 = [theta_fit + (theta_fit * rng.normal(0, 0.05, ndim)) for _ in _np.arange(int(nwalkers))]

    with _fit_mutex, \
            _Pool(processes=processes) as pool, \
            _catch_warnings(category=[RuntimeWarning, UserWarning]):

        _filterwarnings("ignore", message="invalid value encountered in ")
        _filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        # Now we've got exclusive access, we can set the globals required for fitting
        # pylint: disable=global-statement
        global _x, _y, _weights, _fixed_theta, _fit_mask, _ln_prior_func, _stellar_grid
        _x, _y, _weights = x, y, 1 / y_err**2
        _fixed_theta, _fit_mask = _np.where(fit_mask, None, theta0), fit_mask
        _ln_prior_func, _stellar_grid = ln_prior_func, stellar_grid

        if early_stopping_from is None or early_stopping_from <= 0:
            # Min steps required by Autocorr algo to avoid warn msg (not a warning so can't filter)
            early_stopping_from = int(50 * ndim * autocor_tol)

        if verbose:
            print("Running MCMC fit on", f"{processes}" if processes else f"up to {_cpu_count()}",
                f"process(es) with {nwalkers:d} walkers for {nsteps:d}",
                f"steps, sampling every {thin_by:d} steps." if thin_by > 1 else "steps.")
            if early_stopping:
                print(f"Early stopping is considered after {early_stopping_from:d} steps.")

        sampler = EnsembleSampler(int(nwalkers), ndim, _objective_func, pool=pool)
        step = 0
        for _ in sampler.sample(initial_state=p0, iterations=nsteps // thin_by,
                                thin_by=thin_by, tune=True, progress=progress):
            step = sampler.iteration * thin_by
            if early_stopping and step % 1000 == 0:
                try:
                    # The autocor time (tau) is the #steps to effectively forget start position.
                    # As the fit converges the change in tau will tend towards zero.
                    prev_tau, tau = tau, sampler.get_autocorr_time(c=5, tol=autocor_tol) * thin_by
                    if step >= early_stopping_from \
                            and not any(_np.isnan(tau)) \
                            and all(tau < step / 100) \
                            and all(abs(prev_tau - tau) / prev_tau < 0.01):
                        break
                except AutocorrError:
                    # The chain is too short. Can set the quiet arg to True in which case a warning
                    # message is output (but not a Python warning). Cleaner to consume the error.
                    pass

        if verbose and early_stopping and 0 < step < nsteps:
            print(f"Halting MCMC after {step:d} steps as the walkers are past",
                  "100 times the autocorrelation time & the fit has converged.")

    # Get theta into ufloats with std_dev based on the mean +/- 1-sigma values (where fitted)
    samples = samples_from_sampler(sampler, autocor_tol, thin_by, flat=True, verbose=verbose)
    fit_nom, quant_low, quant_high = median_and_quantile_values(samples, axis=0)
    theta_fit = _uarray(theta0, 0)
    theta_fit[fit_mask] = _uarray(fit_nom, _np.mean([quant_low, quant_high], axis=0))

    if verbose:
        _print_theta(theta_fit, fit_mask, "The MCMC fit yielded theta:  ")

    return theta_fit, sampler


def samples_from_sampler(sampler: EnsembleSampler,
                         autocor_tol: int=50,
                         thin_by: int=1,
                         flat: bool=False,
                         verbose: bool=False) -> _np.ndarray:
    """
    Gets the chain of samples from the passed sampler after calculating and discarding the
    estimated burn-in.
    
    :sampler: the completed sampler to inspect
    :autocor_tol: the autocorrelation tolerance
    :thin_by: step interval that was used to inspect the fit's progress and yield samples
    :flat: whether or not to return flattened samples
    :verbose: whether or not to write acceptance, sample and burn-in information to stdout
    :returns: the post burn-in samples
    """
    tau_iters = sampler.get_autocorr_time(c=5, tol=autocor_tol, quiet=True)
    def_tau_iter = sampler.iteration / 10
    burn_in_iters = int(_np.ceil(max(_np.nan_to_num(tau_iters, copy=True, nan=def_tau_iter)) * 2))

    # The chain consists of the samples at each iteration (equivalent to each thin_by step)
    samples = sampler.get_chain(discard=burn_in_iters, flat=flat)

    if verbose:
        print(f"Mean Acceptance fraction:    {_np.mean(sampler.acceptance_fraction):.3f}")
        print( "Autocorrelation steps (tau):", ", ".join(f"{t:.3f}" for t in tau_iters * thin_by))
        print(f"Estimated burn-in steps:     {burn_in_iters * thin_by:,}")
        print(f"Leaving samples of shape:    {samples.shape}", "*flattened" if flat else "")

    return samples


def median_and_quantile_values(values: Union[_np.ndarray[float], _np.ndarray[_UFloat]],
                               q: Tuple[float, float]=(0.16, 0.84),
                               axis: int=0) \
                            -> Tuple[_np.ndarray[float], _np.ndarray[float], _np.ndarray[float]] :
    """
    Will calculate the median and q-th lower & uppers quantile values of the passed
    array along the chosen axis.

    :values: the values to aggregate
    :q: a tuple in the form (q-lower, q-upper) of the lower and upper probabilities to calculate
    :axis: the axis along which the median and quantiles are computed
    :values: a tuple of arrays for the median, lower and upper quantiles
    """
    median = _np.median(values, axis=axis)
    quant_low = median - _np.quantile(values, min(q), axis=axis)
    quant_high = _np.quantile(values, max(q), axis=axis) - median
    return median, quant_low, quant_high


def create_theta(teffs: Union[List[float], float],
                 loggs: Union[List[float], float],
                 radii: Union[List[float], float],
                 dist: float,
                 av: float=0,
                 nstars: int=2,
                 verbose: bool=False) -> _np.ndarray[float]:
    """
    Helper function to validate the teffs, radii, loggs and dist values and create a theta list from
    them. This is the full set of parameters needed to generate a model SED from nstars components.

    The resulting theta array will have the form:
    ```python
    theta = [teff0, ... , teffN, logg0, ..., loggN, rad0, ... , radN, dist, av]
    ```
    where N is nstars - 1.

    Units: teffs in K, logg in dex[cgs], radii in Rsun and distance in parsecs

    Note: theta has to be one-dimensional as scipy minimize will not fit multidimensional theta

    :teffs: effective temps [K] as a list of floats nstars long or a single float (same value each)
    :loggs: stars' log(g) as a list of floats nstars long or a single float (same value each)
    :radii: stars' radii [Rsun] as a list of floats nstars long or a single float (same value each)
    :dist: the distance [parsecs] as a single float
    :av: the Av extinction parameter
    :nstars: the number of stars we're building for
    :returns: the resulting theta list
    """
    theta = _np.empty((nstars * 3 + 2), dtype=float)
    ix = 0
    for name, val in [("teffs", teffs),("loggs", loggs),("radii", radii),("dist", dist),("av", av)]:
        exp_count = 1 if name in ("dist", "av") else nstars

        # Attempt to interpret the value as a List[Number]
        if isinstance(val, Number):
            theta[ix : ix+exp_count] = [val] * exp_count
        elif isinstance(val, Tuple|List|_np.ndarray) \
                and len(val) == exp_count \
                and all(isinstance(v, Number|None) for v in val):
            theta[ix : ix+exp_count] = [t for t in val]
        else:
            raise ValueError(f"{name}=={val} cannot be interpreted as a List[Number]*{exp_count}")

        ix += exp_count

    if verbose:
        print("theta:\t", ", ".join(f"{t:.3e}" if isinstance(t, Number) else f"{t}" for t in theta))
    return theta


def iterate_theta(theta: _np.ndarray[float]):
    """
    The teff, rad and logg for each star is interleaved. This func simplifies access to each
    stars' params by returning a generator which yields the params for each star as a tuple.

    :theta: the theta value to parse and iterate over
    :returns: a generator of the tuple (teff, logg, radius, distance, av)
    """
    nstars = (theta.shape[0] - 2) // 3
    dist = theta[-2]
    av = theta[-1]
    for star in range(nstars):
        yield theta[star], theta[nstars*1 + star], theta[nstars*2 + star], dist, av
