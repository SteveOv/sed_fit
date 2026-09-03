""" A module for fitting stellar model fluxes for multiple stars to stellar SEDs """
from typing import Tuple as _Tuple, List as _List, Callable as _Callable, Union as _Union
from numbers import Number

from threading import Lock as _Lock

import numpy as _np

from scipy.optimize import OptimizeResult

from emcee import EnsembleSampler

import astropy.units as _u
from astropy.constants import iau2015 as _iau2015

from uncertainties import UFloat as _UFloat

from sed_fit.stellar_grids import StellarGrid
from sed_fit import generic_fitter as _generic_fitter

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, no-member
pc = (1 * _u.pc).to(_u.m).value
R_sun = _iau2015.R_sun.to(_u.m).value


# GLOBALS which will be set by (minimize|mcmc)_fit prior to fitting. Hateful things!
# See the emcee documentation with the following link for a discussion of why this is needed.
# https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments
# Basically, by setting these as globals they're pickled & passed to each process only on creation.
# If a class+attributes or the EnsembleSampler's args are used they're pickled on every iteration.
# The code expects _fixed_theta and _fit_mask to be the same size.
_fixed_theta: _np.ndarray[float]
_fit_mask: _np.ndarray[bool]
_x: _np.ndarray[float]
_y: _np.ndarray[float]
_y_err: _np.ndarray[float]
_degr_free: float
_stellar_grid: StellarGrid
_ln_prior_func: _Callable[[_np.ndarray[float]], float]
_ln_likelihood_func: _Callable[[_np.ndarray[float]], float]

# Try to protect the globals as much as possible by wrapping their use within a critical section
_fit_mutex = _Lock()

def _ln_chisq_likelihood_func(y_model: _np.ndarray[float]) -> float:
    """
    A simple default fitting likelihood function used to evaluate the model values against
    the observations, returning a single negative value indicating the goodness of the fit.
    
    Based on a reduced chi^2 metric: chi^2_r = (Σ (y - y_model)^2 / y_err^2) / degr_free

    Based on the assumption that a chi^2_r of 1 indicates the best possible fit (with < 1
    indicating overfitting and > 1 underfitting) we find the absolute difference from one.
    Multiplied by -0.5 to give the ~ln() value.

    Accesses the following global variables which will be set by call to (minimize|mcmc)_fit()
    - _y: the observed y values
    - _y_err: the uncertainties in the observations
    - _degr_free: the degrees of freedom in the model

    :y_model: the model y values
    :returns: the goodness of the fit
    """
    return -0.5 * _np.abs(1 - (_np.sum(((_y - y_model) / _y_err)**2)  / _degr_free))

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

    y_model = _np.array([stellar_grid.get_filter_fluxes(x, teff, logg, 0, rad, dist, av)
                                            for teff, logg, rad, dist, av in iterate_theta(theta)])

    if combine:
        return _np.sum(y_model, axis=0)
    return y_model


def _ln_prob_func(fit_theta: _np.ndarray[float]) -> float:
    """
    The MCMC function which returns the log posterior probability; the probability that the
    candidate params (theta) are those responsible for the observations. This is a negative
    value tending towards zero as the probability increases. Think of this as:

    ln(P(posterior)) = ln(P(prior) * P(likelihood)) = _ln_prior_func() + _ln_likelihood_func()

    This takes the current set of fitted params (fit_theta) and merges them with the fixed params.
    The resulting param set (theta) is first evaluated by the prior function, then a model is
    generated from them, with model_func & its chosen stellar_grid, which is then evaluated against
    the observations with the likelihood function. The ln(product) of the two values is returned.

    :fit_theta: current set of candidate fitted parameters only (those given by the _fit_mask)
    :returns: the result of evaluating the fitted model against the observations
    """
    # Combine the fitted and fixed parameters to make a full set.
    theta = _fixed_theta.copy()
    theta[_fit_mask] = fit_theta

    if _np.isfinite(retval := _ln_prior_func(theta)):
        y_model = model_func(theta, combine=True)
        retval += _ln_likelihood_func(y_model)
        _np.nan_to_num(retval, copy=False, nan=-_np.inf)
    return retval


def minimize_fit(x: _np.ndarray[float],
                y: _np.ndarray[float],
                y_err: _np.ndarray[float],
                theta0: _np.ndarray[float],
                fit_mask: _np.ndarray[float],
                stellar_grid: StellarGrid,
                ln_prior_func: _Callable[[_np.ndarray[float]], float],
                ln_likelihood_func: _Callable[[_np.ndarray[float]],float]=_ln_chisq_likelihood_func,
                **kwargs) -> _Tuple[_np.ndarray[float], OptimizeResult]:
    """
    Quick fit model star(s) to the SED with scipy minimize fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.
    Will choose the best performing fit from the algorithms in methods.

    Will raise a ValueError if theta0 does not pass a priors check with ln_prior_func().

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :stellar_grid: a StellarGrid instance with which model fluxes will be generated
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :ln_likelihood_func: a callback function to evaluate the goodness of fit of model vs observation
    :**kwargs: kwargs passed to the generic_fitter.minimize_fit function
    :returns: the final set of parameters & a scipy OptimizeResult with the details of the outcome
    """
    if not _np.isfinite(ln_prior_func(theta0)):
        raise ValueError("theta0 failed ln_prior_func check.")

    with _fit_mutex:
        _set_globals(x, y, y_err, theta0, fit_mask, stellar_grid, ln_prior_func, ln_likelihood_func)

        return _generic_fitter.minimize_fit(ln_prob_func=_ln_prob_func,
                                            theta0=theta0,
                                            fit_mask=fit_mask,
                                            **kwargs)


def mcmc_fit(x: _np.ndarray[float],
             y: _np.ndarray[float],
             y_err: _np.ndarray[float],
             theta0: _np.ndarray[float],
             fit_mask: _np.ndarray[bool],
             stellar_grid: StellarGrid,
             ln_prior_func: _Callable[[_np.ndarray[float]], float],
             ln_likelihood_func: _Callable[[_np.ndarray[float]], float]=_ln_chisq_likelihood_func,
             **kwargs) -> _Tuple[_np.ndarray[_UFloat], EnsembleSampler]:
    """
    Full fit model star(s) to the SED with an MCMC fit of the model data generated from
    a combination of the fixed params on class iniialization and the fitted ones given here.

    Will run up to niters iterations. Every 1000 iterations will check if the fit has
    converged and will stop early if that is the case

    Will raise a ValueError if theta0 does not pass a priors check with ln_prior_func().

    :x: the wavelength/filter values for the observed SED data
    :y: the flux values, at x, for the observed SED data
    :y_err: the flux error bars, at x, for the observed SED data
    :theta0: the initial set of candidate parameters for the model SED
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :stellar_grid: a StellarGrid instance with which model fluxes will be generated
    :ln_prior_func: a callback function to evaluate the current theta against prior criteria
    :ln_likelihood_func: a callback function to evaluate the goodness of fit of model vs observation
    :**kwargs: kwargs passed to the generic_fitter.mcmc_fit function
    :returns: fitted set of parameters as UFloats and an EnsembleSampler with details of the outcome
    """
    if not _np.isfinite(ln_prior_func(theta0)):
        raise ValueError("theta0 failed ln_prior_func check.")

    with _fit_mutex:
        _set_globals(x, y, y_err, theta0, fit_mask, stellar_grid, ln_prior_func, ln_likelihood_func)

        return _generic_fitter.mcmc_fit(ln_prob_func=_ln_prob_func,
                                        ln_prior_func=ln_prior_func,
                                        theta0=theta0,
                                        fit_mask=fit_mask,
                                        **kwargs)


def create_theta(teffs: _Union[_List[float], float],
                 loggs: _Union[_List[float], float],
                 radii: _Union[_List[float], float],
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
        elif isinstance(val, _Tuple|_List|_np.ndarray) \
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


def _set_globals(x: _np.ndarray[float],
                 y: _np.ndarray[float],
                 y_err: _np.ndarray[float],
                 theta0: _np.ndarray[float],
                 fit_mask: _np.ndarray[bool],
                 stellar_grid: StellarGrid,
                 ln_prior_func: _Callable[[_np.ndarray[float]], float],
                 ln_likelihood_func: _Callable[[_np.ndarray[float]], float]):
    """ Utility function to set the various (hateful) globals required for fitting. """
    # pylint: disable=global-statement, line-too-long
    global _x, _y, _y_err, _degr_free, _fixed_theta, _fit_mask, _stellar_grid, _ln_prior_func, _ln_likelihood_func
    _x, _y, _y_err, _degr_free = x, y, y_err, y.shape[0] - sum(fit_mask)
    _fixed_theta, _fit_mask, _stellar_grid = _np.where(fit_mask,None,theta0), fit_mask, stellar_grid
    _ln_prior_func, _ln_likelihood_func = ln_prior_func, ln_likelihood_func
