""" A base module for generic fitting """
from typing import Tuple as _Tuple, List as _List, Callable as _Callable, Union as _Union
from warnings import filterwarnings as _filterwarnings, catch_warnings as _catch_warnings
from multiprocessing import Pool as _Pool, cpu_count as _cpu_count

import numpy as _np

from scipy.optimize import minimize as _minimize
from scipy.optimize import OptimizeResult as _OptimizeResult, OptimizeWarning as _OptimizeWarning

from emcee import EnsembleSampler as _EnsembleSampler
from emcee.autocorr import AutocorrError as _AutocorrError

from uncertainties import UFloat as _UFloat
from uncertainties.unumpy import uarray as _uarray


def minimize_fit(ln_prob_func: _Callable[[_np.ndarray[float], any], float],
                 theta0: _np.ndarray[float],
                 fit_mask: _np.ndarray[bool]=None,
                 fit_args: _Tuple=None,
                 methods: _List[str]=None,
                 verbose: bool=False) -> _Tuple[_np.ndarray[float], _OptimizeResult]:
    """
    Optimize the parameters for ln_prob_func using scipy's minimize functionality, starting with
    the initial set in theta0. The fit_mask indicates which members of theta0 may be fitted (True)
    and which should be held fixed (False). If fit_mask is omitted it is assumed that all values
    may be fitted. Will fit ln_prob_func with each of the listed methods, returning the result from
    that which yields the result with the smallest magnitude.

    :ln_prob_func: the probability function to optimize - expected to return negative values
    :theta0: the initial set of candidate parameters for the ln_prob_func
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :fit_args: any fixed args required by ln_prob_func in addition to theta
    :methods: scipy optimize fitting algorithms to try, defaults to [Nelder-Mead, SLSQP, None]
    :verbose: whether or not to write progress messages to stdout
    :returns: the final set of parameters & a scipy OptimizeResult with the details of the outcome
    """
    if verbose:
        print_theta(theta0, fit_mask, "minimize_fit(theta0=", ")")

    if fit_mask is None:
        fit_mask = _np.ones_like(theta0, dtype=bool)
    if methods is None:
        methods = ["Nelder-Mead", "SLSQP", None]
    elif isinstance(methods, str):
        methods = [methods]

    with _catch_warnings(category=[RuntimeWarning, _OptimizeWarning]):
        _filterwarnings("ignore", "overflow encountered in scalar power")
        _filterwarnings("ignore", "invalid value encountered in ")
        _filterwarnings("ignore", "Desired error not necessarily achieved due to precision loss.")
        _filterwarnings("ignore", "Unknown solver options:")

        # Make sure we do not modify the starting position
        theta = theta0.copy()
        max_iters = int(1000 * sum(fit_mask))
        best_soln, best_method = None, None
        for method in methods:
            soln = _minimize(lambda *args: abs(ln_prob_func(*args)),
                             x0=theta[fit_mask],
                             # Default args to empty tuple otherwise a None value will be sent
                             args=fit_args or (),
                             method=method,
                             options={ "maxiter": max_iters, "maxfev": max_iters })
            if verbose:
                print(f"({method})", "succeeded" if soln.success else f"failed [{soln.message}]",
                        f"after {soln.nit:d} iterations & {soln.nfev:d} function evaluation(s)",
                        f"[fun = {soln.fun:.6f}]")

            if best_soln is None \
                    or (soln.success and not best_soln.success) \
                    or (soln.success == best_soln.success and soln.fun < best_soln.fun):
                best_soln, best_method = soln, method

    if best_soln is not None and best_soln.success:
        theta[fit_mask] = best_soln.x
        if verbose:
            print_theta(theta, fit_mask,
                        f"The best fit used the '{best_method}' method and yielded theta=")
    else:
        print_theta(theta, fit_mask, "The fit failed so returning input, theta0=")

    return theta, best_soln


def mcmc_fit(ln_prob_func: _Callable[[_np.ndarray[float], any], float],
             ln_prior_func: _Callable[[_np.ndarray[float]], float],
             theta0: _np.ndarray[float],
             fit_mask: _np.ndarray[bool]=None,
             fit_args: _Tuple=None,
             nwalkers: int=100,
             nsteps: int=100000,
             thin_by: int=10,
             seed: int=42,
             processes: int=1,
             autocor_tol: int=50,
             early_stopping: bool=True,
             early_stopping_from: int=None,
             progress: _Union[bool, str]=False,
             verbose: bool=False) -> _Tuple[_np.ndarray[_UFloat], _EnsembleSampler]:
    """
    Sample the parameters for ln_prob_func using emcee MCMC functionality, starting with the initial
    set in theta0. The fit_mask indicates which members of theta0 may be varied (True) and which
    should be held fixed (False). If fit_mask is omitted it is assumed that all values are fitted.

    Will run up to niters iterations. Every 1000 iterations will check if the fit has
    converged and will stop early if that is the case

    :ln_prob_func: the probability function to optimize - expected to return negative values
    :ln_prior_func: the callback function to evaluate the current theta against prior criteria
    :theta0: the initial set of candidate parameters for the ln_prob_func
    :fit_mask: a mask on theta0 to pick the parameters that are fitted, the rest being fixed
    :nwalkers: the number of mcmc walkers to employ
    :nsteps: the maximium number of mcmc steps to make for each walker
    :thin_by: step interval to inspect fit progress
    :seed: optional seed for random behaviour
    :processes: optional number of parallel processes to use, or None to let code choose
    :autocor_tol: the autocorrelation tolerance
    :early_stopping: stop fitting if solution has converged & further improvements are negligible
    :early_stopping_from: override the number of steps before early stopping is considered
    :progress: whether to show a progress bar (see emcee documentation for other values)
    :verbose: whether or not to write progress messages to stdout
    :returns: fitted set of parameters as UFloats and a McmcResult with details of the outcome
    """
    if verbose:
        print_theta(theta0, fit_mask, "mcmc_fit(theta0=", ")")

    if fit_mask is None:
        fit_mask = _np.ones_like(theta0, dtype=bool)

    rng = _np.random.default_rng(seed)
    ndim = sum(fit_mask)
    tau = [_np.inf] * ndim

    # Starting position for the walkers clustered around theta0, via priors to ensure they're valid.
    # We override the normal scale so it's never zero to ensure we have some scatter in p0,
    # otherwise emcee will likely throw a Value Error "Initial state has large condition number".
    locs = theta0[fit_mask]
    scales = _np.maximum(_np.abs(theta0[fit_mask]) * 0.05, 1e-6)
    p0, test_theta = [], theta0.copy()
    while len(p0) < int(nwalkers):
        test_theta[fit_mask] = rng.normal(locs, scales)
        if _np.isfinite(ln_prior_func(test_theta)):
            p0 += [test_theta[fit_mask]]

    if early_stopping_from is None or early_stopping_from <= 0:
        # Min steps required by Autocorr algo to avoid warn msg (not a warning so can't filter)
        early_stopping_from = int(50 * ndim * autocor_tol)

    if verbose:
        print("Running MCMC fit on", f"{processes}" if processes else f"up to {_cpu_count()}",
            f"process(es) with {nwalkers:d} walkers for {nsteps:d}",
            f"steps, sampling every {thin_by:d} steps." if thin_by > 1 else "steps.")
        if early_stopping:
            print(f"Early stopping is considered after {early_stopping_from:d} steps.")

    with _Pool(processes) as pool, _catch_warnings(category=[RuntimeWarning, UserWarning]):
        _filterwarnings("ignore", message="invalid value encountered in ")
        _filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

        sampler = _EnsembleSampler(int(nwalkers), ndim, ln_prob_func, args=fit_args, pool=pool)
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
                except _AutocorrError:
                    # The chain is too short. Can set the quiet arg to True in which case a warning
                    # message is output (but not a Python warning). Cleaner to consume the error.
                    pass

        if verbose and early_stopping and 0 < step < nsteps:
            print(f"Halting MCMC sampling after {step:d} steps as the walkers are beyond",
                    "100 times the autocorrelation time & the fit has converged.")

        # Get theta into ufloats with std_dev based on the mean +/- 1-sigma values (where fitted)
        theta_mcmc = _uarray(theta0, 0)
        samples = samples_from_sampler(sampler, autocor_tol, thin_by, flat=True, verbose=verbose)
        lo, med, hi = _np.quantile(samples, q=(0.16, 0.5, 0.84), axis=0)
        theta_mcmc[fit_mask] = _uarray(med, _np.mean([med-lo, hi-med], axis=0))

    if verbose:
        print_theta(theta_mcmc, fit_mask, "The MCMC fit yielded theta:  ")

    return theta_mcmc, sampler


def samples_from_sampler(sampler: _EnsembleSampler,
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


def print_theta(theta: _np.ndarray[float],
                fit_mask: _np.ndarray[bool]=None,
                prefix: str="",
                suffix: str="",
                number_format: str=".3e"):
    """
    Utility function for pretty printing theta arrays. Fitted values,
    those where the fit_mask is True, will be indicated with an asterisk next to the value.
    If fit_mask is omitted it is assumed that all values are fitted.

    :theta: the raw theta array
    :fit_mask: a mask on theta to pick the parameters that are fitted, the rest being fixed
    :prefix: optional text to print before theta
    :suffix: optional text to print after theta
    :number_format: the string interpolation format to apply to the theta items
    """
    if fit_mask is None:
        fit_mask = _np.ones_like(theta, dtype=bool)
    item_fmt = f"{{0:{number_format}}}{{1}}"
    print((prefix if prefix else '') +
          "[" +
          ", ".join(item_fmt.format(t, "*" if f else "") for t, f in zip(theta, fit_mask)) +
          "]" +
          (suffix if suffix else ''))
