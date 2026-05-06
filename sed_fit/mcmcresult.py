""" The result of an MCMC sampler 'fit' """
import numpy as _np
from uncertainties.unumpy import uarray as _uarray

from emcee import EnsembleSampler

class McmcResult():
    """ The result of an MCMC sampler 'fit' """

    def __init__(self, theta0: _np.ndarray[float], fit_mask: _np.ndarray[bool],
                 sampler: EnsembleSampler, autocor_tol: float, thin_by: int):
        self._theta0 = theta0
        self._fit_mask = fit_mask
        self._sampler = sampler
        self._autocor_tol = autocor_tol
        self._thin_by = thin_by

    @property
    def tau_iters(self) -> int:
        """
        The autocorrelation (tau) iterations (steps/thin_by) for each fitted param.
        These are the estimated number of iterations to 'forget' the start position.
        """
        return self._sampler.get_autocorr_time(c=5, tol=self._autocor_tol, quiet=True)

    @property
    def burn_in_iters(self) -> int:
        """ The estimated number of iterations (steps/thin_by) for the burn-in """
        def_tau_iters = self._sampler.iteration / 10
        return int(_np.ceil(max(_np.nan_to_num(self.tau_iters, copy=True, nan=def_tau_iters)) * 2))

    @property
    def burn_in_steps(self) -> int:
        """ The estimated number of steps (iterations * thin_by) for the burn-in """
        return self.burn_in_iters * self._thin_by

    def get_sample_chain(self, flat: bool=False, discard: int=None) -> _np.ndarray:
        """
        Get the chain of iteration samples, optionally without the burn-in iterations.
        These samples will have been taken every iteration (1 iteration every thin_by step).

        :flat: whether or not to flatten the chain
        :discard: number of "burn in" iters to omit from the chain, or burn_in_iters if None
        :returns: the requested samples
        """
        if discard is None:
            discard = self.burn_in_iters
        return self._sampler.get_chain(discard=discard, flat=flat)

    def get_theta(self,
                  discard: int=None,
                  uncertainty_ratio: float=0.6827) -> _np.ndarray:
        """
        The resulting set of (theta) medians and uncertainties from the MCMC samples.

        :discard: number of "burn in" iters to omit from the samples, or burn_in_iters if None
        :uncertainty_ratio: ratio of samples about median for uncertainty; default equiv to 1-sigma
        :returns: the final theta from the sample chain with +/- uncertainties for fitted values
        """
        samples = self.get_sample_chain(discard=discard, flat=True)
        lo, med, hi = _np.quantile(samples,
                                   q=(0.5 - uncertainty_ratio/2, 0.5, 0.5 + uncertainty_ratio/2),
                                   axis=0)

        theta = _uarray(self._theta0, 0)
        theta[self._fit_mask] = _uarray(med, _np.mean([med-lo, hi-med], axis=0))
        return theta

    def __str__(self):
        return f"""Mean Acceptance fraction:    {_np.mean(self._sampler.acceptance_fraction):.3f}
Autocorrelation steps (tau): {', '.join(f'{t:.3f}' for t in self.tau_iters * self._thin_by)}
Estimated burn-in steps:     {self.burn_in_steps:,}"""
