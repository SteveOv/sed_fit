""" The result of an MCMC sampler 'fit' """
from typing import Tuple as _Tuple
import numpy as _np
from uncertainties import UFloat as _UFloat
from uncertainties.unumpy import uarray as _uarray
from emcee import EnsembleSampler

class McmcResult():
    """ The result of an MCMC sampler 'fit' """

    def __init__(self, theta0: _np.ndarray[float], fit_mask: _np.ndarray[bool],
                 sampler: EnsembleSampler, autocorr_tol: float, thin_by: int):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self._theta0 = theta0
        self._fit_mask = fit_mask
        self._sampler = sampler
        self._autocorr_tol = autocorr_tol
        self._thin_by = thin_by

    @property
    def sampler(self) -> EnsembleSampler:
        """ The emcee sampler from which the results are calculated. """
        return self._sampler

    @property
    def autocorr_tol(self) -> float:
        """ The autocorrelation tolerance used. """
        return self._autocorr_tol

    @property
    def thin_by(self) -> int:
        """ The number of steps the samples were thinned by. """
        return self._thin_by

    @property
    def tau_iters(self) -> _np.ndarray:
        """
        The autocorrelation (tau) iterations (steps/thin_by) for each fitted param.
        These are the estimated number of iterations to 'forget' the start position.
        """
        return self._sampler.get_autocorr_time(c=5, tol=self._autocorr_tol, quiet=True)

    @property
    def burn_in_iters(self) -> int:
        """ The estimated number of iterations (steps/thin_by) for the burn-in """
        def_tau_iters = self._sampler.iteration / 25
        return int(_np.ceil(max(_np.nan_to_num(self.tau_iters, copy=True, nan=def_tau_iters)) * 5))

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
                  quantiles: _Tuple=(0.16, 0.5, 0.84),
                  discard: int=None) -> _np.ndarray[_UFloat]:
        """
        The resulting set of (theta) nominals and uncertainties from the MCMC samples.

        The quantiles argument indicates the range of sample values which make up the final
        theta values on the assumption that the samples are consistent with a normal distribution.
        They are in the form (low, median, high) or (low, high) with the median assumed to be 0.5.
        i.e.: quantiles of (0.16, 0.5, 0.84) will yield the median +/- 1-sigma for each theta value.

        Fixed theta values will be given an uncertainty of zero.

        :quantiles: the quantiles over which to summarise the samples
        :discard: number of "burn in" iters to omit from the samples, or burn_in_iters if None
        :returns: the final theta from the sample chain with +/- uncertainties for fitted values
        """
        if isinstance(quantiles, _Tuple) and 2 <= len(quantiles) <= 3:
            if len(quantiles) == 2:
                quantiles = (quantiles[0], 0.5, quantiles[1])
        else:
            raise ValueError("quantiles must be a tuple in form (low, high) or (low, mid, high)")

        samples = self.get_sample_chain(discard=discard, flat=True)
        lo, med, hi = _np.quantile(samples, q=quantiles, axis=0)

        theta = _uarray(self._theta0, 0)
        theta[self._fit_mask] = _uarray(med, _np.mean([med-lo, hi-med], axis=0))
        return theta

    def __str__(self):
        return f"""Mean Acceptance fraction:    {_np.mean(self._sampler.acceptance_fraction):.3f}
Autocorrelation steps (tau): {', '.join(f'{t:.3f}' for t in self.tau_iters * self._thin_by)}
Estimated burn-in steps:     {self.burn_in_steps:,}"""
