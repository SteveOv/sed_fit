""" Unit tests for the generic_fitter module. """
# pylint: disable=unused-import, too-many-public-methods, line-too-long, invalid-name, no-member
import unittest
import numpy as np

from sed_fit.generic_fitter import minimize_fit, mcmc_fit
class Test_generic_fitter(unittest.TestCase):
    """ Unit tests for the generic_fitter module. """

    # Ideally, these would be private to each test be emcee requires them to support pickling
    @staticmethod
    def ln_prior_func(theta) -> float:
        """ Simple prior function """
        m, b, log_f = theta
        if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
            return 0.0
        return -np.inf

    @staticmethod
    def ln_likelihood_func(theta, x, y, y_err) -> float:
        """ Simple likelihood function """
        m, b, log_f = theta
        model = m * x + b
        sigma2 = y_err**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    @staticmethod
    def ln_prob_func(theta, x, y, y_err):
        """ Simple prob function """
        lp = Test_generic_fitter.ln_prior_func(theta)
        if np.isfinite(lp):
            return lp + Test_generic_fitter.ln_likelihood_func(theta, x, y, y_err)
        return -np.inf


    def test_minimize_fit(self):
        """ Simple test of minimize_fit() against a noisy data on a slope. """
        # Based on the sample in the emcee tutorial
        # https://emcee.readthedocs.io/en/stable/tutorials/line/

        # Generate some synthetic data from the model.
        np.random.seed(123)
        expected_m, expected_b, f_true = -0.9594, 4.294, 0.534
        x, y, y_err = Test_generic_fitter._generate_synth_data(expected_m, expected_b, f_true)

        theta0 = np.array([expected_m, expected_b, np.log(f_true)]) + 0.1 * np.random.randn(3)
        fit_mask = np.ones_like(theta0, dtype=bool)

        theta_min, _ = minimize_fit(ln_prob_func=Test_generic_fitter.ln_prob_func,
                                    theta0=theta0,
                                    fit_mask=fit_mask,
                                    fit_args=(x, y, y_err),
                                    methods=["Nelder-Mead"],
                                    verbose=True)

        self.assertAlmostEqual(expected_m, theta_min[0], 0)
        self.assertAlmostEqual(expected_b, theta_min[1], 0)

    def test_mcmc_fit(self):
        """ Simple test of mcmc_fit() against a noisy data on a slope. """
        # Based on the sample in the emcee tutorial
        # https://emcee.readthedocs.io/en/stable/tutorials/line/

        # Generate some synthetic data from the model.
        np.random.seed(123)
        expected_m, expected_b, f_true = -0.9594, 4.294, 0.534
        x, y, y_err = Test_generic_fitter._generate_synth_data(expected_m, expected_b, f_true)

        theta0 = np.array([expected_m, expected_b, np.log(f_true)]) + 0.1 * np.random.randn(3)
        fit_mask = np.ones_like(theta0, dtype=bool)

        theta_mcmc, _ = mcmc_fit(ln_prob_func=Test_generic_fitter.ln_prob_func,
                                 ln_prior_func=Test_generic_fitter.ln_prior_func,
                                 theta0=theta0,
                                 fit_mask=fit_mask,
                                 fit_args=(x, y, y_err),
                                 verbose=True)

        self.assertAlmostEqual(expected_m, theta_mcmc[0].n, 0)
        self.assertAlmostEqual(expected_b, theta_mcmc[1].n, 0)

    @staticmethod
    def _generate_synth_data(m, b, f, N: int=50):
        x = np.sort(10 * np.random.rand(N))
        y_err = 0.1 + 0.5 * np.random.rand(N)
        y = m * x + b
        y += np.abs(f * y) * np.random.randn(N)
        y += y_err * np.random.randn(N)
        return x, y, y_err

if __name__ == "__main__":
    unittest.main()
