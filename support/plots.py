""" Training and testing specific plots. """
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
from typing import Tuple, List, Union
from itertools import cycle

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table

from uncertainties.unumpy import nominal_values

from sed_fit.stellar_grids import SvoStellarGrid
from sed_fit.fitter import model_func, iterate_theta

def plot_sed(x: u.Quantity,
             fluxes: List[u.Quantity],
             flux_errs: List[u.Quantity]=None,
             fmts: List[str]=None,
             fillstyles: Union[List[str], str]="full",
             labels: List[str]=None,
             show_grid: bool=True,
             figsize: Tuple[float, float]=(6, 4),
             **format_kwargs) -> _Figure:
    """
    Will create a new figure with a single set of axes and will plot one or more sets of SED flux
    datapoints.

    The data and axes will be coerced to units of x=wavelength [um] and y=nu*F(nu) [W / m^2].
    The axes will be set to log-log scale.

    see the matplotlib docs for options for fmt and fillstyles
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

    :x: the x-axis/wavelength datapoints
    :fluxes: one or more sets of flux values at frequencies/wavelengths x
    :flux_errs: optional corresponding flux error bars
    :fmts: fmt options for each set of fluxes or leave as None for default 
    :fillstyles: fillstyle option for each set of fluxes or leave as None for full
    :labels: optional labels for the fluxes
    :title: optional title for the plot
    :show_grid: whether to show a grid within the plotted area
    :figsize: optional size for the figure
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    plot_sed_on_axes(ax, x, fluxes, flux_errs, fmts, fillstyles, labels)

    ax.set(xscale="log", xlabel=f"Wavelength ({u.um:latex_inline})",
           yscale="log", ylabel=f"${{\\rm \\nu F(\\nu)}}$ ({u.W/u.m**2:latex_inline})")
    if show_grid:
        ax.grid(True, which="both", axis="both", alpha=0.33, color="lightgray")
    legend_loc = "best" if labels is not None and any(l is not None for l in labels) else None
    format_axes(ax, legend_loc=legend_loc, **format_kwargs)
    return fig


def plot_fitted_model(sed: Table,
                      theta: ArrayLike,
                      model_grid: SvoStellarGrid,
                      sed_flux_colname: str="sed_fit_flux",
                      sed_flux_err_colname: str="sed_eflux",
                      sed_filter_colname: str="sed_filter",
                      sed_lambda_colname: str="sed_wl",
                      show_component_spectra: bool = True,
                      show_combined_spectrum: bool=False,
                      show_legend: bool=True,
                      show_grid: bool=True,
                      figsize: Tuple[float, float]=(6, 4),
                      **format_kwargs):
    """
    Wraps and extends plot_sed() so that the observed SED points are plotted plus the equivalent
    combined model SED data points from the fitted model. Additionally, the model SED points and
    full spectrum of each component star will be plotted.

    The data and axes will be coerced to units of x=wavelength [um] and y=nu*F(nu) [W / m^2].
    The axes will be set to log-log scale.

    :sed: the x-axis/wavelength datapoints
    :theta: the fitting parameters as passed to sed_fit model_func
    :model_grid: the StellarGrid supplying the model fluxes to the fitting
    :sed_flux_colname: name of the sed's flux column
    :sed_flux_err_colname: name of the sed's flux uncertainties column
    :sed_filter_colname: name of the sed's filter column
    :sed_lambda_colname: name of the sed's wavelength column
    :show_component_spectra: include a low alpha plot of the spectrum for each component
    :show_combined_spectrum: include a low alpha plot of the combined spectrum for the system
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    # Generate model SED fluxes at points x for each set of component star params in theta
    x = model_grid.get_filter_indices(sed[sed_filter_colname])
    theta_noms = nominal_values(theta)
    comp_fluxes = model_func(theta_noms, x, model_grid, combine=False) * model_grid.flux_unit

    # Need a set of plot formats/colours to cover reasonable number of components
    comp_fmts = ["or", "or", "*g", "+m"]
    comp_fillstyles = ["full", "none", "full", "full"]
    comp_colors = ["r", "r", "g", "m"]
    obs_color, comb_color = "k", "b" # colours of the observations and combined fitted model

    nstars = comp_fluxes.shape[0]
    if show_legend:
        labels = ["fitted pair", "observations"] + [f"fitted star {i+1}" for i in range(nstars)]
    else:
        labels = [None] * (2 + nstars)

    xlabel = f"Wavelength ({u.um:latex_inline})"
    ylabel = f"${{\\rm \\nu F(\\nu)}}$ ({u.W/u.m**2:latex_inline})"
    lam = sed[sed_lambda_colname].to(u.um, equivalencies=u.spectral())
    combined_model_flux = np.sum(comp_fluxes, axis=0)

    # Plot the fitted model against the chosen SED + show each star's contribution
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    plot_sed_on_axes(ax,
                    x=lam,
                    fluxes=[combined_model_flux, sed[sed_flux_colname].quantity] +list(comp_fluxes),
                    flux_errs=[None, sed[sed_flux_err_colname].quantity] + [None]*nstars,
                    fmts=["." + comb_color, "*" + obs_color] + list(_cycle_for(comp_fmts, nstars)),
                    fillstyles=["full", "full"] + list(_cycle_for(comp_fillstyles, nstars)),
                    labels=labels)

    # Plot the raw spectra for each component as a background
    if show_combined_spectrum or show_component_spectra:
        spec_lams = model_grid.wavelengths * model_grid.wavelength_unit
        mask = spec_lams >= sed[sed_lambda_colname].quantity.min() * 0.8
        mask &= spec_lams <= sed[sed_lambda_colname].quantity.max() * 1.2
        comb_spec_flux = np.zeros_like(model_grid.wavelengths, dtype=float)
        for (teff, logg, rad, dist, av), c in zip(iterate_theta(theta_noms),
                                                _cycle_for(comp_colors, nstars)):
            spec_flux = model_grid.get_fluxes(wavelengths=model_grid.wavelengths, teff=teff,
                                              logg=logg, metal=0, radius=rad, distance=dist, av=av)
            comb_spec_flux += spec_flux
            if show_component_spectra:
                ax.plot(spec_lams[mask], spec_flux[mask] * model_grid.flux_unit,
                        c=c, alpha=0.25, zorder=-100)
        if show_combined_spectrum:
            ax.plot(spec_lams[mask], comb_spec_flux[mask] * model_grid.flux_unit,
                    c=comb_color, alpha=0.75, zorder=-100)
        ax.set(xscale="log", xlabel=xlabel, yscale="log", ylabel=ylabel)

    if show_grid:
        ax.grid(True, which="both", axis="both", alpha=0.33, color="lightgray")
    legend_loc = "best" if labels is not None and any(l is not None for l in labels) else None
    format_axes(ax, legend_loc=legend_loc, **format_kwargs)
    return fig


def plot_sed_on_axes(ax: _Axes,
                     x: u.Quantity,
                     fluxes: List[u.Quantity],
                     flux_errs: List[u.Quantity]=None,
                     fmts: List[str]=None,
                     fillstyles: Union[List[str], str]="full",
                     labels: List[str]=None):
    """
    Will plot a sed to the passed axes. The data and axes will be coerced to units of
    x=wavelength [um] and y=nu*F(nu) [W / m^2].

    see the matplotlib docs for options for fmt and fillstyles
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

    :ax: the axes to plot to
    :x: the x-axis/wavelength datapoints
    :fluxes: one or more sets of flux values at frequencies/wavelengths x
    :flux_errs: optional corresponding flux error bars (must have same dimensions as fluxes)
    :fmts: fmt options for each set of fluxes or leave as None for default 
    :fillstyles: fillstyle option for each set of fluxes or leave as None for full
    :labels: optional labels for the fluxes
    """
    if isinstance(fluxes, u.Quantity):
        fluxes = [fluxes]
    if isinstance(flux_errs, u.Quantity):
        flux_errs = [flux_errs]
    if isinstance(fmts, str):
        fmts = [fmts] * len(fluxes)
    if isinstance(fillstyles, str):
        fillstyles = [fillstyles] * len(fluxes)
    if isinstance(labels, str):
        labels = [labels] + [None] * len(fluxes)-1

    vfv_unit = u.W / u.m**2
    lam = x.to(u.um, equivalencies=u.spectral())
    freq = x.to(u.Hz, equivalencies=u.spectral())

    for flux, flux_err, fmt, fs, label in zip(fluxes, flux_errs, fmts, fillstyles, labels):
        vfv, vfv_err = None, None
        if flux is not None:
            if flux.unit.is_equivalent(vfv_unit):
                vfv = flux.to(vfv_unit , equivalencies=u.spectral_density(freq))
            else:
                vfv = (flux * freq).to(vfv_unit , equivalencies=u.spectral_density(freq))
        if flux_err is not None:
            if flux_err.unit.is_equivalent(vfv_unit):
                vfv_err = flux_err.to(vfv_unit , equivalencies=u.spectral_density(freq))
            else:
                vfv_err = (freq * flux_err).to(vfv_unit, equivalencies=u.spectral_density(freq))

        if vfv is not None:
            ax.errorbar(lam, vfv, vfv_err, fmt=fmt, fillstyle=fs, alpha=0.5, label=label)

    ax.set(xscale="log", xlabel=f"Wavelength [{u.um:latex_inline}]",
           yscale="log", ylabel=f"${{\\rm \\nu F(\\nu)}}$ [{u.W/u.m**2:latex_inline}]")
   

def format_axes(ax: _Axes, title: str=None,
                xlabel: str=None, ylabel: str=None,
                xticklable_top: bool=False, yticklabel_right: bool=False,
                invertx: bool=False, inverty: bool=False,
                xlim: Tuple[float, float]=None, ylim: Tuple[float, float]=None,
                minor_ticks: bool=True, legend_loc: str=None):
    """
    General purpose formatting function for a set of Axes. Will carry out the
    formatting instructions indicated by the arguments and will set all ticks
    to internal and on all axes.

    :ax: the Axes to format
    :title: optional title to give the axes, overriding prior code - set to "" to surpress
    :xlabel: optional x-axis label text to set, overriding prior code - set to "" to surpress
    :ylabel: optional y-axis label text to set, overriding prior code - set to "" to surpress
    :xticklabel_top: move the x-axis ticklabels and label to the top
    :yticklabel_right: move the y-axis ticklabels and label to the right
    :invertx: invert the x-axis
    :inverty: invert the y-axis
    :xlim: set the lower and upper limits on the x-axis
    :ylim: set the lower and upper limits on the y-axis
    :minor_ticks: enable or disable minor ticks on both axes
    :legend_loc: if set will enable to legend and set its position.
    For available values see matplotlib legend(loc="")
    """
    # pylint: disable=too-many-arguments
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if invertx:
        ax.invert_xaxis()
    if inverty:
        ax.invert_yaxis()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if minor_ticks:
        ax.minorticks_on()
    else:
        ax.minorticks_off()
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, bottom=True, left=True, right=True)
    if xticklable_top:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    if yticklabel_right:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    if legend_loc:
        ax.legend(loc=legend_loc)
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()

def _cycle_for(init_list: List, num_items: int):
    """
    Util func to cycle over the passed list and stop after yielding the required number of items
    """
    for i, v in enumerate(cycle(init_list)):
        if i == num_items:
            break
        yield v
