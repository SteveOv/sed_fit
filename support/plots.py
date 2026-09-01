""" Training and testing specific plots. """
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
from typing import Tuple, List, Union
from itertools import cycle, zip_longest
from numbers import Number
from pathlib import Path
from inspect import getsourcefile

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table

from uncertainties.unumpy import nominal_values, std_devs

from sed_fit.stellar_grids import SvoStellarGrid
from sed_fit.fitter import model_func, iterate_theta

from .data.mist.read_mist_models import ISO

# Units for the wavelemgth (x) and flux (y) axes
lam_unit = u.um
flux_unit = u.W / u.m**2

_this_dir = Path(getsourcefile(lambda:0)).parent

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
    plot_sed_on_axes(ax, x, fluxes, flux_errs,
                     fmts, fillstyles=fillstyles, marker_sizes=7.5, labels=labels)

    ax.set(xscale="log", xlabel=f"Wavelength ({lam_unit:latex_inline})",
           yscale="log", ylabel=f"${{\\rm \\nu F(\\nu)}}$ ({flux_unit:latex_inline})")
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
                      show_combined_fit: bool=True,
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
    :show_combined_spectrum: include a plot of the combined spectrum for the system
    :show_combined_fit: include plots of the combined fitted values for the system
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    # Generate model SED fluxes at points x for each set of component star params in theta
    x = model_grid.get_filter_indices(sed[sed_filter_colname])
    theta_noms = nominal_values(theta)
    model_fluxes = model_func(theta_noms, x, model_grid, combine=False) * model_grid.flux_unit

    # Need a set of plot formats/colours to cover reasonable number of components
    star_fmts = ["og", "og", "*k", "+m"]
    star_fillstyles = ["full", "none", "full", "full"]
    star_colors = ["g", "g", "k", "m"]
    obs_color, comb_color = "r", "b" # colours of the observations and combined fitted model

    nstars = model_fluxes.shape[0]
    subs = ["ABCDEFGHIJKLM"[n] for n in range(nstars)]
    if show_legend:
        labels = ["fitted pair", "observations"] + [f"fitted star {sub}" for sub in subs]
    else:
        labels = [None] * (2 + nstars)

    xlabel = f"Wavelength ({lam_unit:latex_inline})"
    ylabel = f"${{\\rm \\nu F(\\nu)}}$ ({flux_unit:latex_inline})"
    lam = sed[sed_lambda_colname].to(u.um, equivalencies=u.spectral())
    comb_model_flux = np.sum(model_fluxes, axis=0)

    # Plot the fitted model against the chosen SED + show each star's contribution
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    slc = slice(None) if show_combined_fit else slice(1, None)
    plot_sed_on_axes(ax,
            x=lam,
            fluxes=([comb_model_flux, sed[sed_flux_colname].quantity] + list(model_fluxes))[slc],
            flux_errs=([None, sed[sed_flux_err_colname].quantity] + [None]*nstars)[slc],
            fmts=(["." + comb_color, "o" + obs_color] + list(_cycle_for(star_fmts, nstars)))[slc],
            fillstyles=(["full", "full"] + list(_cycle_for(star_fillstyles, nstars)))[slc],
            marker_sizes=3,
            labels=labels[slc])

    # Plot the raw spectra for each component as a background
    if show_combined_spectrum or show_component_spectra:
        def plot_spec(lams, flux, color, alpha, zorder=-100):
            vfv = flux.to(flux_unit, equivalencies=u.spectral() + u.spectral_density(lams))
            ax.plot(lams, vfv, c=color, alpha=alpha, lw=0.75, zorder=zorder)

        spec_lams = np.geomspace(*model_grid.wavelength_range, 5000) * model_grid.wavelength_unit
        mask = spec_lams >= sed[sed_lambda_colname].quantity.min() * 0.8
        mask &= spec_lams <= sed[sed_lambda_colname].quantity.max() * 1.2
        comb_spec_flux = np.zeros((sum(mask)), dtype=float)
        for (teff, logg, rad, dist, av), c in zip(iterate_theta(theta_noms),
                                                _cycle_for(star_colors, nstars)):
            spec_flux = model_grid.get_fluxes(wavelengths=spec_lams[mask].value, teff=teff,
                                              logg=logg, radius=rad, distance=dist, av=av)
            comb_spec_flux += spec_flux
            if show_component_spectra:
                plot_spec(spec_lams[mask], spec_flux * model_grid.flux_unit, c, 0.25)
        if show_combined_spectrum:
            plot_spec(spec_lams[mask], comb_spec_flux * model_grid.flux_unit, comb_color, 0.75)
        ax.set(xscale="log", xlabel=xlabel, yscale="log", ylabel=ylabel)

    if show_grid:
        ax.grid(True, which="both", axis="both", alpha=0.33, color="lightgray")
    legend_loc = "best" if labels is not None and any(l is not None for l in labels) else None
    format_axes(ax, legend_loc=legend_loc, **format_kwargs)
    return fig


def plot_sed_on_axes(ax: _Axes,
                     x: u.Quantity,
                     fluxes: List[u.Quantity],
                     flux_errs: List[u.Quantity],
                     fmts: List[str],
                     fillstyles: Union[List[str], str]="full",
                     marker_sizes: Union[List[float], float]=5.0,
                     alphas: Union[List[float], float]=0.75,
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
    :sizes: the marker sizes for each set of fluxes
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
    if isinstance(marker_sizes, Number):
        marker_sizes = [marker_sizes] * len(fluxes)
    if isinstance(alphas, Number):
        alphas = [alphas] * len(fluxes)

    lams = x.to(u.um, equivalencies=u.spectral())
    with u.set_enabled_equivalencies(u.spectral() + u.spectral_density(lams)):
        for flux, flux_err, fmt, fs, ms, alpha, label \
                in zip(fluxes, flux_errs, fmts, fillstyles, marker_sizes, alphas, labels):
            vfv = None if flux is None else flux.to(flux_unit)
            vfv_err = None if flux_err is None else flux_err.to(flux_unit)
            if vfv is not None:
                ax.errorbar(lams, vfv, vfv_err, fmt=fmt, fillstyle=fs, ms=ms,
                            mew=0.75, elinewidth=0.75, alpha=alpha, label=label)


def plot_hr_diagram(teffs: ArrayLike,
                    luminosities: ArrayLike,
                    labels: ArrayLike=None,
                    plot_zams: bool=False,
                    **format_kwargs) -> _Figure:
    """
    Plots a log(L) vs log(T_eff) Hertzsprung-Russell diagram with an optional ZAMS line.
    Returns the figure of the plot for the calling code to show or save.

    :teffs: mass values to plot on the x-axis in shape (#sets, #teffs) or (#teffs) for 1 set
    :luminosities: radius values to plot on the y-axis in shape (#sets, #lums) or (#lums) for 1 set
    :labels: optional labels text for each set (if multiple sets) or item (if a single set)
    :plot_zams: whether or not to include a zero age main-sequence line on the figure
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the Figure
    """
    # Masses & radii both support multiple sets, but they must be the same shape
    if teffs.shape != luminosities.shape:
        raise ValueError("teffs and luminosities are not of the same shape")
    if labels is not None and len(labels) != teffs.shape[0]:
        raise ValueError("labels do not match the teffs or luminosities")

    # Smaller markers the more items there are to be plotted
    ms = max(2, 7 - np.log10(teffs.shape[-1]))

    teff_noms, teff_errs = nominal_values(teffs), std_devs(teffs)
    lum_noms, lum_errs = nominal_values(luminosities), std_devs(luminosities)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    labels = labels or [None] * teffs.shape[0]
    fmts = _cycle_for(["o", "o", "d", "d", "h", "h", "p", "p"], teffs.shape[0])
    fillstyles = _cycle_for(["full", "none"], teffs.shape[0])
    for ix, (teffn, teffe, lumn, lume, label, fmt, fs) in enumerate(zip(teff_noms, teff_errs,
                                                                    lum_noms, lum_errs,
                                                                    labels, fmts, fillstyles)):
        ax.errorbar(x=teffn, xerr=teffe, y=lumn, yerr=lume,
                    fmt=fmt, ms=ms, lw=1.0, markeredgewidth=1.0,
                    fillstyle=fs, zorder=ix, label=label)

    xlim = (min(3000, max(np.min(teff_noms - teff_errs) * 0.8, 1e-3)),
            max(20000, np.max(teff_noms + teff_errs) * 1.2))
    ylim = (min(0.001, max(np.min(lum_noms - lum_errs) * 0.8, 1e-3)),
            max(5000, np.max(lum_noms + lum_errs) * 1.2))
    ax.set(xlabel=r"$\log{(T_{\rm eff}\,/\,{\rm K})}$", xscale="log", xlim=xlim,
           ylabel=r"$\log{(L\,/\,{\rm L_{\odot}})}$", yscale="log", ylim=ylim)

    xticks = [x for x in [3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8] if min(xlim)<10**x<max(xlim)]
    ax.set_xticks([10**x for x in xticks], minor=False)
    ax.set_xticklabels(xticks, minor=False)

    yticks = [y for y in [-3, -2, -1, 0, 1, 2, 3, 4, 5] if min(ylim)<10**y<max(ylim)]
    ax.set_yticks([10**y for y in yticks], minor=False)
    ax.set_yticklabels(yticks, minor=False)

    if plot_zams:
        zams = _get_solar_isochrone_eep_values(eep=202, phase=0.0, cols=["log_Teff", "log_L"])
        zteff = np.linspace(zams[0].min(), zams[0].max(), 250)
        zsort = np.argsort(zams[0])
        zlum = make_interp_spline(zams[0, zsort], zams[1, zsort], k=1)(zteff) # smoothing
        ax.plot(10**zteff, 10**zlum, ls="--", lw=1, c="k", zorder=-100, alpha=0.5, label="ZAMS")

    format_axes(ax, **format_kwargs)
    ax.tick_params(axis="x", which="minor", top=False, bottom=False, labelbottom=False)
    return fig


def plot_predictions_vs_labels(fitted_values: ArrayLike,
                               label_values: ArrayLike,
                               captions: ArrayLike,
                               xlabel_prefix: str="label",
                               ylabel_prefix: str="fitted",
                               hl_mask1: np.ndarray[bool]=None,
                               hl_mask2: np.ndarray[bool]=None,
                               hl_mask3: np.ndarray[bool]=None,
                               fill_mask: np.ndarray[bool]=None,
                               cols: int=2,
                               big_markers: bool=None) -> _Figure:
    """
    Will create a plot figure with a grid of axes, one per label, showing the
    fitted vs label values. It is up to calling code to show or save the figure.

    :fitted_values: the fitted values structured array
    :label_values: the known/label values structured array
    :captions: the plot captions - the name of each parameter
    :xlabel_prefix: the prefix text for the labels/x-axis label
    :ylabel_prefix: the prefix text for the predictions/y-axis label
    :hl_mask1: optional mask for targets to be plotted with 1st highlight marker (square)
    :hl_mask2: optional mask for targets to be plotted with 2nd highlight marker (diamond)
    :hl_mask3: optional mask for targets to be plotted with 3rd highlight marker (pentagon)
    :fill_mask: optional mask for targets to be plotted with a filled marker
    :big_markers: if True, or if not set and inst count < 100, then plot larger markers
    :returns: the Figure
    """
    param_count = len(list(fitted_values.dtype.names))
    inst_count = fitted_values.size
    if label_values.size != inst_count:
        raise ValueError("label_values is a different size to the fitted_values")
    if hl_mask1 is not None and hl_mask1.size != inst_count:
        raise ValueError("hl_mask1 is a different size to the fitted_values")
    if hl_mask2 is not None and hl_mask2.size != inst_count:
        raise ValueError("hl_mask2 is a different size to the fitted_values")
    if hl_mask3 is not None and hl_mask3.size != inst_count:
        raise ValueError("hl_mask3 is a different size to the fitted_values")
    if fill_mask is not None and hl_mask3.size != inst_count:
        raise ValueError("fill_mask is a different size to the fitted_values")
    if captions.shape[0] != param_count:
        raise ValueError("captions is of a different length to the number of params")

    # Colours to use to ensure consistency across plots.
    # Attempted to select for accessibility (i.e.: contrast, consideration of colour blindness)
    #    base colour (lighter); darker for alternatives/highlights
    plot_colors = [ "tab:blue", "tab:orange", "darkgreen", "darkred", "k" ]
    ref_line_color = "darkgray"

    if hl_mask1 is None:
        hl_mask1 = np.zeros((inst_count), dtype=bool)
    if hl_mask2 is None:
        hl_mask2 = np.zeros((inst_count), dtype=bool)
    if hl_mask3 is None:
        hl_mask3 = np.zeros((inst_count), dtype=bool)
    if fill_mask is None:
        fill_mask = np.zeros((inst_count), dtype=bool)
    fillstyles = np.where(fill_mask, "full", "none")

    # Special aspect ratio for each axes of 3.0:2.9, slightly wider than high, to look balanced
    # with a sqaure plot area and slightly more width for y-tick labels than those for x-ticks.
    rows = int(np.ceil(param_count / cols))
    fig, axes = plt.subplots(rows, cols, constrained_layout=True,
                             figsize=(cols * 2.7, rows * 2.7))
    axes = axes.flatten()

    # The markers, marker sizes and alpha values are different depending on small/large dataset
    if big_markers or (big_markers is None and inst_count < 100):
        fmt = ["o", "s", "d", "p"]
        c = [plot_colors[0], plot_colors[4], plot_colors[4], plot_colors[4]]
        ms = [7.0, 8.5, 9.5, 10.5]
        alpha = [(0.66 if np.any(hl_mask1 | hl_mask2 | hl_mask3) else 1.0), 1.0, 1.0, 1.0]
    else:
        fmt = ["o", "o", "o", "o"]
        c = [plot_colors[0], plot_colors[1], plot_colors[4], plot_colors[4]]
        ms = [3.0, 3.0, 3.0, 3.0]
        alpha = [0.25, 0.50, 0.75, 0.75]

    for (ax, param, caption) in zip_longest(axes.flatten(), fitted_values.dtype.names, captions):
        if param:
            lbl_vals = nominal_values(label_values[param])
            lbl_errs = std_devs(label_values[param])
            fit_vals = nominal_values(fitted_values[param])
            fit_errs = std_devs(fitted_values[param])

            # Set the "view" x & y limits over the data and draw a diagonal line for "exact" match
            vmin = np.nan_to_num(min(lbl_vals.min(), fit_vals.min())) # pylint: disable=nested-min-max
            vmax = np.nan_to_num(max(lbl_vals.max(), fit_vals.max())) # pylint: disable=nested-min-max
            if param.startswith("Teff"):
                vmin = min(vmin, 2800)
                vmax = max(vmax, 28000)
            elif param.startswith("R"):
                vmin = min(0.2, vmin)
                vmax = max(5.0, vmax)
            vpad = 0.05 * (vmax - vmin)
            vdiag = (vmin - vpad, vmax + vpad)
            vrange = max(vdiag) - min(vdiag)
            ax.plot(vdiag, vdiag, color=ref_line_color, linestyle="--", linewidth=1.0, zorder=-10)

            # in order of increasing z
            non_hl_mask = ~(hl_mask1 | hl_mask2 | hl_mask3)
            for fix, mask in enumerate([non_hl_mask, hl_mask1, hl_mask2, hl_mask3]):
                for x, y, xerr, yerr, fs in zip(lbl_vals[mask], fit_vals[mask],
                                                lbl_errs[mask], fit_errs[mask], fillstyles[mask]):
                    ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, capsize=None,
                                c=c[fix], lw=ms[fix]/7.5, markeredgewidth=ms[fix]/7.5,
                                fmt=fmt[fix], ms=ms[fix]*0.66, alpha=alpha[fix], fillstyle=fs)

            format_axes(ax, xlim=vdiag, ylim=vdiag, xlabel=f"{xlabel_prefix} {caption}",
                        ylabel=f"{ylabel_prefix} {caption}")

            # Make sure the plot areas are squared and have similar label areas.
            ax.set_aspect("equal", "box")
            ax.tick_params("y", rotation=90)

            # We want up to 5 tick labels at suitable points across the range of values.
            maj_tick_labels = None
            if param.startswith("Teff"):
                maj_ticks = [5000, 10000, 15000, 20000, 25000]
                maj_tick_labels = ["5", "10", "15", "20", "25"]
                ax.set(xlabel=ax.get_xlabel().replace("K", r"1000\,K"),
                       ylabel=ax.get_ylabel().replace("K", r"1000\,K"))
            elif param.startswith("R"):
                maj_ticks = [1.0, 2.0, 3.0, 4.0, 5.0]
            else:
                # Adapt to the view range and finds a step which will give 4, 5 or 6 ticks.
                # Suspect logic; may not work universally but it's good enough for current results.
                for tick_step in [0.1, 0.5, 1, 2, 2.5, 5, 10]:
                    if vrange / tick_step < 6.5:
                        break
            ax.set_xticks(maj_ticks, minor=False)
            ax.set_yticks(maj_ticks, minor=False)
            if maj_tick_labels is not None:
                # Override default labelling
                ax.set_xticklabels(maj_tick_labels, minor=False)
                ax.set_yticklabels(maj_tick_labels, minor=False)
        else:
            ax.axis("off") # remove the unused ax
    return fig


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


def _get_solar_isochrone_eep_values(eep: int, phase: int, cols: List[str]) -> ArrayLike:
    """
    Gets the requested column values from the solar metallicity MIST isochrone,
    searching by eep and phase.

    Common eep values are 202 (ZAMS) & 453 (TAMS) with phase 0.0

    :iso: the MIST ISO to search
    :eep: the eep (equivalent evolutionary point) to find across the iso
    :phase: the phase to find across the iso, where 0.0 is main-sequence
    :cols: the columns to return the values for
    :returns: the requested data
    """
    iso_file = _this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos" \
                        / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
    iso = ISO(str(iso_file), verbose=False)

    rows = (ab[(ab["EEP"]==eep) & (ab["phase"]==phase)] for ab in iso.isos if eep in ab["EEP"])
    return np.array([list(row[0][cols]) for row in rows if len(row) > 0]).transpose()
