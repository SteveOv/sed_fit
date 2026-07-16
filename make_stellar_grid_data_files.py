""" A module for creating the shipped StellarGrid data files for the sed_fit package. """
# pylint: disable=protected-access
from pathlib import Path
from sed_fit.stellar_grids import StellarGrid, BtSettlGrid, KuruczGrid

def quick_test_stellar_grid_lookup(sgrid: StellarGrid) -> Path:
    """
    Quick test lookup with the requested model
    """
    # Test what has been saved
    print("Teffs:", ",".join(f"{t:.2f}" for t in sgrid._model_full_interp.grid[0]))
    print("loggs:", ",".join(f"{l:.2f}" for l in sgrid._model_full_interp.grid[1]))
    print("metals:", ",".join(f"{m:.2f}" for m in sgrid._model_full_interp.grid[2]))

    print( "Filters:", ", ".join(sgrid._filter_names_list))

    print(f"\nRanges: teff={sgrid.teff_range} {sgrid.teff_unit:unicode},",
          f"logg={sgrid.logg_range} {sgrid.logg_unit:unicode}, metal = {sgrid.metal_range}")

    for (teff,                      logg,                   rad,        dist) in [
        (min(sgrid.teff_range),     4.0,                    None,       None),
        (max(sgrid.teff_range),     4.0,                    None,       None),
        (6000.0,                    min(sgrid.logg_range),  1.0,        100.0),
        (6000.0,                    max(sgrid.logg_range),  1.0,        100.0),
        (5750.0,                    4.0,                    1.0,        100.0),
    ]:
        print( "Flux for filter 'GAIA/GAIA3:Gbp',",
              f"teff={teff}, logg={logg}, metal=0, alpha=0, R={rad}, dist={dist}:",
              f"{sgrid.get_filter_fluxes(['GAIA/GAIA3:Gbp'], teff, logg, 0, rad, dist)[0]:.6e}",
              f"{sgrid.flux_unit:unicode}")


if __name__ == "__main__":
    #
    # BtSettl AGSS Grid
    #
    # Download bt-settl-aggs ascii model grids from following url
    # https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
    # To create default data file;
    # - show all results for teff 2000 to 40000, logg 3.5 to 5.5 and metal -0.5 to 0.5.
    # - mark all ascii, click retrieve files (wait to assemble into archive file) then download
    # - decompress the archive contents into the ../.cache/.modelgrids/bt-settl-agss/ dir
    print()
    in_files = sorted((Path.cwd() / ".cache/.modelgrids/bt-settl-agss/").glob("lte*.dat.txt"))
    data_file = BtSettlGrid.DEF_DATA_FILE

    BtSettlGrid.make_grid_file(in_files, data_file)

    quick_test_stellar_grid_lookup(BtSettlGrid(data_file, use_quick_mode=False, verbose=True))

    #
    # Kurucz Grid
    #
    # Download Kurucz ODFNEW /NOVER ascii model grids from following url
    # https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=Kurucz2003all
    # To create default data file;
    # - show all results for teff 3500 to 30000, logg 3.0 to 5.0, metal -0.5 to 0.5 & alpha == 0.
    # - mark all ascii, click retrieve files (wait to assemble into archive file) then download
    # - decompress the archive contents into the ../.cache/.modelgrids/kurucz-odfnew-nover/ dir
    print()
    in_files = sorted((Path.cwd() / ".cache/.modelgrids/kurucz-odfnew-nover/").glob("*.fl.dat.txt"))
    data_file = KuruczGrid.DEF_DATA_FILE

    KuruczGrid.make_grid_file(in_files, data_file)

    quick_test_stellar_grid_lookup(KuruczGrid(data_file, use_quick_mode=False, verbose=True))
