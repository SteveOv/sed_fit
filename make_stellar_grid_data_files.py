""" A module for creating the shipped StellarGrid data files for the sed_fit package. """
# pylint: disable=protected-access
from pathlib import Path
from sed_fit.stellar_grids import StellarGrid, BtSettlGrid

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

    qteff = min(sgrid.teff_range)
    print(f"Test flux for 'GAIA/GAIA3:Gbp' filter, teff={qteff}, logg=4.0, metal=0, alpha=0:",
          ", ".join(f"{f:.3f}" for f in sgrid.get_filter_fluxes(["GAIA/GAIA3:Gbp"], qteff, 4, 0)),
          f"[{sgrid.flux_unit:unicode}]")

if __name__ == "__main__":
    # Download bt-settl-aggs ascii model grids from following url
    # https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss
    # then decompress the tgz contents into the ../.cache/.modelgrids/bt-settl-agss dir
    in_files = sorted((Path.cwd() / ".cache/.modelgrids/bt-settl-agss/").glob("lte*.dat.txt"))
    data_file = Path("./sed_fit/data/stellar_grids/bt-settl-agss/bt-settl-agss.npz")

    BtSettlGrid.make_grid_file(in_files, data_file)

    quick_test_stellar_grid_lookup(BtSettlGrid(data_file, verbose=True))
