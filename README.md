# sed_fit
SED fitting library suitable for single or multiple stars

## Using sed_fit as a package with pip
You can install sed_fit is as a pip package. Simply run the following within the context
of your own base or custom python virtual environment:

```sh
$ pip install git+https://github.com/SteveOv/sed_fit
```
This will install the fitter module, the pre-built stellar grids and any required support
libraries. With this setup you will be able to perform both minimize fitting and mcmc
sampling of SED observations against the pre-built stellar grids.

While the [binary_sed_fit.ipynb](../main/binary_sed_fit.ipynb) jupyter page is not
installed as part of the package, it can be viewed directly on GitHub where it offers
a useful tutorial on using the fitter and model grids.

## Setup of the runtime for the entire repo
Alternatively you can set up the entire code base, which has been developed within a
Python3 virtual environment supporting Python 3.9-3.12, matplotlib, astropy, astroquery,
lightkurve, emcee, and the custom [deblib](https://github.com/SteveOv/deblib) package
upon which the code is dependent. The dependencies are documented in the
[requirements.txt](../main/requirements.txt) file.

Having first cloned this GitHub repo, open a Terminal at the root of the local repo
and run the following commands. First to create and activate the venv;

```sh
$ python -m venv .sed_fit
$ source .sed_fit/bin/activate
```
Then run the following to set up the required packages:
```sh
$ pip install -r requirements.txt
```
You may need to install the jupyter kernel in the new venv if you wish to run
[binary_sed_fit.ipynb](../main/binary_sed_fit.ipynb):
```sh
$ ipython kernel install --user --name=.sed_fit
```

#### Alternative, conda virtual environment
To set up an `sed_fit` conda environment, from the root of the local repo run the
following command;
```sh
$ conda env create -f environment.yaml
```
You will need to activate the environment whenever you wish to run any of these modules.
Use the following command;
```sh
$ conda activate sed_fit
```

