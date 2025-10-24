# sed_fit
SED fitting library suitable for single or multiple stars

## Setup of runtime environment
This code base was developed within the context of a Python3 virtual environment which
supports Python 3.9-3.12, matplotlib, astropy, astroquery, lightkurve, emcee, and the custom
[deblib](https://github.com/SteveOv/deblib) package upon which the code is dependent.
The dependencies are documented in the [requirements.txt](../main/requirements.txt)
file.

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
You may need to install the jupyter kernel in the new venv:
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

