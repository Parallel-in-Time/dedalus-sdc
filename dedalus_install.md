# Installation of Dedalus (V3)

> :warning: Only works with **Python 3** versions

## Directly from Github :

Installation command :

```bash
# Install dependencies
conda install -c conda-forge c-compiler "h5py=*=mpi*" "cython<3.0"

# Install Dedalus v3
conda uninstall --force dedalus
CC=mpicc pip install --no-cache --no-build-isolation http://github.com/dedalusproject/dedalus/zipball/master/
```

Update command :

```bash
CC=mpicc pip3 install --upgrade --force-reinstall --no-deps --no-cache --no-build-isolation http://github.com/dedalusproject/dedalus/zipball/master/
```

## Using a local git repository

Installation command :

```bash
git clone -b master https://github.com/DedalusProject/dedalus
cd dedalus
CC=mpicc pip3 install --no-cache --no-build-isolation .
```

Update command :

```bash
cd /path/to/dedalus/repo
git pull
CC=mpicc pip3 install --upgrade --force-reinstall --no-deps --no-cache --no-build-isolation .
```

## Uninstall

For both approaches :

```bash
pip uninstall dedalus
```

More details : https://dedalus-project.readthedocs.io/en/latest/pages/installation.html