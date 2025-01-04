# partial-rankings

Paired comparisons are a standard method to infer a ranking between a series of players/actors. A shortcoming of many of these methods is that they lack mechanisms that allow for partial rankings --rankings where multiple nodes can have the same rank. This package contains models to infer partial rankings from pairwise comparisons as described in **PREPRINT**.

## Project organization

```
├── environment.yml    <- Conda environment configuration file
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile
├── README.md          <- This file.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- Final data sets.
│   └── raw            <- Original data sets (wolf data set).
│
├── example.ipyb       <- Jupyter notebook containing an example use case of the
│                         partial_rankings algorithm when applied to a set of dominance 
│                         interactions among a pack of wolves.
│
├── pyproject.toml     <- Project configuration file with package metadata
│
├── requirements.txt   <- Requirements file for reproducing the analysis environment.
│
├── setup.cfg          <- Configuration file for flake8
│
└── partial_rankings   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes partial_rankings a Python module
    │
    ├── dataset.py              <- Code to read or generate data
    │
    ├── decos.py                <- Useful decorators
    │
    ├── model.py                <- Code to fit partial rankings algorithm
    │
    ├── preprocessing.py        <- Code to extract information from match lists
    │
    └── utils.py                <- Utility functions
```

## Installation
The ``partial-rankings`` package can be installed through pip:

```bash
pip install partial-rankings
```

To ensure that all dependencies are correctly installed it is recommended to create a Conda envrionment from the ``envrionment.yml`` file by running

```bash
conda env create --file=envrionment.yml
```

which will install the ``partial-rankings`` package along with all of its dependencies.

## Typical usage
Once the package has been installed it can be imported as

```python
import partial_rakings
```

Below is a typical use case:
```python
from partial_rankings.dataset import read_matchlist
from partial_rankings.model import partial_rankings
from partial_rankings.preprocessing import get_N, get_M, get_edges
```

```python
# Load match list
matchlist = read_matchlist("../data/raw/match_lists/wolf.txt")

# Extract algorithm inputs
N = get_N(matchlist)  # Number of players
M = get_M(matchlist)  # Number of matches
e_out, e_in = get_edges(matchlist)  # Out and in edges

# Fit model
model_fit = partial_rankings(N, M, e_out, e_in, full_trace=True)
```

See ``example.ipynb`` for further details.

--------
