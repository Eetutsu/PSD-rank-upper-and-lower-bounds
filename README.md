# PSD-rank-upper-and-lower-bounds

PSD-rank-upper-and-lower-bounds is python program that calculates the lower bound and upper bound for PSD-rank for a nonnegative matrix using various methods found in two sources:
 
Some upper and lower bounds on PSD-rank [https://arxiv.org/pdf/1407.4308]()

POSITIVE SEMIDEFINITE RANK [https://arxiv.org/pdf/1407.4095]()

Also two heuristic methods are implemented to approximate the PSD-factorization for a nonnegative matrix.

Source: Algorithms for Positive Semidefinite Factorization [https://arxiv.org/pdf/1707.07953]()
 
## Installation

Clone the repository and install dependencies via [pip](https://pip.pypa.io/en/stable/): [Picos](https://picos-api.gitlab.io/picos/), [NumPy](https://numpy.org/)

```bash
pip install picos
pip install numpy
git clone https://github.com/Eetutsu/PSD-rank-upper-and-lower-bounds.git
```


## Files
### upper_bound.py
Contains all the implemented upper bounds
### lower_bound.py
Contains all the implemented lower bounds
### solve_PSD_rank.py
Calculates the bound for PSD-rank for a given matrix
### heuristic_methods.py
Contains all the implemented methods for PSD-factorization
## Usage

```python
from solve_PSD_rank import solve
from heuristic_methods import FPGPsd_facto

M = [
    [1 / 2, 1 / 2, 0, 0],
    [1 / 2, 0, 1 / 2, 0],
    [1 / 2, 0, 0, 1 / 2],
    [0, 1 / 2, 0, 1 / 2],
    [0, 0, 1 / 2, 1 / 2],
]

# returns 3
solve(M)

# returns two lists of 3x3 matrices {A_1,..,A_4} and {B_1,...,B_5} which are the PSD-factors of matrix M
FPGPsd_facto(M) 
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)