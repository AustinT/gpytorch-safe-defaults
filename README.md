# Actually exact GPyTorch

<div align="center">
<i>A convenient way to make GPyTorch use exact GP inference!</i>
</div>
<br>

By default, GPyTorch uses a bunch of approximations to help GP inference run
faster than O(N^3). _Normally_ these work well, but there are many circumstances
when these approximations fail, most prominently when the condition number of the
covariance matrix is large. For this reason, some users prefer to turn the approximations off.
This package makes this easy. Instead of turning the approximations off one by one,
we provide a single object to turn them *all* off.

## Installation

Clone the repo, then install with pip:

```bash
pip install actually_exact_gpytorch
```

## Usage

The package provides a single object `exact_gpytorch` that can be used in two ways:

1. As a function to set safe defaults globally:

```python
from actually_exact_gpytorch import exact_gpytorch

exact_gpytorch()  # Sets safe defaults globally
```

2. As a context manager to temporarily set safe defaults:

```python
from actually_exact_gpytorch import exact_gpytorch

with exact_gpytorch:
    # Your code here with safe defaults
    pass
```

## What it does

When used, this package sets the following linear_operator settings to `False`:
- `_fast_covar_root_decomposition._default`
- `_fast_log_prob._default`
- `_fast_solves._default`

NOTE: future versions of this package may turn off more things (as needed).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
