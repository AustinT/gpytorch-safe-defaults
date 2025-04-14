# actually_exact_gpytorch

A small package to set safe defaults for linear_operator computations in GPyTorch
(turning off many approximations which are on by default).

## Installation

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

This ensures more numerically stable computations in GPyTorch, though potentially at the cost of some performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
