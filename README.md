# gpytorch_safe_defaults

A small package to set safe defaults for linear_operator computations in GPyTorch
(turning off many approximations which are on by default).

## Installation

```bash
pip install gpytorch_safe_defaults
```

## Usage

The package provides a single object `safe_defaults` that can be used in two ways:

1. As a function to set safe defaults globally:

```python
from gpytorch_safe_defaults import safe_defaults

safe_defaults()  # Sets safe defaults globally
```

2. As a context manager to temporarily set safe defaults:

```python
from gpytorch_safe_defaults import safe_defaults

with safe_defaults:
    # Your code here with safe defaults
    pass
```

## What it does

When used, this package sets the following linear_operator settings to `False`:
- `_fast_covar_root_decomposition._default`
- `_fast_log_prob._default`
- `_fast_solves._default`

This ensures more numerically stable computations in GPyTorch, though potentially at the cost of some performance.

## Example

```python
import torch
import gpytorch
from gpytorch_safe_defaults import safe_defaults

# Without safe defaults
model1 = gpytorch.models.ExactGP(...)  # Uses default settings

# With safe defaults
with safe_defaults:
    model2 = gpytorch.models.ExactGP(...)  # Uses safe defaults
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
