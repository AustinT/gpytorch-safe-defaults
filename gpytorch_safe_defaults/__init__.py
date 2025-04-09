"""
gpytorch_safe_defaults - A package to set safe defaults for linear_operator computations.
"""


class SafeDefaults:
    """A class to manage safe defaults for linear_operator computations."""

    def __call__(self) -> None:
        """Set safe defaults for linear_operator computations."""
        try:
            import linear_operator
        except ImportError:
            raise ImportError("linear_operator is not installed. Please install it with: pip install linear_operator")

        linear_operator.settings._fast_covar_root_decomposition._default = False
        linear_operator.settings._fast_log_prob._default = False
        linear_operator.settings._fast_solves._default = False

    def __enter__(self) -> None:
        """Context manager to temporarily set safe defaults."""
        try:
            import linear_operator
        except ImportError:
            raise ImportError("linear_operator is not installed. Please install it with: pip install linear_operator")

        with linear_operator.settings.fast_computations(False, False, False):
            yield

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        pass


safe_defaults = SafeDefaults()
