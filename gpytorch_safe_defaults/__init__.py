"""
gpytorch_safe_defaults - A package to set safe defaults for linear_operator computations.
"""


class SafeDefaults:
    """A class to manage safe defaults for linear_operator computations."""

    def _import_linear_operator(self):
        """Helper method to import linear_operator with proper error handling."""
        try:
            import linear_operator

            return linear_operator
        except ImportError as err:
            raise ImportError(
                "linear_operator is not installed. Please install it with: pip install linear_operator"
            ) from err

    def __call__(self) -> None:
        """Set safe defaults for linear_operator computations."""
        linear_operator = self._import_linear_operator()
        linear_operator.settings._fast_covar_root_decomposition._default = False
        linear_operator.settings._fast_log_prob._default = False
        linear_operator.settings._fast_solves._default = False

    def __enter__(self) -> None:
        """Context manager to temporarily set safe defaults."""
        linear_operator = self._import_linear_operator()
        with linear_operator.settings.fast_computations(False, False, False):
            yield

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        pass


safe_defaults = SafeDefaults()
