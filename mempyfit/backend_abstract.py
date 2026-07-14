
class FittingBackend:
    """Abstract base class for fitting backends.

    This class defines the minimal interface for backends that solve a
    fitting problem, such as SciPy optimizers or pyABC.

    Example:
        >>> backend = ScipyBackend(problem)
        >>> backend.run()
"""

    def __init__(self):
        None

    def __repr__(self):
        return f"FittingProblem={self.prob}"

    