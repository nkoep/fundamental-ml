import numpy as np


def ensure_random_state(random_state):
    """Derive PRNG from given random state.

    Parameters
    ----------
    random_state : int or numpy.random.Generator
        Either an integer used to seed numpy's default PRNG, or an already
        seeded PRNG.

    Returns
    -------
    random_state : numpy.random.Generator
        The derived PRNG.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
