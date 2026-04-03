"""Tests for hmm_generator.simulate_profile."""
import numpy as np

from hmm_simulator.hmm_generator import simulate_profile

N_TIMEPOINTS = 321
VALID_STATES = {0, 1, 2, 3, 4}

# Small window — keeps tests fast and stays within valid age-group bounds.
INIT_AGE = 0
AGE_MAX = 60


class TestSimulateProfile:
    """Tests for simulate_profile."""

    def test_returns_correct_length(self):
        """Output array length equals n_timepoints."""
        np.random.seed(0)
        x = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        assert len(x) == N_TIMEPOINTS

    def test_returns_ndarray(self):
        """Return type is np.ndarray."""
        np.random.seed(0)
        x = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        assert isinstance(x, np.ndarray)

    def test_values_are_valid_states(self):
        """All values belong to the set of valid states plus missing."""
        np.random.seed(0)
        x = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        unique = set(x.astype(int))
        assert unique.issubset(VALID_STATES), (
            f"Unexpected state values: {unique - VALID_STATES}"
        )

    def test_observed_window_is_non_zero(self):
        """Timepoints between init_age and age_max should contain states."""
        np.random.seed(0)
        x = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        window = x[INIT_AGE:AGE_MAX]
        assert np.any(window != 0)

    def test_before_init_age_is_missing(self):
        """Timepoints before init_age remain at the missing fill value."""
        np.random.seed(0)
        init_age = 10
        x = simulate_profile(N_TIMEPOINTS, init_age, AGE_MAX, missing=0)
        assert np.all(x[:init_age] == 0)

    def test_custom_missing_value_before_init(self):
        """Custom missing value is used for timepoints before init_age."""
        np.random.seed(0)
        init_age = 10
        x = simulate_profile(N_TIMEPOINTS, init_age, AGE_MAX, missing=-1)
        assert np.all(x[:init_age] == -1)

    def test_reproducible_with_seed(self):
        """Same seed produces identical profiles."""
        np.random.seed(3)
        x1 = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        np.random.seed(3)
        x2 = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        np.testing.assert_array_equal(x1, x2)

    def test_multiple_profiles_differ(self):
        """Two independently drawn profiles should (almost certainly) differ."""
        np.random.seed(0)
        x1 = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        x2 = simulate_profile(N_TIMEPOINTS, INIT_AGE, AGE_MAX)
        assert not np.array_equal(x1, x2)

    def test_various_age_windows(self):
        """Profile is valid for several age windows including across partitions."""
        np.random.seed(0)
        for init, end in [(0, 40), (0, 60), (13, 60), (30, 110)]:
            x = simulate_profile(N_TIMEPOINTS, init, end)
            assert len(x) == N_TIMEPOINTS
            assert set(x.astype(int)).issubset(VALID_STATES)
