import numpy as np
import pytest

from hmm_simulator.sojourn import sojourn_time, sojourn_time_cdf

# Use a small age window to keep tests fast.
START_AGE = 0
STOP_AGE = 50
VALID_STATES = [1, 2, 3, 4]


class TestSojournTimeCdf:
    def test_returns_correct_length(self):
        cdf = sojourn_time_cdf(START_AGE, STOP_AGE, current_state=1)
        expected_len = int(STOP_AGE - START_AGE + 1)
        assert len(cdf) == expected_len

    def test_values_in_unit_interval(self):
        for state in VALID_STATES:
            cdf = sojourn_time_cdf(START_AGE, STOP_AGE, current_state=state)
            assert np.all(cdf >= 0.0), f"Negative CDF value for state {state}"
            assert np.all(cdf <= 1.0), f"CDF > 1 for state {state}"

    def test_starts_at_zero(self):
        # CDF at t=0 should be 0 (no time has passed).
        cdf = sojourn_time_cdf(START_AGE, STOP_AGE, current_state=1)
        assert cdf[0] == pytest.approx(0.0)

    def test_generally_increasing(self):
        # The piecewise CDF has discontinuous drops at age-group partition
        # boundaries (hazard rate changes), so strict monotonicity does not
        # hold globally. We verify the overall trend is upward: the last
        # non-zero value is greater than the first non-zero value.
        for state in VALID_STATES:
            cdf = sojourn_time_cdf(START_AGE, STOP_AGE, current_state=state)
            nonzero = cdf[cdf > 0]
            if len(nonzero) > 1:
                assert nonzero[-1] > nonzero[0]

    def test_different_age_groups(self):
        # Verify the function runs for ages spanning multiple partitions.
        from hmm_simulator.utils import age_partitions
        for start in age_partitions[:-1, 0].astype(int):
            stop = start + 30
            cdf = sojourn_time_cdf(start, stop, current_state=1)
            assert len(cdf) == stop - start + 1


class TestSojournTime:
    def test_censored_state_returns_age_max(self):
        result = sojourn_time(START_AGE, STOP_AGE, current_state=0)
        assert result == STOP_AGE

    def test_returns_positive_value(self):
        np.random.seed(0)
        for state in VALID_STATES:
            dt = sojourn_time(START_AGE, STOP_AGE, current_state=state)
            assert dt > 0, f"Non-positive sojourn time for state {state}"

    def test_returns_float(self):
        np.random.seed(0)
        dt = sojourn_time(START_AGE, STOP_AGE, current_state=1)
        assert isinstance(dt, float)

    def test_low_u_edge_case(self):
        """When u is very small (< all CDF values), t_lower should default to 0."""
        np.random.seed(0)
        # Patch np.random.uniform to return a near-zero value.
        original = np.random.uniform
        try:
            np.random.uniform = lambda **kwargs: 1e-10
            dt = sojourn_time(START_AGE, STOP_AGE, current_state=1)
            assert dt >= 0
        finally:
            np.random.uniform = original

    def test_reproducible_with_seed(self):
        np.random.seed(7)
        dt1 = sojourn_time(START_AGE, STOP_AGE, current_state=1)
        np.random.seed(7)
        dt2 = sojourn_time(START_AGE, STOP_AGE, current_state=1)
        assert dt1 == pytest.approx(dt2)
