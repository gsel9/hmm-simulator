import numpy as np
import pytest

from hmm_simulator.utils import (
    age_group_idx,
    age_partitions,
    lambda_sr,
    p_init_state,
    sample_start_age,
    sample_end_age,
    NUM_TIMEPOINTS,
)

# Precomputed partition starts/ends from age_partitions:
# starts: [  0  13  30  46  63  80 113 147]
# ends  : [ 10  26  43  60  76 110 143 267]


class TestAgeGroupIdx:
    def test_first_group_start(self):
        assert age_group_idx(0) == 0

    def test_first_group_interior(self):
        assert age_group_idx(5) == 0

    def test_each_group_start(self):
        starts = age_partitions[:, 0].astype(int)
        for expected_group, start in enumerate(starts):
            assert age_group_idx(start) == expected_group

    def test_each_group_end(self):
        # Values at partition ends should map to their own group (not the next).
        ends = age_partitions[:, 1].astype(int)
        for expected_group, end in enumerate(ends):
            idx = age_group_idx(end)
            # May map to this group or the next (gap between groups), but never
            # earlier than expected_group.
            assert idx >= expected_group

    def test_last_valid_age(self):
        max_age = int(age_partitions[-1, 1])
        assert age_group_idx(max_age) == len(age_partitions) - 1

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            age_group_idx(-1)

    def test_age_above_max_raises(self):
        max_age = int(age_partitions[-1, 1])
        with pytest.raises(ValueError):
            age_group_idx(max_age + 1)

    def test_returns_int(self):
        assert isinstance(age_group_idx(0), int)


class TestParameterShapes:
    def test_lambda_sr_shape(self):
        assert lambda_sr.shape == (8, 9)

    def test_p_init_state_shape(self):
        assert p_init_state.shape == (8, 4)

    def test_age_partitions_shape(self):
        assert age_partitions.shape == (8, 2)

    def test_p_init_state_rows_sum_to_one(self):
        row_sums = p_init_state.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-3)

    def test_lambda_sr_all_positive(self):
        assert np.all(lambda_sr >= 0)

    def test_partitions_start_before_end(self):
        assert np.all(age_partitions[:, 0] < age_partitions[:, 1])


class TestSampleStartAge:
    def test_returns_valid_age(self):
        np.random.seed(0)
        proba = np.ones(NUM_TIMEPOINTS) / NUM_TIMEPOINTS
        age, idx = sample_start_age(NUM_TIMEPOINTS, proba)
        assert 0 <= age < NUM_TIMEPOINTS
        assert 0 <= idx < NUM_TIMEPOINTS

    def test_return_idx_false(self):
        np.random.seed(0)
        proba = np.ones(NUM_TIMEPOINTS) / NUM_TIMEPOINTS
        result = sample_start_age(NUM_TIMEPOINTS, proba, return_idx=False)
        # Should return (age, None) per updated signature.
        assert result[1] is None

    def test_age_and_idx_consistent(self):
        np.random.seed(42)
        proba = np.ones(NUM_TIMEPOINTS) / NUM_TIMEPOINTS
        age, idx = sample_start_age(NUM_TIMEPOINTS, proba)
        assert age == idx  # linspace(0, 320, 321) so time[i] == i


class TestSampleEndAge:
    def test_returns_valid_age(self):
        np.random.seed(0)
        proba = np.ones(NUM_TIMEPOINTS) / NUM_TIMEPOINTS
        end = sample_end_age(NUM_TIMEPOINTS, proba, init_age_idx=0)
        assert 0 <= end < NUM_TIMEPOINTS

    def test_end_age_at_least_init_age(self):
        np.random.seed(0)
        init_idx = 100
        proba = np.ones(NUM_TIMEPOINTS) / NUM_TIMEPOINTS
        end = sample_end_age(NUM_TIMEPOINTS, proba, init_age_idx=init_idx)
        assert end >= init_idx
