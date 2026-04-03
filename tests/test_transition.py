import numpy as np
import pytest

from hmm_simulator.transition import initial_state, legal_transitions, next_state
from hmm_simulator.utils import lambda_sr


# Pick a lambda row for a mid-range age group.
LAMBDAS = lambda_sr[0]

VALID_STATES = [1, 2, 3, 4]


class TestLegalTransitions:
    def test_state_0_returns_empty(self):
        result = legal_transitions(0, LAMBDAS)
        assert len(result) == 0

    def test_state_1_returns_two_transitions(self):
        result = legal_transitions(1, LAMBDAS)
        assert len(result) == 2

    def test_state_2_returns_three_transitions(self):
        result = legal_transitions(2, LAMBDAS)
        assert len(result) == 3

    def test_state_3_returns_three_transitions(self):
        result = legal_transitions(3, LAMBDAS)
        assert len(result) == 3

    def test_state_4_returns_two_transitions(self):
        result = legal_transitions(4, LAMBDAS)
        assert len(result) == 2

    def test_norm_sums_to_one(self):
        for state in VALID_STATES:
            result = legal_transitions(state, LAMBDAS, norm=True)
            np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_unnormed_all_positive(self):
        for state in VALID_STATES:
            result = legal_transitions(state, LAMBDAS, norm=False)
            assert np.all(result >= 0)

    def test_unknown_state_returns_empty(self):
        result = legal_transitions(99, LAMBDAS)
        assert len(result) == 0


class TestInitialState:
    def test_returns_valid_state(self):
        np.random.seed(0)
        for _ in range(50):
            state = initial_state(init_age=0)
            assert state in VALID_STATES

    def test_all_age_groups(self):
        """initial_state should work for any valid age group start."""
        np.random.seed(0)
        from hmm_simulator.utils import age_partitions
        for start in age_partitions[:, 0].astype(int):
            state = initial_state(init_age=start)
            assert state in VALID_STATES


class TestNextState:
    @pytest.mark.parametrize("current_state,allowed", [
        (1, {0, 2}),
        (2, {0, 1, 3}),
        (3, {0, 2, 4}),
        (4, {0, 1}),
    ])
    def test_next_state_within_allowed(self, current_state, allowed):
        np.random.seed(0)
        age = 0  # group 0
        for _ in range(100):
            result = next_state(age=age, current_state=current_state, censoring=0)
            assert result in allowed, (
                f"State {current_state} produced unexpected next state {result}"
            )

    def test_censored_state_returns_censoring(self):
        result = next_state(age=0, current_state=0, censoring=0)
        assert result == 0

    def test_custom_censoring_value(self):
        np.random.seed(0)
        # With a custom censoring value, ensure censoring sentinel is used.
        results = {next_state(age=0, current_state=1, censoring=9) for _ in range(200)}
        # Should only contain 2 (progression) or 9 (censoring).
        assert results.issubset({2, 9})
