import numpy as np
import pytest

from hmm_simulator.sparsify import sample_screenings

N_TIMEPOINTS = 321


def _make_dense_profiles(n_samples=10, seed=0):
    """Build simple synthetic dense profiles for testing."""
    np.random.seed(seed)
    X = np.zeros((n_samples, N_TIMEPOINTS), dtype=float)
    for i in range(n_samples):
        start = np.random.randint(10, 80)
        end = np.random.randint(start + 20, 150)
        X[i, start:end] = np.random.randint(1, 5, size=end - start)
    return X


class TestSampleScreenings:
    def setup_method(self):
        self.X = _make_dense_profiles()

    def test_returns_ndarray(self):
        result = sample_screenings(self.X, stepsize=2)
        assert isinstance(result, np.ndarray)

    def test_output_has_correct_timepoints(self):
        result = sample_screenings(self.X, stepsize=2)
        assert result.shape[1] == N_TIMEPOINTS

    def test_stepsize_1_preserves_all_values(self):
        """stepsize=1 keeps every timepoint — non-zero counts should not decrease."""
        result = sample_screenings(self.X, stepsize=1)
        for x_orig, x_sparse in zip(self.X, result):
            orig_nonzero = set(np.where(x_orig > 0)[0])
            sparse_nonzero = set(np.where(x_sparse > 0)[0])
            assert sparse_nonzero.issubset(orig_nonzero)

    def test_stepsize_reduces_observations(self):
        """A larger stepsize should yield fewer non-zero entries per profile."""
        result_fine = sample_screenings(self.X, stepsize=1)
        result_coarse = sample_screenings(self.X, stepsize=5)
        fine_count = (result_fine != 0).sum()
        coarse_count = (result_coarse != 0).sum()
        assert coarse_count <= fine_count

    def test_kept_values_match_original(self):
        """Non-zero values in the sparse output must equal the original."""
        result = sample_screenings(self.X, stepsize=3)
        for x_orig, x_sparse in zip(self.X, result):
            kept = np.where(x_sparse > 0)[0]
            np.testing.assert_array_equal(x_sparse[kept], x_orig[kept])

    def test_no_values_outside_original_window(self):
        """Sparse profiles must not have non-zero values where original was zero."""
        result = sample_screenings(self.X, stepsize=2)
        for x_orig, x_sparse in zip(self.X, result):
            zero_in_orig = x_orig == 0
            assert np.all(x_sparse[zero_in_orig] == 0)

    def test_with_uniform_proba(self):
        proba = np.ones(N_TIMEPOINTS) / N_TIMEPOINTS
        result = sample_screenings(
            self.X,
            stepsize=2,
            proba_init_age=proba,
            proba_dropout=proba,
        )
        assert isinstance(result, np.ndarray)
        if len(result) > 0:
            assert result.shape[1] == N_TIMEPOINTS

    def test_all_zero_profiles_excluded(self):
        """Profiles where all kept values are zero should be dropped."""
        X = np.zeros((5, N_TIMEPOINTS))
        # All-zero input — every profile will be skipped.
        result = sample_screenings(X, stepsize=1)
        assert len(result) == 0
