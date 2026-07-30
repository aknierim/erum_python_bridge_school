import numpy as np
import pytest

from erum_demo.functions import exp_fit, lin_fit, sigmoid_fit


class TestLinFit:
    """Tests for lin_fit."""

    @pytest.mark.parametrize(
        "x,a,b,expected",
        [
            (
                np.array([0.0, 1.0, 2.0, 3.0]),
                2.0,
                1.0,
                np.array([1.0, 3.0, 5.0, 7.0]),
            ),
            (
                np.array([-2.0, -1.0, 0.0, 1.0]),
                -3.0,
                4.0,
                np.array([10.0, 7.0, 4.0, 1.0]),
            ),
            (
                np.array([5.0, 10.0, 15.0]),
                0.0,
                7.5,
                np.array([7.5, 7.5, 7.5]),
            ),
        ],
    )
    def test_returns_expected(self, x, a, b, expected):
        result = lin_fit(x, a, b)

        np.testing.assert_allclose(result, expected)

    def test_preserves_shape(self):
        x = np.array([[0.0, 1.0], [2.0, 3.0]])

        result = lin_fit(x, a=2.0, b=1.0)

        assert result.shape == x.shape

    def test_no_input_modification(self):
        x = np.array([1.0, 2.0, 3.0])
        x_original = x.copy()

        lin_fit(x, a=2.0, b=1.0)

        np.testing.assert_array_equal(x, x_original)

    def test_empty_array(self):
        x = np.array([])

        result = lin_fit(x, a=2.0, b=1.0)

        np.testing.assert_array_equal(result, np.array([]))


class TestExpFit:
    """Tests for exp_fit."""

    def test_returns_expected(self):
        x = np.array([0.0, 1.0, 2.0])
        a = 2.0
        b = 0.5
        c = 1.0

        expected = np.array(
            [
                3.0,
                2.0 * np.exp(-0.5) + 1.0,
                2.0 * np.exp(-1.0) + 1.0,
            ]
        )

        result = exp_fit(x, a, b, c)

        np.testing.assert_allclose(result, expected)

    def test_x_zero_returns_a_plus_c(self):
        x = np.array([0.0])
        a = 4.5
        b = 2.0
        c = -1.5

        result = exp_fit(x, a, b, c)

        np.testing.assert_allclose(result, np.array([a + c]))

    def test_preserves_shape(self):
        x = np.array([[0.0, 1.0], [2.0, 3.0]])

        result = exp_fit(x, a=2.0, b=0.5, c=1.0)

        assert result.shape == x.shape

    def test_no_input_modification(self):
        x = np.array([0.0, 1.0, 2.0])
        x_original = x.copy()

        exp_fit(x, a=2.0, b=0.5, c=1.0)

        np.testing.assert_array_equal(x, x_original)

    def test_empty_array(self):
        x = np.array([])

        result = exp_fit(x, a=2.0, b=0.5, c=1.0)

        np.testing.assert_array_equal(result, np.array([]))


class TestSigmoidFit:
    """Tests for sigmoid_fit."""

    def test_returns_expected(self):
        x = np.array([-1.0, 0.0, 1.0])
        a = 2.0
        b = 0.0
        c = 1.0

        expected = np.array(
            [
                2.0 / (1.0 + np.exp(1.0)) + 1.0,
                2.0 / 2.0 + 1.0,
                2.0 / (1.0 + np.exp(-1.0)) + 1.0,
            ]
        )

        result = sigmoid_fit(x, a, b, c)

        np.testing.assert_allclose(result, expected)

    def test_preserves_shape(self):
        x = np.array([[0.0, 1.0], [2.0, 3.0]])

        result = sigmoid_fit(x, a=2.0, b=1.0, c=0.5)

        assert result.shape == x.shape

    def test_no_input_modification(self):
        x = np.array([0.0, 1.0, 2.0])
        x_original = x.copy()

        sigmoid_fit(x, a=2.0, b=1.0, c=0.5)

        np.testing.assert_array_equal(x, x_original)

    def test_hempty_array(self):
        x = np.array([])

        result = sigmoid_fit(x, a=2.0, b=1.0, c=0.5)

        np.testing.assert_array_equal(result, np.array([]))
