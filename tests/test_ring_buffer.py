"""
Tests for the RingBuffer utility class.

Coverage:
- Initialization: shape, capacity, dtype, initial state
- Append: single item, multiple items, wraparound, dtype coercion
- Retrieval: get_ordered ordering, get_last correctness
- Edge cases: empty buffer, capacity-1 buffer, multiple wraparounds
"""

import numpy as np
import pytest

from dhbv2.utils import RingBuffer


# ---------------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------------- #


class TestRingBufferInit:
    """Verify RingBuffer initialization and defaults."""

    def test_shape_and_capacity(self):
        """Buffer capacity should match first dimension of shape."""
        buf = RingBuffer(shape=(10, 1, 3))
        assert buf.capacity == 10
        assert buf.buffer.shape == (10, 1, 3)

    def test_default_dtype(self):
        """Default dtype should be float64."""
        buf = RingBuffer(shape=(5, 1, 2))
        assert buf.buffer.dtype == np.float64

    def test_custom_dtype(self):
        """Buffer should respect custom dtype."""
        buf = RingBuffer(shape=(5, 1, 2), dtype=np.float32)
        assert buf.buffer.dtype == np.float32

    def test_initial_state(self):
        """Buffer should start empty with ptr=0 and is_full=False."""
        buf = RingBuffer(shape=(5, 1, 2))
        assert buf.ptr == 0
        assert buf.is_full is False
        assert len(buf) == 0

    def test_buffer_initialized_to_zeros(self):
        """Buffer contents should be all zeros initially."""
        buf = RingBuffer(shape=(5, 1, 2))
        np.testing.assert_array_equal(buf.buffer, 0.0)

    def test_1d_shape(self):
        """Buffer should work with simple 1D shape."""
        buf = RingBuffer(shape=(3, 4))
        assert buf.capacity == 3
        assert buf.buffer.shape == (3, 4)


class TestRingBufferAppend:
    """Verify append operations and circular wraparound."""

    def test_single_append_advances_pointer(self):
        """Single append should advance pointer and increase length."""
        buf = RingBuffer(shape=(5, 1, 2))
        buf.append(np.array([[1.0, 2.0]]))
        assert buf.ptr == 1
        assert len(buf) == 1
        assert buf.is_full is False

    def test_multiple_appends(self):
        """Multiple appends should advance pointer accordingly."""
        buf = RingBuffer(shape=(5, 1, 2))
        for i in range(3):
            buf.append(np.array([[float(i), float(i)]]))
        assert buf.ptr == 3
        assert len(buf) == 3
        assert buf.is_full is False

    def test_fill_to_capacity_marks_full(self):
        """Filling to capacity should set is_full=True and reset ptr to 0."""
        buf = RingBuffer(shape=(5, 1, 2))
        for i in range(5):
            buf.append(np.array([[float(i), float(i)]]))
        assert buf.is_full is True
        assert buf.ptr == 0
        assert len(buf) == 5

    def test_wraparound_overwrites_oldest(self):
        """Appending past capacity should overwrite oldest entries."""
        buf = RingBuffer(shape=(3, 1, 2))
        for i in range(5):
            buf.append(np.array([[float(i), float(i)]]))
        assert buf.is_full is True
        assert len(buf) == 3
        # After 5 appends into capacity-3 buffer: ptr = 5 % 3 = 2
        assert buf.ptr == 2

    def test_auto_convert_list_input(self):
        """Non-ndarray input should be auto-converted to buffer's dtype."""
        buf = RingBuffer(shape=(3, 2), dtype=np.float32)
        buf.append([1.0, 2.0])
        assert buf.buffer[0].dtype == np.float32
        np.testing.assert_array_almost_equal(buf.buffer[0], [1.0, 2.0])

    def test_dtype_preserved_on_append(self):
        """Input dtype should be cast to buffer's dtype on storage."""
        buf = RingBuffer(shape=(3, 1, 2), dtype=np.float32)
        buf.append(np.array([[1.0, 2.0]], dtype=np.float64))
        assert buf.buffer.dtype == np.float32


class TestRingBufferRetrieval:
    """Verify get_ordered and get_last return correct data."""

    def test_get_ordered_before_full(self):
        """get_ordered should return only appended items when not full."""
        buf = RingBuffer(shape=(5, 1, 2))
        for i in range(3):
            buf.append(np.array([[float(i), float(i * 10)]]))

        result = buf.get_ordered()
        assert result.shape == (3, 1, 2)
        np.testing.assert_array_equal(result[0, 0], [0.0, 0.0])
        np.testing.assert_array_equal(result[2, 0], [2.0, 20.0])

    def test_get_ordered_after_wraparound(self):
        """get_ordered should return oldest-to-newest after wraparound."""
        buf = RingBuffer(shape=(3, 1, 2))
        for i in range(5):
            buf.append(np.array([[float(i), float(i)]]))

        result = buf.get_ordered()
        assert result.shape == (3, 1, 2)
        # Items 0,1 overwritten; items 2,3,4 remain (oldest=2, newest=4)
        np.testing.assert_array_equal(result[:, 0, 0], [2.0, 3.0, 4.0])

    def test_get_ordered_after_multiple_wraps(self):
        """Ordering should be correct after many full wraparounds."""
        buf = RingBuffer(shape=(3, 1, 1))
        for i in range(10):
            buf.append(np.array([[float(i)]]))

        result = buf.get_ordered()
        np.testing.assert_array_equal(result[:, 0, 0], [7.0, 8.0, 9.0])

    def test_get_ordered_empty_buffer(self):
        """get_ordered on empty buffer should return array with 0 in time dim."""
        buf = RingBuffer(shape=(5, 1, 2))
        result = buf.get_ordered()
        assert result.shape == (0, 1, 2)

    def test_get_last_returns_most_recent(self):
        """get_last should return the single most recently appended item."""
        buf = RingBuffer(shape=(5, 1, 2))
        buf.append(np.array([[1.0, 2.0]]))
        buf.append(np.array([[3.0, 4.0]]))

        result = buf.get_last()
        assert result.shape == (1, 1, 2)
        np.testing.assert_array_equal(result[0, 0], [3.0, 4.0])

    def test_get_last_after_wraparound(self):
        """get_last should return correct item after wraparound."""
        buf = RingBuffer(shape=(3, 1, 2))
        for i in range(5):
            buf.append(np.array([[float(i), float(i)]]))

        result = buf.get_last()
        np.testing.assert_array_equal(result[0, 0], [4.0, 4.0])

    def test_get_last_empty_buffer_returns_zeros(self):
        """get_last on empty buffer should return zeros."""
        buf = RingBuffer(shape=(5, 1, 2))
        result = buf.get_last()
        assert result.shape == (1, 1, 2)
        np.testing.assert_array_equal(result, 0.0)


class TestRingBufferEdgeCases:
    """Edge cases and boundary conditions."""

    def test_capacity_one(self):
        """Buffer with capacity=1 should always hold the latest item."""
        buf = RingBuffer(shape=(1, 1, 2))
        buf.append(np.array([[1.0, 2.0]]))
        assert len(buf) == 1
        assert buf.is_full is True

        buf.append(np.array([[3.0, 4.0]]))
        result = buf.get_last()
        np.testing.assert_array_equal(result[0, 0], [3.0, 4.0])

        ordered = buf.get_ordered()
        assert ordered.shape == (1, 1, 2)
        np.testing.assert_array_equal(ordered[0, 0], [3.0, 4.0])

    def test_get_ordered_consistency_with_get_last(self):
        """Last element of get_ordered should match get_last."""
        buf = RingBuffer(shape=(5, 1, 3))
        for i in range(7):
            buf.append(np.array([[float(i), float(i), float(i)]]))

        ordered = buf.get_ordered()
        last = buf.get_last()
        np.testing.assert_array_equal(ordered[-1:], last)

    def test_length_never_exceeds_capacity(self):
        """len() should never exceed capacity regardless of append count."""
        buf = RingBuffer(shape=(4, 2))
        for i in range(100):
            buf.append(np.array([float(i), float(i)]))
            assert len(buf) <= 4

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_roundtrip(self, dtype):
        """Appended data should be retrievable with correct dtype."""
        buf = RingBuffer(shape=(3, 2), dtype=dtype)
        buf.append(np.array([1.5, 2.5], dtype=dtype))

        result = buf.get_last()
        assert result.dtype == dtype
        np.testing.assert_array_almost_equal(result[0], [1.5, 2.5])
