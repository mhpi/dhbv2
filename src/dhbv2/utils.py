import numpy as np
from numpy.typing import NDArray


class RingBuffer:
    """
    Fixed-size circular buffer.

    Handles rolling windows without memory reallocation and fragmentation.

    Parameters
    ----------
    shape
        Tuple defining the shape of the buffer, e.g., (timesteps, space, vars).
    dtype
        Data type of the buffer elements.
    """

    def __init__(
        self,
        shape: tuple,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.dtype = dtype

        self.buffer = np.zeros(shape, dtype=dtype)
        self.capacity = shape[0]
        self.ptr = 0
        self.is_full = False

    def append(self, item: NDArray) -> None:
        """Overwrite the oldest item with new data.

        Parameters
        ----------
        item
            New data to append to the buffer. Expected shape: (space, vars).
        """
        if not isinstance(item, np.ndarray):
            item = np.array(item, dtype=self.dtype)

        self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.is_full = True

    def get_ordered(self) -> NDArray:
        """Return buffer.

        Returns
        -------
        np.ndarray
            Buffer contents ordered from oldest to newest.
        """
        if not self.is_full:
            # Return data up to current ptr
            return self.buffer[: self.ptr]

        # Roll so the oldest data (currently at ptr) moves to index 0
        return np.roll(self.buffer, shift=-self.ptr, axis=0)

    def get_last(self) -> NDArray:
        """Get the most recently added item.

        Returns
        -------
        NDArray
            The last item added to the buffer. Shape (1, space, vars).
        """
        if self.ptr == 0 and not self.is_full:
            # Return empty if no items have been added
            return np.zeros((1, *self.buffer.shape[1:]), dtype=self.buffer.dtype)

        # The last item added is at ptr - 1
        idx = (self.ptr - 1) % self.capacity
        return self.buffer[idx : idx + 1]

    def __len__(self):
        return self.capacity if self.is_full else self.ptr
