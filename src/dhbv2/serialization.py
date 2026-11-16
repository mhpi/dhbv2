"""Support for ngen BMI Serialization Protocol (checkpoint/restore).

See NOAA-OWP/ngen/doc/BMI_SERIALIZATION_PROTOCOL.md. The protocol reserves four
BMI variables, resolved by name and never listed in the input/output variable
names. A BMI mixes in 'SerializableBmi' and implements how its state is
captured, validated, and applied; this module handles the reserved variables,
the payload format, and rollback on a failed restore.
"""

import hashlib
import json
import struct
import zlib
from typing import Any

import numpy as np
import torch

from dhbv2.log import log

SERIAL_CREATE = 'ngen::serialization_create'
SERIAL_FREE = 'ngen::serialization_free'
SERIAL_SIZE = 'ngen::serialization_size'
SERIAL_STATE = 'ngen::serialization_state'

SERIAL_VAR_UNITS = {
    SERIAL_CREATE: 'ngen::trigger',
    SERIAL_FREE: 'ngen::trigger',
    SERIAL_SIZE: 'bytes',
    SERIAL_STATE: 'ngen::opaque',
}

_VERSION = 1
_HEADER = struct.Struct('<8sII')

# Array dtype kinds a payload may hold: float, signed/unsigned int, bool.
_DTYPE_KINDS = 'fiub'

# Cap on the decompressed payload, so a corrupt stream cannot exhaust memory.
# A single-catchment payload decompresses to about 11 KB.
_MAX_BYTES = 64 * 1024**2


def _pack_arrays(value: Any, arrays: list[np.ndarray]) -> Any:
    """Swap every array in a nested container for an index into 'arrays'.

    Parameters
    ----------
    value
        Nested dicts, lists, and tuples of arrays, tensors, and scalars.
    arrays
        Receives the arrays in the order they are indexed.

    Returns
    -------
    Any
        The same structure, JSON-serializable, with each array or tensor
        replaced by {'#': i}.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        arrays.append(np.ascontiguousarray(value))
        return {'#': len(arrays) - 1}
    if isinstance(value, dict):
        return {k: _pack_arrays(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack_arrays(v, arrays) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Cannot serialize a value of type {type(value).__name__}.")


def _unpack_arrays(value: Any, arrays: list[torch.Tensor]) -> Any:
    """Invert '_pack_arrays', swapping each {'#': i} for 'arrays[i]'."""
    if isinstance(value, dict):
        if set(value) == {'#'}:
            index = value['#']
            if not isinstance(index, int) or not 0 <= index < len(arrays):
                raise ValueError(f"Serialized array index {index!r} is invalid.")
            return arrays[index]
        return {k: _unpack_arrays(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack_arrays(v, arrays) for v in value]
    return value


def _decode_arrays(table: Any, body: bytes, offset: int) -> list[torch.Tensor]:
    """Read the raw arrays that follow the JSON index.

    Parameters
    ----------
    table
        The index's [dtype, shape] entries, in storage order.
    body
        The decompressed payload body.
    offset
        Byte offset of the first array.

    Returns
    -------
    list[torch.Tensor]
        One CPU tensor per table entry.
    """
    if not isinstance(table, list):
        raise ValueError("Serialized array table must be a list.")

    arrays = []
    for entry in table:
        try:
            dtype_str, shape = entry
            dtype = np.dtype(dtype_str)
            shape = tuple(int(n) for n in shape)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid serialized array entry {entry!r}.") from e
        if dtype.kind not in _DTYPE_KINDS or not dtype.isnative:
            raise ValueError(f"Unsupported serialized array dtype '{dtype_str}'.")
        if any(n < 0 for n in shape):
            raise ValueError(f"Invalid serialized array shape {shape}.")

        count = int(np.prod(shape, dtype=np.int64))
        end = offset + count * dtype.itemsize
        if end > len(body):
            raise ValueError(
                f"Serialized state is truncated: an array needs bytes up to "
                f"{end}, but the payload has {len(body)}.",
            )
        values = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
        arrays.append(torch.from_numpy(values.reshape(shape).copy()))
        offset = end

    if offset != len(body):
        raise ValueError(
            f"Serialized state has {len(body) - offset} unexpected trailing bytes.",
        )
    return arrays


def encode_payload(magic: bytes, index: dict[str, Any], state: Any) -> bytes:
    """Encode a model state as a protocol payload.

    Parameters
    ----------
    magic
        8 bytes identifying the model that wrote the payload.
    index
        JSON-serializable entries stored next to the state (e.g., metadata).
        Must not use the keys 'state' or 'arrays'.
    state
        Nested dicts, lists, and tuples of arrays, tensors, and scalars.

    Returns
    -------
    bytes
        The payload (see the module docstring for the layout).
    """
    arrays = []
    index = {**index, 'state': _pack_arrays(state, arrays)}
    index['arrays'] = [[a.dtype.str, list(a.shape)] for a in arrays]
    index_bytes = json.dumps(index, separators=(',', ':')).encode('utf-8')

    header = _HEADER.pack(magic, _VERSION, len(index_bytes))
    body = b''.join([index_bytes, *(a.tobytes() for a in arrays)])
    return header + zlib.compress(body)


def decode_payload(magic: bytes, blob: bytes) -> dict[str, Any]:
    """Decode a payload made by 'encode_payload'.

    Checks the format, not the contents; the BMI validates those.

    Parameters
    ----------
    magic
        The 8 magic bytes the payload must start with.
    blob
        The payload.

    Returns
    -------
    dict[str, Any]
        The index entries plus 'state', with every array as a CPU tensor.
    """
    if len(blob) < _HEADER.size:
        raise ValueError(
            f"Serialized state is {len(blob)} bytes, shorter than its "
            f"{_HEADER.size}-byte header.",
        )
    found_magic, version, index_size = _HEADER.unpack_from(blob)
    if found_magic != magic:
        raise ValueError(
            f"Serialized state has magic {found_magic!r}, expected {magic!r}. "
            "It was not written by this model.",
        )
    if version != _VERSION:
        raise ValueError(
            f"Unsupported serialized state format version {version} "
            f"(expected {_VERSION}).",
        )

    inflater = zlib.decompressobj()
    try:
        body = inflater.decompress(blob[_HEADER.size :], _MAX_BYTES)
    except zlib.error as e:
        raise ValueError(f"Failed decompressing serialized state: {e}") from e
    if inflater.unconsumed_tail:
        raise ValueError(f"Serialized state exceeds {_MAX_BYTES} bytes decompressed.")
    if not inflater.eof or inflater.unused_data:
        raise ValueError("Serialized state is truncated or has trailing bytes.")

    if index_size > len(body):
        raise ValueError(
            f"Serialized state is truncated: its index needs {index_size} "
            f"bytes, but the body has {len(body)}.",
        )
    try:
        index = json.loads(body[:index_size].decode('utf-8'))
    except ValueError as e:
        raise ValueError(f"Failed decoding serialized state index: {e}") from e
    if not isinstance(index, dict) or not {'state', 'arrays'} <= set(index):
        raise ValueError("Serialized state index must hold 'state' and 'arrays'.")

    arrays = _decode_arrays(index.pop('arrays'), body, index_size)
    index['state'] = _unpack_arrays(index['state'], arrays)
    return index


class SerializableBmi:
    """Mixin implementing the ngen BMI Serialization Protocol for a BMI.

    Handles the reserved variables, the save sequence (create, size, state,
    free), and the restore sequence (size, state), restoring the previous state
    if a restore fails partway.

    A BMI using it calls '_init_serial_vars' in its constructor, routes the
    reserved names in its getters and 'set_value' to '_serial_vars' and
    '_set_serial_value', and implements:
    - '_runtime_model': the dPL model, which must provide 'get_runtime_state',
      'load_runtime_state', and 'state_dict'.
    - '_capture_state': encode the full state as bytes.
    - '_decode_state': decode and validate bytes without touching any state.
    - '_apply_state': write a decoded payload into the BMI and model.

    It reads the BMI attributes '_initialized', '_timestep', 'pt_dtype', and
    'verbose'.
    """

    def _init_serial_vars(self) -> None:
        """Create the arrays behind the reserved variables.

        They are numpy arrays because ngen's Python adapter reads every
        variable through 'get_value_ptr'.
        """
        self._serial_vars = {
            SERIAL_CREATE: np.zeros(1, dtype=np.int32),
            SERIAL_FREE: np.zeros(1, dtype=np.int32),
            SERIAL_SIZE: np.zeros(1, dtype=np.int64),
            SERIAL_STATE: np.zeros(0, dtype=np.uint8),
        }
        self._weights_sha256 = None

    def _runtime_model(self) -> torch.nn.Module:
        """Return the dPL model whose runtime state is serialized."""
        raise NotImplementedError("Derived classes must implement '_runtime_model'.")

    def _capture_state(self) -> bytes:
        """Encode the full BMI and model state as bytes."""
        raise NotImplementedError("Derived classes must implement '_capture_state'.")

    def _decode_state(self, blob: bytes) -> dict[str, Any]:
        """Decode and validate a payload without touching model state."""
        raise NotImplementedError("Derived classes must implement '_decode_state'.")

    def _apply_state(self, payload: dict[str, Any]) -> None:
        """Write a decoded payload into the BMI and model."""
        raise NotImplementedError("Derived classes must implement '_apply_state'.")

    def _capture_backup(self) -> bytes:
        """Capture everything needed to undo a failed restore."""
        return self._capture_state()

    def _serial_var_nbytes(self) -> int:
        """Size of the last capture, or of the payload announced for restore."""
        return int(self._serial_vars[SERIAL_SIZE][0])

    def _set_serial_value(self, name: str, src: Any) -> None:
        """Handle 'set_value' on a reserved serialization variable.

        Parameters
        ----------
        name
            Reserved variable name.
        src
            Value passed by the caller. Ignored for the two triggers.
        """
        if name == SERIAL_CREATE:
            self._create_serialization()
        elif name == SERIAL_FREE:
            self._free_serialization()
        elif name == SERIAL_SIZE:
            self._serial_vars[SERIAL_SIZE][0] = int(np.asarray(src).ravel()[0])
        else:
            if isinstance(src, (bytes, bytearray)):
                blob = np.frombuffer(src, dtype=np.uint8)
            else:
                blob = np.asarray(src, dtype=np.uint8).ravel()
            expected = int(self._serial_vars[SERIAL_SIZE][0])
            if expected and blob.size != expected:
                raise ValueError(
                    f"Serialized state has {blob.size} bytes, but "
                    f"'{SERIAL_SIZE}' announced {expected}.",
                )
            self._restore_state(blob.tobytes())

    def _create_serialization(self) -> None:
        """Capture the current state into the serialization buffer.

        Does not change the model's computed state.
        """
        blob = self._capture_state()
        self._serial_vars[SERIAL_STATE] = np.frombuffer(blob, dtype=np.uint8).copy()
        self._serial_vars[SERIAL_SIZE][0] = len(blob)

    def _free_serialization(self) -> None:
        """Release the serialization buffer. Safe to call at any time."""
        self._serial_vars[SERIAL_STATE] = np.zeros(0, dtype=np.uint8)
        self._serial_vars[SERIAL_SIZE][0] = 0

    def _require_runtime_state(self) -> None:
        """Check that the BMI and installed dmg support serialization."""
        if not self._initialized:
            raise RuntimeError(
                "BMI must be initialized before its state can be saved or restored.",
            )
        model = self._runtime_model()
        if not hasattr(model, 'get_runtime_state'):
            raise RuntimeError(
                f"State serialization requires a dmg whose {type(model).__name__} "
                "supports 'get_runtime_state'. Update dmg and hydrodl2.",
            )

    def _weights_digest(self) -> str:
        """SHA-256 of the loaded model weights, computed once."""
        if self._weights_sha256 is None:
            digest = hashlib.sha256()
            for key, value in self._runtime_model().state_dict().items():
                digest.update(key.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            self._weights_sha256 = digest.hexdigest()
        return self._weights_sha256

    def _check_metadata(self, meta: Any, expected: dict[str, Any]) -> None:
        """Check a payload's metadata against this BMI's.

        Parameters
        ----------
        meta
            Metadata decoded from the payload.
        expected
            This BMI's metadata. 'catchment_id' is only compared when both
            sides set it.
        """
        if not isinstance(meta, dict):
            raise ValueError("Serialized state metadata must be a mapping.")
        for key, value in expected.items():
            found = meta.get(key)
            if key == 'catchment_id' and (found is None or value is None):
                continue
            if found != value:
                raise ValueError(
                    f"Serialized state does not match this BMI: '{key}' is "
                    f"{found!r}, expected {value!r}.",
                )

    def _check_array(self, name: str, value: Any, shape: tuple) -> None:
        """Check that a decoded array has the expected shape and BMI dtype."""
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Serialized '{name}' is not an array.")
        if tuple(value.shape) != tuple(shape) or value.dtype != self.pt_dtype:
            raise ValueError(
                f"Serialized '{name}' has shape {tuple(value.shape)} and dtype "
                f"{value.dtype}, expected {tuple(shape)} and {self.pt_dtype}.",
            )

    def _restore_state(self, blob: bytes) -> None:
        """Restore BMI and model state from bytes made by '_capture_state'.

        The payload is fully decoded and validated before anything is written.
        If writing still fails, the previous state is put back and the error
        is raised, so a failed restore never leaves a half-restored model.

        Parameters
        ----------
        blob
            Serialized state.
        """
        self._require_runtime_state()
        payload = self._decode_state(blob)

        backup = self._decode_state(self._capture_backup())
        try:
            self._apply_state(payload)
        except Exception:
            self._apply_state(backup)
            raise

        if self.verbose:
            log.info(f"Restored state at step {self._timestep} ({len(blob)} bytes)")
