"""TurboCore: high-performance compression of float32 data.

    >>> import numpy as np, turbocore
    >>> x = np.linspace(0, 1, 1_000_000, dtype=np.float32)
    >>> stream = turbocore.compress(x)                      # lossless
    >>> np.array_equal(turbocore.decompress(stream), x)
    True
    >>> small = turbocore.compress(x, quantization="float16")   # lossy
    >>> turbocore.analyze_quantization(x, "float16")["error"]["psnr_db"]  # what that costs

All arrays are float32. Arrays of any shape are accepted and treated as flat;
the shape is not stored, so ``decompress`` returns a 1-D array (reshape it
yourself). Other dtypes are rejected rather than converted, because narrowing
float64 silently would make lossless compression lose data.

The native calls release the GIL and use all hardware threads by default
(``threads=0``).
"""

from ._core import (
    TurboCoreError,
    __version__,
    analyze_error as _analyze_error,
    analyze_quantization as _analyze_quantization,
    build_info,
    compress as _compress,
    decompress as _decompress,
    inspect as _inspect,
    is_backend_available,
    quantize_round_trip as _quantize_round_trip,
    runtime_support,
)

__all__ = [
    "TurboCoreError",
    "__version__",
    "analyze_error",
    "analyze_quantization",
    "build_info",
    "compress",
    "decompress",
    "inspect",
    "is_backend_available",
    "quantize_round_trip",
    "runtime_support",
]


def compress(
    data,
    *,
    quantization=None,
    predictor="delta",
    byte_shuffle=True,
    level=6,
    block_elements=65536,
    backend="auto",
    threads=0,
):
    """Compress a float32 array to ``bytes``.

    quantization    None for lossless; "float16" or "bfloat16" to round every
                    value to that format first (lossy). ``decompress`` then
                    returns exactly ``quantize_round_trip(data, quantization)``.
    predictor       "delta", "xor" or "none": decorrelation of neighbouring
                    values, applied to their bit patterns (always reversible).
    byte_shuffle    Group equal-significance bytes together before entropy
                    coding. Almost always a large win on float data.
    level           1 (fastest) to 9 (smallest; searches exhaustively).
    block_elements  Elements per independently decodable block.
    backend         "auto", "scalar", "avx2" or "cuda". Never changes the output.
    threads         Worker threads; 0 means one per hardware thread.

    Raises ValueError for arguments out of range and TypeError if ``data`` is
    not a float32 array.
    """
    return _compress(
        data,
        quantization=quantization,
        predictor=predictor,
        byte_shuffle=byte_shuffle,
        level=level,
        block_elements=block_elements,
        backend=backend,
        threads=threads,
    )


def decompress(data, *, backend="auto", threads=0):
    """Decompress a TurboCore stream (any bytes-like object) to a 1-D float32 array.

    Raises TurboCoreError if the stream is not a TurboCore stream, is damaged,
    or fails its checksums.
    """
    return _decompress(data, backend=backend, threads=threads)


def inspect(data):
    """Describe a stream from its header, without decompressing it.

    Returns a dict with element_count, block_count, block_elements, predictor,
    byte_shuffle, quantization (None if lossless) and compressed_bytes.
    """
    return _inspect(data)


def quantize_round_trip(data, format, *, backend="auto"):
    """Round a float32 array to "float16" or "bfloat16" and convert it back.

    The result is what a lossy stream of ``data`` decompresses to.
    """
    return _quantize_round_trip(data, format, backend=backend)


def analyze_error(reference, reconstructed, *, peak="max_abs"):
    """Error metrics of ``reconstructed`` against ``reference`` (both float32).

    Returns a dict with mean_squared_error, psnr_db, max_absolute_error,
    max_relative_error, max_ulp_distance (in float32 units in the last place)
    and counts of how NaN, infinity, overflow and underflow were handled.
    Non-finite pairs are counted, not averaged in.

    peak   The signal peak for PSNR: "max_abs" (largest |reference|), "range"
           (max - min of reference), or a positive number.
    """
    return _analyze_error(reference, reconstructed, peak=peak)


def analyze_quantization(data, format, *, peak="max_abs"):
    """What rounding ``data`` to "float16" or "bfloat16" would cost.

    Returns a dict with the ``analyze_error`` metrics under "error", plus the
    error in units in the last place of the target format
    (max_target_ulp_error is at most 0.5 for correct rounding) and its
    histogram.
    """
    return _analyze_quantization(data, format, peak=peak)
