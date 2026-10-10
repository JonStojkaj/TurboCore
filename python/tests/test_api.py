"""Tests of the Python API, run by CTest with PYTHONPATH pointing at the build tree.

Plain functions and asserts, run by the small runner at the bottom, so the
only requirement beyond the interpreter is NumPy.
"""

import math
import threading

import numpy as np

import turbocore


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def smooth(n):
    t = np.arange(n, dtype=np.float64) * 1e-3
    return (1000 + 40 * np.sin(t) + 3 * np.sin(17 * t)).astype(np.float32)


def special_values():
    patterns = [
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001,
        0xFFFFFFFF, 0x00000001, 0x007FFFFF, 0x7F7FFFFF, 0x3F800000, 0x477FF000,
    ]  # fmt: skip
    return np.array(patterns * 50, dtype=np.uint32).view(np.float32)


def raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return True
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"expected {exception.__name__}, got {type(other).__name__}: {other}") from other
    raise AssertionError(f"expected {exception.__name__}, nothing was raised")


def test_lossless_round_trip_is_bit_exact():
    rng = np.random.default_rng(1)
    for data in (
        smooth(200_000),
        rng.standard_normal(50_000, dtype=np.float32),
        rng.integers(0, 2**32, 30_000, dtype=np.uint32).view(np.float32),  # every bit pattern, NaNs included
        special_values(),
        np.zeros(0, dtype=np.float32),
        np.array([3.25], dtype=np.float32),
    ):
        stream = turbocore.compress(data)
        assert isinstance(stream, bytes)
        restored = turbocore.decompress(stream)
        assert restored.dtype == np.float32 and restored.ndim == 1
        assert np.array_equal(bits(restored), bits(data))


def test_options():
    data = smooth(30_000)
    for predictor in ("none", "delta", "xor"):
        for byte_shuffle in (True, False):
            for level in (1, 9):
                stream = turbocore.compress(
                    data, predictor=predictor, byte_shuffle=byte_shuffle, level=level, block_elements=4096
                )
                assert np.array_equal(bits(turbocore.decompress(stream)), bits(data))
                info = turbocore.inspect(stream)
                assert info["predictor"] == predictor and info["byte_shuffle"] == byte_shuffle
                assert info["element_count"] == data.size and info["block_elements"] == 4096
                assert info["quantization"] is None and info["compressed_bytes"] == len(stream)


def test_smooth_data_compresses():
    data = smooth(200_000)
    assert len(turbocore.compress(data)) < data.nbytes / 4


def test_threads_and_backends_do_not_change_the_stream():
    data = smooth(300_000)
    reference = turbocore.compress(data, threads=1, backend="scalar")
    for threads in (2, 8, 0):
        assert turbocore.compress(data, threads=threads) == reference
    assert turbocore.is_backend_available("scalar") and turbocore.is_backend_available("auto")
    for backend in ("avx2", "cuda"):
        if turbocore.is_backend_available(backend):
            assert turbocore.compress(data, backend=backend) == reference
            assert np.array_equal(bits(turbocore.decompress(reference, backend=backend)), bits(data))
        else:
            assert raises(turbocore.TurboCoreError, turbocore.compress, data, backend=backend)


def test_shapes_and_memory_layouts():
    data = smooth(6000).reshape(20, 300)
    flat = data.reshape(-1)
    # Any shape is accepted and flattened in C order.
    assert turbocore.compress(data) == turbocore.compress(flat)
    # A strided view is the values it shows, not the memory underneath.
    assert np.array_equal(turbocore.decompress(turbocore.compress(flat[::3])), flat[::3])
    # Fortran-ordered input is compressed in its logical (C) order.
    fortran = np.asfortranarray(data)
    assert np.array_equal(turbocore.decompress(turbocore.compress(fortran)), flat)


def test_other_dtypes_are_rejected_not_converted():
    for dtype in (np.float64, np.int32, np.float16):
        assert raises(TypeError, turbocore.compress, np.ones(10, dtype=dtype))
    assert raises(TypeError, turbocore.compress, [1.0, 2.0, 3.0])
    assert raises(TypeError, turbocore.analyze_quantization, np.ones(10), "float16")


def test_bytes_like_inputs():
    data = smooth(10_000)
    stream = turbocore.compress(data)
    for view in (bytearray(stream), memoryview(stream), np.frombuffer(stream, dtype=np.uint8)):
        assert np.array_equal(turbocore.decompress(view), data)
    assert raises(TypeError, turbocore.decompress, np.zeros(8, dtype=np.float32))


def test_lossy_mode_matches_quantize_round_trip():
    rng = np.random.default_rng(2)
    data = np.concatenate([smooth(50_000), rng.standard_normal(20_000, dtype=np.float32) * 1e3, special_values()])
    lossless_size = len(turbocore.compress(data))
    for fmt in ("float16", "bfloat16"):
        stream = turbocore.compress(data, quantization=fmt)
        assert turbocore.inspect(stream)["quantization"] == fmt
        assert len(stream) < lossless_size
        expected = turbocore.quantize_round_trip(data, fmt)
        assert np.array_equal(bits(turbocore.decompress(stream)), bits(expected))


def test_float16_agrees_with_numpy():
    # NumPy's float16 is IEEE binary16 with round-to-nearest-even: an
    # independent implementation of the same conversion.
    rng = np.random.default_rng(3)
    exponents = rng.integers(100, 146, 400_000, dtype=np.uint32)  # 2^-27 .. 2^18
    mantissas = rng.integers(0, 2**23, 400_000, dtype=np.uint32)
    signs = rng.integers(0, 2, 400_000, dtype=np.uint32)
    data = ((signs << 31) | (exponents << 23) | mantissas).view(np.float32)
    data = np.concatenate([data, np.array([0.0, -0.0, np.inf, -np.inf, 65504, 65520, 2.0**-24, 2.0**-25], np.float32)])

    with np.errstate(over="ignore", under="ignore"):
        numpy_result = data.astype(np.float16).astype(np.float32)
    ours = turbocore.quantize_round_trip(data, "float16")
    assert np.array_equal(bits(ours), bits(numpy_result))

    # NaN: same class and sign; payload conventions are checked on the C++ side.
    nan = np.array([np.nan, -np.nan], dtype=np.float32)
    ours_nan = turbocore.quantize_round_trip(nan, "float16")
    assert np.all(np.isnan(ours_nan)) and np.array_equal(np.signbit(ours_nan), np.signbit(nan))


def test_bfloat16_agrees_with_integer_rounding():
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 2**32, 300_000, dtype=np.uint64)
    raw = raw[(raw & 0x7FFFFFFF) <= 0x7F800000]  # no NaNs
    kept = raw >> 16
    remainder = raw & 0xFFFF
    kept = kept + ((remainder > 0x8000) | ((remainder == 0x8000) & (kept & 1 == 1)))
    expected = (kept << 16).astype(np.uint32)
    ours = turbocore.quantize_round_trip(raw.astype(np.uint32).view(np.float32), "bfloat16")
    assert np.array_equal(bits(ours), expected)


def test_analyze_error_matches_numpy():
    rng = np.random.default_rng(5)
    reference = rng.uniform(-50, 50, 100_000).astype(np.float32)
    reconstructed = (reference + rng.normal(0, 1e-3, reference.size)).astype(np.float32)
    report = turbocore.analyze_error(reference, reconstructed)

    difference = reconstructed.astype(np.float64) - reference.astype(np.float64)
    mse = math.fsum(difference * difference) / difference.size  # exactly rounded sum
    assert math.isclose(report["mean_squared_error"], mse, rel_tol=1e-13)
    assert math.isclose(report["max_absolute_error"], np.max(np.abs(difference)), rel_tol=0)
    peak = float(np.max(np.abs(reference)))
    assert report["peak"] == peak
    assert math.isclose(report["psnr_db"], 10 * math.log10(peak * peak / mse), rel_tol=1e-12)
    assert report["compared_count"] == reference.size and report["nan_introduced_count"] == 0

    by_range = turbocore.analyze_error(reference, reconstructed, peak="range")
    assert by_range["peak"] == float(reference.max()) - float(reference.min())
    assert turbocore.analyze_error(reference, reconstructed, peak=255.0)["peak"] == 255.0
    assert turbocore.analyze_error(reference, reference)["psnr_db"] == math.inf
    assert raises(ValueError, turbocore.analyze_error, reference, reconstructed, peak="loud")
    assert raises(ValueError, turbocore.analyze_error, reference, reconstructed, peak=-1.0)
    assert raises(ValueError, turbocore.analyze_error, reference, reconstructed[:-1])


def test_analyze_quantization():
    rng = np.random.default_rng(6)
    data = rng.uniform(-1000, 1000, 200_000).astype(np.float32)
    f16 = turbocore.analyze_quantization(data, "float16")
    bf16 = turbocore.analyze_quantization(data, "bfloat16")
    for report in (f16, bf16):
        assert report["above_half_ulp_count"] == 0
        assert report["max_target_ulp_error"] <= 0.5
        assert sum(report["target_ulp_histogram"]) == report["error"]["compared_count"] == data.size
    assert f16["format"] == "float16" and bf16["format"] == "bfloat16"
    assert f16["error"]["psnr_db"] > bf16["error"]["psnr_db"] + 15  # three more mantissa bits

    out_of_range = np.array([1e6, 1e-10, 1.0], dtype=np.float32)
    report = turbocore.analyze_quantization(out_of_range, "float16")["error"]
    assert report["overflow_to_infinity_count"] == 1 and report["underflow_to_zero_count"] == 1

    # The prediction is exactly what the lossy stream delivers.
    decoded = turbocore.decompress(turbocore.compress(data, quantization="float16"))
    assert turbocore.analyze_error(data, decoded)["mean_squared_error"] == f16["error"]["mean_squared_error"]


def test_errors():
    data = smooth(5000)
    stream = bytearray(turbocore.compress(data, block_elements=1000))
    assert raises(turbocore.TurboCoreError, turbocore.decompress, stream[:-1])
    assert raises(turbocore.TurboCoreError, turbocore.decompress, b"not a turbocore stream at all, really")
    assert raises(turbocore.TurboCoreError, turbocore.inspect, b"")
    damaged = bytearray(stream)
    damaged[32 + 4] ^= 0xFF  # first block's checksum
    assert raises(turbocore.TurboCoreError, turbocore.decompress, damaged)
    assert issubclass(turbocore.TurboCoreError, RuntimeError)

    for bad in (
        {"level": 0},
        {"level": 10},
        {"block_elements": 0},
        {"predictor": "magic"},
        {"quantization": "float8"},
        {"backend": "quantum"},
        {"threads": -1},
    ):
        assert raises(ValueError, turbocore.compress, data, **bad)


def test_gil_is_released():
    # Compression running in several Python threads at once must work and
    # produce the same stream; with the GIL held it would merely be serial,
    # so this is a correctness smoke test of the release, not a timing test.
    data = smooth(400_000)
    reference = turbocore.compress(data, threads=1)
    results = [None] * 4

    def work(i):
        results[i] = turbocore.compress(data, threads=2)

    workers = [threading.Thread(target=work, args=(i,)) for i in range(len(results))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert all(r == reference for r in results)


def test_metadata():
    info = turbocore.build_info()
    assert info["version"] == turbocore.__version__
    assert set(info) == {"version", "avx2_compiled", "cuda_compiled"}
    support = turbocore.runtime_support()
    assert support["best_backend"] in ("scalar", "avx2")
    assert support["cuda_device_count"] >= 0


if __name__ == "__main__":
    tests = [(name, function) for name, function in sorted(globals().items()) if name.startswith("test_")]
    for name, function in tests:
        function()
        print(f"ok  {name}")
    print(f"{len(tests)} Python API tests passed (turbocore {turbocore.__version__}, numpy {np.__version__})")
