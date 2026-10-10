"""Cross-checks TurboCore's DEFLATE codec against zlib in both directions.

Usage: zlib_interop.py <path to turbocore_deflate_cli>

zlib is an independent implementation of RFC 1951, so agreement here means
the codec is correct DEFLATE and not merely consistent with itself.
"""

import random
import struct
import subprocess
import sys
import zlib

CLI = sys.argv[1]
RAW = -15  # zlib wbits for a raw DEFLATE stream with a 32 KiB window


def run(mode, argument, data, *extra):
    result = subprocess.run([CLI, mode, str(argument), *map(str, extra)], input=data, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(f"{mode} {argument} failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


def fibonacci_bytes(rng):
    """Symbol frequencies that force Huffman codes past the 15-bit limit."""
    counts = [1, 1]
    while len(counts) < 26:
        counts.append(counts[-1] + counts[-2])
    data = bytearray()
    for symbol, count in enumerate(counts):
        data += bytes([symbol]) * count
    rng.shuffle(data)
    return bytes(data)


def shuffled_float_deltas():
    """What the float pipeline actually feeds the codec: byte planes of deltas."""
    import math

    words = [struct.unpack("<I", struct.pack("<f", 1000 + 40 * math.sin(i * 1e-3)))[0] for i in range(30000)]
    deltas = [words[0]] + [(b - a) & 0xFFFFFFFF for a, b in zip(words, words[1:])]
    packed = struct.pack(f"<{len(deltas)}I", *deltas)
    return b"".join(packed[plane::4] for plane in range(4))


def cases():
    rng = random.Random(1234)
    text = b"".join(b"the quick brown fox jumps over the lazy dog %d\n" % (i % 17) for i in range(3000))
    far = rng.randbytes(40000)
    return {
        "empty": b"",
        "one_byte": b"x",
        "zeros": bytes(100000),
        "run_258": b"\xab" * 258,
        "text": text,
        "random": rng.randbytes(70000),
        "far_repeat": far + far,
        "skewed": bytes(rng.choices(range(256), weights=[1 / (i + 1) ** 2 for i in range(256)], k=80000)),
        "fibonacci": fibonacci_bytes(rng),
        "float_deltas": shuffled_float_deltas(),
        "large_mixed": text + rng.randbytes(50000) + bytes(300000) + text,
    }


def zlib_streams(data):
    """Raw DEFLATE encodings of `data` covering every block type."""
    for level in (0, 1, 6, 9):  # level 0 emits stored blocks
        c = zlib.compressobj(level, zlib.DEFLATED, RAW)
        yield f"zlib level {level}", c.compress(data) + c.flush()

    c = zlib.compressobj(6, zlib.DEFLATED, RAW, 8, zlib.Z_FIXED)
    yield "zlib fixed codes", c.compress(data) + c.flush()

    # Flushes split the data into many blocks, with empty stored blocks and
    # byte alignment in between.
    c = zlib.compressobj(6, zlib.DEFLATED, RAW)
    stream = b""
    for start in range(0, len(data), 7001):
        stream += c.compress(data[start : start + 7001]) + c.flush(zlib.Z_FULL_FLUSH)
    yield "zlib multi-block", stream + c.flush()


def main():
    checks = 0
    for name, data in cases().items():
        # TurboCore encoder -> zlib decoder.
        # Segment hints change how the stream is split into blocks (and which
        # block types appear), so sweep them: none, short, and a quarter of
        # the input, as the float pipeline would pass for 4 byte planes.
        for level in (1, 6, 9):
            for hint in (0, 4096, max(1, len(data) // 4)):
                stream = run("deflate", level, data, hint)
                d = zlib.decompressobj(RAW)
                decoded = d.decompress(stream) + d.flush()
                label = f"{name}: level {level}, hint {hint}"
                assert decoded == data, f"{label}: zlib decoded our stream to different data"
                assert d.eof, f"{label}: zlib found no final block in our stream"
                assert not d.unused_data, f"{label}: our stream has trailing bytes"
                assert len(stream) <= len(data) + 10 * (len(data) // 4096 + 1), f"{label}: larger than stored"
                checks += 1

        # zlib encoder -> TurboCore decoder.
        for label, stream in zlib_streams(data):
            decoded = run("inflate", len(data), stream)
            assert decoded == data, f"{name}: we decoded a {label} stream to different data"
            checks += 1

    print(f"zlib interop: {checks} checks passed")


if __name__ == "__main__":
    main()
