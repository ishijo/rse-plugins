#!/usr/bin/env python3
"""
Zarr Compression Codec Comparison Benchmark

Benchmarks all major compression codecs and Blosc shuffle modes on
realistic scientific data (correlated structure, not pure random).

Tests:
- Blosc with all cname options and all shuffle modes
- Standalone codecs (Zstd, LZ4, Gzip, Zlib, BZ2)
- Delta filter + Blosc+Zstd
- No compression baseline

Reports: compression ratio, compression time, decompression time, compressed size

Dependencies: zarr, numpy, numcodecs (standard Zarr dependencies)

Usage:
    python codec-comparison.py
"""

import time
import numpy as np
import zarr
from numcodecs import (
    Blosc, Zstd, LZ4, GZip, Zlib, BZ2, Delta
)
import tempfile
import shutil


def create_realistic_data(shape=(1000, 1000), dtype='f8'):
    """
    Create realistic scientific data with correlation structure.

    Uses cumulative sum to create spatial correlation, mimicking
    real scientific data (temperature fields, etc.) rather than
    pure random noise which compresses poorly.
    """
    # Start with random noise
    noise = np.random.randn(*shape)

    # Add spatial correlation via cumulative sum in both dimensions
    # This creates realistic gradients like in temperature/pressure fields
    correlated = np.cumsum(np.cumsum(noise, axis=0), axis=1)

    # Normalize to reasonable range
    correlated = (correlated - correlated.mean()) / correlated.std()
    correlated = correlated * 10 + 15  # Mean ~15, std ~10 (like temperature)

    return correlated.astype(dtype)


def benchmark_codec(name, compressor, data, filters=None):
    """
    Benchmark a single codec configuration.

    Parameters
    ----------
    name : str
        Codec name for display
    compressor : Codec or None
        Compression codec to test
    data : ndarray
        Test data
    filters : list or None
        Pre-compression filters

    Returns
    -------
    dict
        Benchmark results
    """
    # Create temporary directory for test
    tmpdir = tempfile.mkdtemp()

    try:
        # Compression (write)
        start_compress = time.time()

        arr = zarr.open_array(
            f"{tmpdir}/test.zarr",
            mode="w",
            shape=data.shape,
            chunks=data.shape,  # Single chunk for consistent comparison
            dtype=data.dtype,
            compressor=compressor,
            filters=filters
        )
        arr[:] = data

        compress_time = time.time() - start_compress

        # Get compressed size
        compressed_size = arr.nbytes_stored

        # Decompression (read)
        start_decompress = time.time()
        _ = arr[:]
        decompress_time = time.time() - start_decompress

        # Calculate metrics
        uncompressed_size = arr.nbytes
        ratio = uncompressed_size / compressed_size if compressed_size > 0 else 0

        return {
            'name': name,
            'ratio': ratio,
            'compress_time': compress_time,
            'decompress_time': decompress_time,
            'compressed_kb': compressed_size / 1024,
            'uncompressed_kb': uncompressed_size / 1024
        }

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)


def main():
    """Run comprehensive codec comparison benchmark."""

    print("=" * 80)
    print("ZARR COMPRESSION CODEC COMPARISON")
    print("=" * 80)

    # Create test data
    print("\nGenerating test data (1000 x 1000 float64 with spatial correlation)...")
    data = create_realistic_data(shape=(1000, 1000), dtype='f8')

    print(f"Uncompressed size: {data.nbytes / 1024:.0f} KB ({data.nbytes / 1024**2:.2f} MB)")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Mean: {data.mean():.2f}, Std: {data.std():.2f}")

    # Test configurations
    configs = []

    # Blosc variants with all shuffle modes
    print("\nTesting Blosc variants...")

    # Blosc+LZ4 (all shuffle modes)
    configs.append(("Blosc+LZ4 (NOSHUFFLE)", Blosc(cname='lz4', clevel=5, shuffle=Blosc.NOSHUFFLE), None))
    configs.append(("Blosc+LZ4 (SHUFFLE)", Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE), None))
    configs.append(("Blosc+LZ4 (BITSHUFFLE)", Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE), None))

    # Blosc+Zstd (all shuffle modes) - DEFAULT for v2
    configs.append(("Blosc+Zstd (NOSHUFFLE)", Blosc(cname='zstd', clevel=3, shuffle=Blosc.NOSHUFFLE), None))
    configs.append(("Blosc+Zstd (SHUFFLE)", Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE), None))
    configs.append(("Blosc+Zstd (BITSHUFFLE)", Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE), None))

    # Standalone codecs
    print("Testing standalone codecs...")
    configs.append(("Zstd (level 3)", Zstd(level=3), None))
    configs.append(("LZ4 (acceleration 1)", LZ4(acceleration=1), None))
    configs.append(("Gzip (level 6)", GZip(level=6), None))
    configs.append(("Zlib (level 6)", Zlib(level=6), None))
    configs.append(("BZ2 (level 9)", BZ2(level=9), None))

    # Filter + compression
    print("Testing filters + compression...")
    configs.append(("Delta + Blosc+Zstd", Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE), [Delta(dtype='f8')]))

    # No compression baseline
    configs.append(("No compression", None, None))

    # Run benchmarks
    results = []
    for name, compressor, filters in configs:
        print(f"  Benchmarking: {name:30s}", end=" ... ")
        result = benchmark_codec(name, compressor, data, filters)
        results.append(result)
        print(f"{result['ratio']:.2f}x")

    # Display results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Table header
    header = f"{'Codec':35s} | {'Ratio':>6s} | {'Comp (s)':>9s} | {'Decomp (s)':>11s} | {'Size (KB)':>10s}"
    print(header)
    print("-" * 80)

    # Sort by compression ratio (descending)
    results.sort(key=lambda x: x['ratio'], reverse=True)

    # Print rows
    for r in results:
        row = f"{r['name']:35s} | {r['ratio']:6.2f} | {r['compress_time']:9.4f} | {r['decompress_time']:11.4f} | {r['compressed_kb']:10.0f}"
        print(row)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_ratio = max(results, key=lambda x: x['ratio'])
    fastest_compress = min(results, key=lambda x: x['compress_time'])
    fastest_decompress = min(results, key=lambda x: x['decompress_time'])

    print(f"\nBest compression ratio:   {best_ratio['name']:35s} ({best_ratio['ratio']:.2f}x)")
    print(f"Fastest compression:      {fastest_compress['name']:35s} ({fastest_compress['compress_time']:.4f}s)")
    print(f"Fastest decompression:    {fastest_decompress['name']:35s} ({fastest_decompress['decompress_time']:.4f}s)")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. General purpose (balanced):     Blosc+Zstd (BITSHUFFLE) - excellent ratio + fast")
    print("2. Speed priority:                 Blosc+LZ4 (SHUFFLE) - fastest with good ratio")
    print("3. Maximum compression:            BZ2 or Delta + Blosc+Zstd")
    print("4. Floating-point scientific data: Use BITSHUFFLE for best results")
    print("5. Integer or correlated data:     Add Delta filter for better compression")

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("\n- BITSHUFFLE is particularly effective for floating-point data")
    print("- Delta filter works well with spatially/temporally correlated data")
    print("- Larger chunks generally compress better (more patterns to exploit)")
    print("- Test on YOUR actual data - results vary by data characteristics")
    print("- Single-threaded execution (safe for multi-process use)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Ensure Blosc threading is disabled for safety
    import blosc
    blosc.use_threads = False
    print("Blosc threading disabled for single-process safety")

    main()
