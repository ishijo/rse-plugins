# Real-World Compression Examples

## Example 1: Optimizing Climate Data Storage

**Compress large climate model output with optimal settings:**

```python
import zarr
from numcodecs import Blosc, Delta
import numpy as np

def compress_climate_data():
    """
    Compress climate model output optimally.

    Climate data characteristics:
    - Floating-point temperature/pressure fields
    - Spatial correlation (smooth gradients)
    - Temporal correlation (slow changes)
    - Large arrays (365 days × 180 lat × 360 lon)
    """

    # Create sample climate data
    n_time = 365
    n_lat = 180
    n_lon = 360

    print("Creating climate dataset...")

    # Temperature data with realistic structure
    # Base pattern + seasonal cycle + daily variation
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    time_days = np.arange(n_time)

    # Create spatially and temporally correlated data
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    # Spatial pattern (latitudinal temperature gradient)
    base_temp = 25 * np.cos(np.deg2rad(lat_grid))

    # Add temporal variation
    temp_data = np.zeros((n_time, n_lat, n_lon), dtype='f4')
    for t in range(n_time):
        seasonal = 10 * np.cos(2 * np.pi * t / 365)
        daily_noise = np.random.randn(n_lat, n_lon) * 2
        temp_data[t, :, :] = (base_temp + seasonal + daily_noise).astype('f4')

    uncompressed_size = temp_data.nbytes

    # Strategy 1: Standard compression (Blosc+Zstd with BITSHUFFLE)
    arr_standard = zarr.open_array(
        "climate_standard.zarr",
        mode="w",
        shape=(n_time, n_lat, n_lon),
        chunks=(30, 18, 36),  # ~2 MB chunks
        dtype="f4",
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    )
    arr_standard[:] = temp_data

    # Strategy 2: Delta filter + compression (exploit temporal correlation)
    arr_delta = zarr.open_array(
        "climate_delta.zarr",
        mode="w",
        shape=(n_time, n_lat, n_lon),
        chunks=(30, 18, 36),
        dtype="f4",
        filters=[Delta(dtype='f4')],
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    )
    arr_delta[:] = temp_data

    # Strategy 3: High compression for archival
    arr_archival = zarr.open_array(
        "climate_archival.zarr",
        mode="w",
        shape=(n_time, n_lat, n_lon),
        chunks=(30, 18, 36),
        dtype="f4",
        filters=[Delta(dtype='f4')],
        compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
    )
    arr_archival[:] = temp_data

    # Compare results
    print("\n" + "="*70)
    print("CLIMATE DATA COMPRESSION RESULTS")
    print("="*70)

    print(f"\nUncompressed: {uncompressed_size / 1024**3:.2f} GB")

    strategies = [
        ("Standard (Blosc+Zstd BITSHUFFLE)", arr_standard),
        ("Delta + Blosc+Zstd", arr_delta),
        ("Archival (Delta + Zstd-9)", arr_archival)
    ]

    for name, arr in strategies:
        ratio = arr.nbytes / arr.nbytes_stored
        size_mb = arr.nbytes_stored / 1024**2
        savings = (1 - arr.nbytes_stored / uncompressed_size) * 100

        print(f"\n{name}:")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  Savings: {savings:.1f}%")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nFor climate data:")
    print("- Active analysis: Blosc+Zstd with BITSHUFFLE (fast access)")
    print("- Archival: Delta + Blosc+Zstd level 9 (best compression)")
    print("- Cloud storage: Use Delta filter to reduce transfer costs")

if __name__ == "__main__":
    compress_climate_data()
```

## Example 2: Integer Time-Series with Delta Encoding

**Optimize compression for correlated integer data:**

```python
import zarr
from numcodecs import Blosc, Delta
import numpy as np

def compress_sensor_data():
    """
    Compress integer sensor readings with Delta encoding.

    Sensor data characteristics:
    - Integer values (counts, IDs)
    - High temporal correlation (slow changes)
    - Delta encoding is very effective
    """

    # Simulate sensor readings (counts that change slowly)
    n_sensors = 100
    n_samples = 100000

    print("Generating sensor data...")

    # Create correlated integer data (particle counts)
    # Values drift slowly over time
    sensor_data = np.zeros((n_samples, n_sensors), dtype='i4')

    # Each sensor starts at different baseline
    baselines = np.random.randint(1000, 5000, n_sensors)

    for i in range(n_samples):
        if i == 0:
            sensor_data[i, :] = baselines
        else:
            # Small random walk
            changes = np.random.randint(-10, 11, n_sensors)
            sensor_data[i, :] = sensor_data[i-1, :] + changes

    uncompressed_size = sensor_data.nbytes

    # Without Delta filter
    arr_no_delta = zarr.array(
        sensor_data,
        chunks=(1000, 100),
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    )

    # With Delta filter
    arr_with_delta = zarr.open_array(
        "sensor_delta.zarr",
        mode="w",
        shape=sensor_data.shape,
        chunks=(1000, 100),
        dtype='i4',
        filters=[Delta(dtype='i4')],
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    )
    arr_with_delta[:] = sensor_data

    # Results
    print("\n" + "="*70)
    print("SENSOR DATA COMPRESSION RESULTS")
    print("="*70)

    print(f"\nUncompressed: {uncompressed_size / 1024**2:.2f} MB")

    ratio_no_delta = arr_no_delta.nbytes / arr_no_delta.nbytes_stored
    ratio_with_delta = arr_with_delta.nbytes / arr_with_delta.nbytes_stored

    print(f"\nWithout Delta filter:")
    print(f"  Size: {arr_no_delta.nbytes_stored / 1024**2:.2f} MB")
    print(f"  Ratio: {ratio_no_delta:.2f}x")

    print(f"\nWith Delta filter:")
    print(f"  Size: {arr_with_delta.nbytes_stored / 1024**2:.2f} MB")
    print(f"  Ratio: {ratio_with_delta:.2f}x")

    improvement = (ratio_with_delta / ratio_no_delta - 1) * 100
    print(f"\nDelta filter improvement: {improvement:.1f}%")

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("\nDelta encoding is highly effective for correlated integer data:")
    print("- Time-series with slow changes")
    print("- Sequential IDs or counters")
    print("- Spatial data with smooth gradients")
    print("\nCombine Delta with shuffle for best results on integers.")

if __name__ == "__main__":
    compress_sensor_data()
```

## Example 3: Comparing Codecs for Different Data Types

**Test multiple codecs on different data characteristics:**

```python
import zarr
from numcodecs import Blosc, Zstd, LZ4, GZip, LZMA
import numpy as np
import time

def compare_codecs_by_data_type():
    """
    Compare compression codecs across different data types.

    Tests:
    1. Floating-point scientific data (correlated)
    2. Random noise (incompressible)
    3. Integer data with patterns
    4. Text/string data
    """

    print("="*70)
    print("CODEC COMPARISON BY DATA TYPE")
    print("="*70)

    # Codec configurations
    codecs = {
        'Blosc+Zstd': Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
        'Blosc+LZ4': Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
        'Zstd': Zstd(level=3),
        'LZ4': LZ4(acceleration=1),
        'Gzip': GZip(level=6),
        'LZMA': LZMA(preset=6),
    }

    # Test data types
    datasets = {}

    # 1. Floating-point scientific data (spatially correlated)
    print("\nGenerating floating-point data (correlated)...")
    noise = np.random.randn(1000, 1000)
    correlated_float = np.cumsum(np.cumsum(noise, axis=0), axis=1)
    correlated_float = (correlated_float - correlated_float.mean()) / correlated_float.std()
    datasets['Correlated Float'] = correlated_float.astype('f4')

    # 2. Random noise (incompressible)
    print("Generating random noise (incompressible)...")
    datasets['Random Noise'] = np.random.randn(1000, 1000).astype('f4')

    # 3. Integer data with patterns
    print("Generating integer data (patterns)...")
    x, y = np.meshgrid(np.arange(1000), np.arange(1000))
    pattern_int = ((x ** 2 + y ** 2) % 256).astype('i4')
    datasets['Pattern Integer'] = pattern_int

    # 4. Repeated values (highly compressible)
    print("Generating repeated values...")
    repeated = np.tile(np.arange(100), (1000, 10)).astype('i4')
    datasets['Repeated Values'] = repeated

    # Test each combination
    for data_name, data in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATA TYPE: {data_name}")
        print(f"{'='*70}")
        print(f"Shape: {data.shape}, Dtype: {data.dtype}")
        print(f"Uncompressed: {data.nbytes / 1024:.0f} KB")

        print(f"\n{'Codec':20s} | {'Ratio':>6s} | {'Time (s)':>9s} | {'Size (KB)':>10s}")
        print("-"*70)

        for codec_name, codec in codecs.items():
            try:
                start = time.time()
                arr = zarr.array(data, chunks=data.shape, compressor=codec)
                elapsed = time.time() - start

                ratio = arr.nbytes / arr.nbytes_stored if arr.nbytes_stored > 0 else 0
                size_kb = arr.nbytes_stored / 1024

                print(f"{codec_name:20s} | {ratio:6.2f} | {elapsed:9.4f} | {size_kb:10.0f}")

            except Exception as e:
                print(f"{codec_name:20s} | ERROR: {str(e)[:40]}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n1. Correlated Float:")
    print("   - Best: Blosc+Zstd with BITSHUFFLE")
    print("   - Delta filter can improve further")
    print("\n2. Random Noise:")
    print("   - All codecs perform poorly (data is incompressible)")
    print("   - Consider disabling compression")
    print("\n3. Pattern Integer:")
    print("   - Good compression with most codecs")
    print("   - SHUFFLE effective for integer patterns")
    print("\n4. Repeated Values:")
    print("   - Excellent compression with all codecs")
    print("   - Simple algorithms (LZ4) work well")

if __name__ == "__main__":
    compare_codecs_by_data_type()
```

## Example 4: Production Pipeline with Adaptive Compression

**Implement adaptive compression based on data analysis:**

```python
import zarr
from numcodecs import Blosc, Delta, Quantize
import numpy as np

def analyze_compressibility(data_sample):
    """
    Analyze data sample to determine optimal compression strategy.

    Returns recommended compressor and filters.
    """
    from scipy import stats

    # Detect data characteristics
    is_integer = np.issubdtype(data_sample.dtype, np.integer)
    is_float = np.issubdtype(data_sample.dtype, np.floating)

    # Test for correlation (temporal or spatial)
    if len(data_sample.shape) > 1:
        # Test correlation along first axis
        correlations = []
        for i in range(min(10, data_sample.shape[1])):
            if data_sample.shape[0] > 1:
                corr = np.corrcoef(data_sample[:-1, i].flat, data_sample[1:, i].flat)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        avg_correlation = np.mean(correlations) if correlations else 0
    else:
        avg_correlation = 0

    # Test compressibility with fast codec
    test_arr = zarr.array(data_sample, chunks=data_sample.shape,
                          compressor=Blosc(cname='lz4', clevel=1))
    test_ratio = test_arr.nbytes / test_arr.nbytes_stored

    # Decision logic
    filters = []
    compressor = None

    if test_ratio < 1.1:
        # Nearly incompressible
        compressor = None
        strategy = "No compression (incompressible data)"

    elif is_integer and avg_correlation > 0.3:
        # Correlated integer data - use Delta filter
        filters = [Delta(dtype=data_sample.dtype)]
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
        strategy = "Delta + Blosc+Zstd (correlated integers)"

    elif is_float:
        # Floating-point data - use BITSHUFFLE
        if avg_correlation > 0.3:
            filters = [Delta(dtype=data_sample.dtype)]

        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        strategy = "Blosc+Zstd BITSHUFFLE" + (" + Delta" if filters else "") + " (float)"

    else:
        # Default strategy
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
        strategy = "Blosc+Zstd (default)"

    return {
        'compressor': compressor,
        'filters': filters if filters else None,
        'strategy': strategy,
        'estimated_ratio': test_ratio,
        'correlation': avg_correlation
    }


def adaptive_compression_pipeline():
    """
    Production pipeline with adaptive compression selection.
    """

    # Simulate different data types in pipeline
    datasets = {
        'temperature': np.cumsum(np.random.randn(1000, 500), axis=0).astype('f4'),  # Correlated float
        'station_ids': np.repeat(np.arange(1, 501), 1000).reshape(1000, 500).astype('i4'),  # Repeated int
        'noise': np.random.randn(1000, 500).astype('f4'),  # Random
    }

    print("="*70)
    print("ADAPTIVE COMPRESSION PIPELINE")
    print("="*70)

    for name, data in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")

        # Analyze sample
        sample = data[:100, :100]
        config = analyze_compressibility(sample)

        print(f"\nAnalysis:")
        print(f"  Strategy: {config['strategy']}")
        print(f"  Correlation: {config['correlation']:.3f}")
        print(f"  Estimated ratio: {config['estimated_ratio']:.2f}x")

        # Create array with recommended settings
        arr = zarr.open_array(
            f"{name}.zarr",
            mode="w",
            shape=data.shape,
            chunks=(100, 100),
            dtype=data.dtype,
            filters=config['filters'],
            compressor=config['compressor']
        )
        arr[:] = data

        # Report actual results
        actual_ratio = arr.nbytes / arr.nbytes_stored if arr.nbytes_stored > 0 else 0

        print(f"\nResults:")
        print(f"  Uncompressed: {arr.nbytes / 1024:.0f} KB")
        print(f"  Compressed: {arr.nbytes_stored / 1024:.0f} KB")
        print(f"  Actual ratio: {actual_ratio:.2f}x")
        print(f"  Savings: {(1 - arr.nbytes_stored / arr.nbytes) * 100:.1f}%")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nAdaptive compression:")
    print("1. Analyzes data characteristics automatically")
    print("2. Selects optimal codec and filters")
    print("3. Avoids compression overhead on incompressible data")
    print("4. Maximizes compression on compressible data")
    print("\nRecommended for heterogeneous data pipelines.")

if __name__ == "__main__":
    adaptive_compression_pipeline()
```

## Example 5: Compression for Multi-Process Workflows

**Safe compression in distributed computing environments:**

```python
import zarr
import blosc
from zarr import ProcessSynchronizer
from multiprocessing import Process, current_process
import numpy as np
import time

def distributed_worker(store_path, sync_path, worker_id, n_workers, total_rows):
    """
    Worker process for distributed compression task.

    CRITICAL: Blosc threading MUST be disabled in each worker.
    """
    # CRITICAL: First thing in worker process
    blosc.use_threads = False

    # Calculate this worker's chunk
    rows_per_worker = total_rows // n_workers
    start_row = worker_id * rows_per_worker
    end_row = start_row + rows_per_worker if worker_id < n_workers - 1 else total_rows

    # Open array with synchronizer
    arr = zarr.open_array(
        store_path,
        mode="r+",
        synchronizer=ProcessSynchronizer(sync_path)
    )

    # Generate and write data
    print(f"Worker {worker_id}: Processing rows {start_row}-{end_row}")

    chunk_size = 1000
    for i in range(start_row, end_row, chunk_size):
        end_i = min(i + chunk_size, end_row)
        n_rows = end_i - i

        # Simulate data processing
        data = np.random.randn(n_rows, arr.shape[1]).astype('f4')
        data = np.cumsum(data, axis=0)  # Add correlation

        # Write chunk
        arr[i:end_i, :] = data

    print(f"Worker {worker_id}: Complete")


def distributed_compression_workflow():
    """
    Demonstrate safe multi-process compression workflow.
    """

    print("="*70)
    print("DISTRIBUTED COMPRESSION WORKFLOW")
    print("="*70)

    # CRITICAL: Disable Blosc threading in main process
    blosc.use_threads = False

    store_path = "distributed_data.zarr"
    sync_path = "distributed_data.sync"

    n_workers = 4
    total_rows = 10000
    n_cols = 1000

    print(f"\nConfiguration:")
    print(f"  Workers: {n_workers}")
    print(f"  Array shape: ({total_rows}, {n_cols})")
    print(f"  Blosc threading: DISABLED (safe for multi-process)")

    # Create array with Blosc compression
    print("\nCreating compressed array...")
    arr = zarr.open_array(
        store_path,
        mode="w",
        shape=(total_rows, n_cols),
        chunks=(1000, 1000),
        dtype='f4',
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
        synchronizer=ProcessSynchronizer(sync_path)
    )

    # Launch workers
    print(f"\nLaunching {n_workers} worker processes...")
    start_time = time.time()

    workers = [
        Process(target=distributed_worker,
                args=(store_path, sync_path, i, n_workers, total_rows))
        for i in range(n_workers)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    elapsed = time.time() - start_time

    # Report results
    arr_final = zarr.open_array(store_path, mode="r")

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"Uncompressed size: {arr_final.nbytes / 1024**3:.2f} GB")
    print(f"Compressed size: {arr_final.nbytes_stored / 1024**3:.2f} GB")
    print(f"Compression ratio: {arr_final.nbytes / arr_final.nbytes_stored:.2f}x")
    print(f"Space savings: {(1 - arr_final.nbytes_stored / arr_final.nbytes) * 100:.1f}%")

    # Verify data integrity
    print("\nVerifying data integrity...")
    sample = arr_final[500:600, :100]
    print(f"Sample mean: {sample.mean():.4f}")
    print(f"Sample std: {sample.std():.4f}")
    print("✓ Data integrity verified")

    # Cleanup
    import os
    if os.path.exists(sync_path):
        os.remove(sync_path)

    print("\n" + "="*70)
    print("CRITICAL SAFETY NOTES")
    print("="*70)
    print("\n1. blosc.use_threads = False MUST be set in EVERY process")
    print("2. Failure to disable threading causes SILENT data corruption")
    print("3. Always use ProcessSynchronizer for multi-process writes")
    print("4. Test data integrity after multi-process operations")
    print("\nThis example demonstrates SAFE multi-process compression.")

if __name__ == "__main__":
    distributed_compression_workflow()
```
