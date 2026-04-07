# Compression Codecs Patterns

## Pattern 1: Selecting Optimal Codec for Use Case

**Choose the right codec based on priorities:**

```python
import zarr
from numcodecs import Blosc, Zstd, LZ4, GZip, LZMA
import numpy as np

# Priority: Speed (hot data, frequent access)
arr_fast = zarr.open_array(
    "fast_access.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
)

# Priority: Compression ratio (archival, cold storage)
arr_archival = zarr.open_array(
    "archival.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    compressor=LZMA(preset=9)  # Maximum compression, very slow
)

# Priority: Balance (general purpose)
arr_balanced = zarr.open_array(
    "balanced.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
)

# Priority: Compatibility (cross-platform, cross-tool)
arr_portable = zarr.open_array(
    "portable.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    compressor=GZip(level=6)  # Universally supported
)

# Priority: Simplicity (v3 default)
arr_v3 = zarr.open_array(
    "simple_v3.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    zarr_format=3,
    compressors="zstd"  # Simple string specification
)
```

**Decision helper:**
```python
def select_compressor(priority='balanced'):
    """Select compressor based on priority."""
    from numcodecs import Blosc, Zstd, LZ4, GZip, LZMA

    if priority == 'speed':
        return Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    elif priority == 'ratio':
        return LZMA(preset=9)
    elif priority == 'balanced':
        return Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    elif priority == 'compatibility':
        return GZip(level=6)
    else:
        return Blosc(cname='zstd', clevel=3)  # Default

# Usage
compressor = select_compressor('speed')
arr = zarr.zeros((1000, 1000), compressor=compressor)
```

## Pattern 2: Configuring Blosc for Scientific Data

**Optimize Blosc for floating-point scientific data:**

```python
import zarr
from numcodecs import Blosc
import numpy as np

# Scientific floating-point data (temperature, pressure, etc.)
# BITSHUFFLE is critical for best compression

compressor = Blosc(
    cname='zstd',           # Zstandard algorithm (best ratio)
    clevel=3,               # Level 3 (good balance)
    shuffle=Blosc.BITSHUFFLE,  # BITSHUFFLE for floating-point
    blocksize=0             # Auto-select optimal block size
)

arr = zarr.open_array(
    "scientific.zarr",
    mode="w",
    shape=(365, 1000, 2000),  # (time, lat, lon)
    chunks=(30, 100, 200),
    dtype="f4",
    compressor=compressor
)

# Fill with realistic data
data = np.random.randn(365, 1000, 2000).astype("f4") * 5 + 15
arr[:] = data

# Check effectiveness
ratio = arr.nbytes / arr.nbytes_stored
print(f"Compression ratio: {ratio:.2f}x")
print(f"Uncompressed: {arr.nbytes / 1024**3:.2f} GB")
print(f"Compressed: {arr.nbytes_stored / 1024**3:.2f} GB")
```

**Compare shuffle modes:**
```python
import zarr
from numcodecs import Blosc
import numpy as np

data = np.linspace(0, 100, 10000000).astype('f4')

# Test all shuffle modes
shuffle_modes = {
    'NOSHUFFLE': Blosc.NOSHUFFLE,
    'SHUFFLE': Blosc.SHUFFLE,
    'BITSHUFFLE': Blosc.BITSHUFFLE
}

for name, mode in shuffle_modes.items():
    arr = zarr.array(
        data,
        chunks=(100000,),
        compressor=Blosc(cname='zstd', clevel=3, shuffle=mode)
    )
    ratio = arr.nbytes / arr.nbytes_stored
    print(f"{name:12s}: {ratio:5.2f}x ({arr.nbytes_stored / 1024**2:.2f} MB)")

# Expected: BITSHUFFLE gives best results for floating-point
```

## Pattern 3: Safe Multi-Process Compression

**Safely use Blosc in multi-process environments:**

```python
import blosc
import zarr
from zarr import ProcessSynchronizer
from multiprocessing import Process
import numpy as np

def worker_process(store_path, sync_path, rank, n_rows):
    """Worker process - MUST disable Blosc threading."""
    # CRITICAL: Disable Blosc threading in each process
    blosc.use_threads = False

    # Open array with process synchronizer
    arr = zarr.open_array(
        store_path,
        mode="r+",
        synchronizer=ProcessSynchronizer(sync_path)
    )

    # Write this worker's chunk
    start = rank * n_rows
    end = start + n_rows
    data = np.random.randn(n_rows, arr.shape[1]).astype("f4")
    arr[start:end, :] = data

    print(f"Worker {rank}: wrote rows {start}-{end}")

def parallel_write_safe():
    """Safe parallel write with Blosc compression."""
    # CRITICAL: Disable Blosc threading in main process
    blosc.use_threads = False

    store_path = "parallel_data.zarr"
    sync_path = "parallel_data.sync"

    # Create array in main process
    arr = zarr.open_array(
        store_path,
        mode="w",
        shape=(10000, 1000),
        chunks=(1000, 1000),
        dtype="f4",
        compressor=zarr.Blosc(cname='zstd', clevel=3),
        synchronizer=ProcessSynchronizer(sync_path)
    )

    # Launch worker processes
    n_workers = 10
    rows_per_worker = arr.shape[0] // n_workers

    processes = [
        Process(target=worker_process, args=(store_path, sync_path, i, rows_per_worker))
        for i in range(n_workers)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("All workers complete")

    # Verify
    arr_read = zarr.open_array(store_path, mode="r")
    print(f"Final array shape: {arr_read.shape}")
    print(f"Compression ratio: {arr_read.nbytes / arr_read.nbytes_stored:.2f}x")

    # Cleanup
    import os
    if os.path.exists(sync_path):
        os.remove(sync_path)

if __name__ == "__main__":
    parallel_write_safe()
```

**Thread control pattern:**
```python
import blosc
import zarr
from threading import Thread
import numpy as np

# For multi-threading (within single process)
# Can keep Blosc threading enabled OR limit threads

def single_process_multithreading():
    """Multi-threading within single process."""

    # Option 1: Keep Blosc threading, limit threads
    blosc.set_nthreads(4)  # Use 4 internal Blosc threads
    blosc.use_threads = True

    # Option 2: Disable Blosc threading for consistency
    # blosc.use_threads = False

    arr = zarr.open_array(
        "threaded.zarr",
        mode="w",
        shape=(10000, 1000),
        chunks=(1000, 1000),
        dtype="f4",
        compressor=zarr.Blosc(cname='zstd', clevel=3),
        synchronizer=zarr.ThreadSynchronizer()
    )

    def write_chunk(start):
        arr[start:start+1000, :] = np.random.randn(1000, 1000).astype("f4")

    threads = [Thread(target=write_chunk, args=(i*1000,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

## Pattern 4: Applying Pre-Compression Filters

**Use filters to improve compression for specific data types:**

```python
import zarr
from numcodecs import Delta, Quantize, Blosc
import numpy as np

# Pattern 1: Delta filter for time-series data
# Temperature data with small temporal changes
time_series = 15 + np.cumsum(np.random.randn(10000, 100, 100) * 0.1, axis=0)

arr_delta = zarr.open_array(
    "timeseries_delta.zarr",
    mode="w",
    shape=time_series.shape,
    chunks=(100, 100, 100),
    dtype="f4",
    filters=[Delta(dtype='f4')],  # Delta encoding before compression
    compressor=Blosc(cname='zstd', clevel=3)
)
arr_delta[:] = time_series.astype("f4")

# Without Delta filter for comparison
arr_no_delta = zarr.array(
    time_series.astype("f4"),
    chunks=(100, 100, 100),
    compressor=Blosc(cname='zstd', clevel=3)
)

print(f"With Delta filter:    {arr_delta.nbytes / arr_delta.nbytes_stored:.2f}x")
print(f"Without Delta filter: {arr_no_delta.nbytes / arr_no_delta.nbytes_stored:.2f}x")
# Delta filter should give significantly better compression

# Pattern 2: Quantize filter for limited-precision floats
# Round to 2 decimal places - acceptable for many applications
arr_quantized = zarr.open_array(
    "quantized.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    filters=[Quantize(digits=2, dtype='f4')],  # Keep 2 decimal places
    compressor=Blosc(cname='zstd', clevel=3)
)

data_precise = np.random.randn(10000, 10000).astype("f4") * 10 + 15
arr_quantized[:] = data_precise

# Values like 15.123456 become 15.12
# Much more compressible
ratio = arr_quantized.nbytes / arr_quantized.nbytes_stored
print(f"Quantized compression ratio: {ratio:.2f}x")
```

**Filter pipeline:**
```python
from numcodecs import Delta, Quantize, Blosc

# Multiple filters applied in order
filters = [
    Quantize(digits=2, dtype='f4'),  # First: quantize
    Delta(dtype='f4')                 # Then: delta encoding
]

arr = zarr.open_array(
    "filtered.zarr",
    mode="w",
    shape=(1000, 1000),
    dtype="f4",
    filters=filters,  # Applied in order before compression
    compressor=Blosc(cname='zstd', clevel=3)
)
```

## Pattern 5: Benchmarking and Selecting Codecs

**Test multiple codecs on your data to find best option:**

```python
import zarr
from numcodecs import Blosc, Zstd, LZ4, GZip, LZMA
import numpy as np
import time

def benchmark_codecs(data):
    """Benchmark multiple codecs on data."""

    codecs = {
        'Blosc+Zstd': Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
        'Blosc+LZ4': Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
        'Zstd': Zstd(level=3),
        'LZ4': LZ4(acceleration=1),
        'Gzip': GZip(level=6),
        'LZMA': LZMA(preset=6),
        'None': None
    }

    results = {}

    for name, compressor in codecs.items():
        # Time compression
        start = time.time()
        arr = zarr.array(data, chunks=data.shape, compressor=compressor)
        compress_time = time.time() - start

        # Time decompression
        start = time.time()
        _ = arr[:]
        decompress_time = time.time() - start

        # Metrics
        ratio = arr.nbytes / arr.nbytes_stored if arr.nbytes_stored > 0 else 0

        results[name] = {
            'ratio': ratio,
            'compress_time': compress_time,
            'decompress_time': decompress_time,
            'size_mb': arr.nbytes_stored / 1024**2
        }

        print(f"{name:15s}: {ratio:5.2f}x  |  Compress: {compress_time:.3f}s  |  Decompress: {decompress_time:.3f}s")

    return results

# Test on your data
data = np.random.randn(5000, 5000).astype("f4")  # Replace with your actual data
results = benchmark_codecs(data)

# Find best for your priority
best_ratio = max(results.items(), key=lambda x: x[1]['ratio'])
fastest = min(results.items(), key=lambda x: x[1]['compress_time'])

print(f"\nBest ratio: {best_ratio[0]} ({best_ratio[1]['ratio']:.2f}x)")
print(f"Fastest: {fastest[0]} ({fastest[1]['compress_time']:.3f}s)")
```

**Adaptive codec selection:**
```python
def select_adaptive_codec(data_sample):
    """Select codec based on data characteristics."""
    import numpy as np
    from numcodecs import Blosc, GZip

    # Test compressibility with fast codec
    test = zarr.array(data_sample, chunks=data_sample.shape, compressor=Blosc(cname='lz4', clevel=1))
    ratio = test.nbytes / test.nbytes_stored

    if ratio < 1.1:
        # Nearly incompressible - disable compression
        return None
    elif ratio < 2.0:
        # Moderately compressible - use fast codec
        return Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    else:
        # Highly compressible - use better ratio codec
        return Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

# Use on your data
sample = data[:100, :100]  # Small sample
codec = select_adaptive_codec(sample)
print(f"Selected codec: {codec}")
```

## Pattern 6: Codec Migration and Recompression

**Change compression codec on existing arrays:**

```python
import zarr
from numcodecs import Blosc, Zstd

# Open existing array (old compression)
arr_old = zarr.open_array("old_data.zarr", mode="r")

print(f"Old compressor: {arr_old.compressor}")
print(f"Old size: {arr_old.nbytes_stored / 1024**2:.2f} MB")

# Create new array with different compression
arr_new = zarr.open_array(
    "new_data.zarr",
    mode="w",
    shape=arr_old.shape,
    chunks=arr_old.chunks,
    dtype=arr_old.dtype,
    compressor=Zstd(level=5)  # New compressor
)

# Copy data (decompresses old, recompresses with new codec)
arr_new[:] = arr_old[:]

# Copy metadata
for key, value in arr_old.attrs.items():
    arr_new.attrs[key] = value

print(f"New compressor: {arr_new.compressor}")
print(f"New size: {arr_new.nbytes_stored / 1024**2:.2f} MB")
print(f"Size change: {(arr_new.nbytes_stored - arr_old.nbytes_stored) / arr_old.nbytes_stored * 100:.1f}%")
```

**Chunk-wise recompression (memory efficient):**
```python
import zarr
from numcodecs import Blosc

def recompress_array(source_path, dest_path, new_compressor):
    """Recompress array chunk by chunk."""
    src = zarr.open_array(source_path, mode="r")

    dst = zarr.open_array(
        dest_path,
        mode="w",
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        compressor=new_compressor
    )

    # Copy chunk by chunk
    for i in range(src.cdata_shape[0]):
        for j in range(src.cdata_shape[1]):
            chunk = src.blocks[i, j]
            dst.blocks[i, j] = chunk

    # Copy metadata
    dst.attrs.update(src.attrs)

    print(f"Recompressed: {source_path} → {dest_path}")
    print(f"Old: {src.nbytes_stored / 1024**2:.2f} MB")
    print(f"New: {dst.nbytes_stored / 1024**2:.2f} MB")

# Usage
recompress_array(
    "old.zarr",
    "new.zarr",
    Blosc(cname='zstd', clevel=7, shuffle=Blosc.BITSHUFFLE)
)
```

## Pattern 7: Disabling Compression Strategically

**Know when to disable compression:**

```python
import zarr
import numpy as np

# Case 1: Already compressed data (images, video)
compressed_images = np.load("jpeg_images.npy")  # Already JPEG compressed

arr_images = zarr.open_array(
    "images.zarr",
    mode="w",
    shape=compressed_images.shape,
    chunks=(10, 512, 512, 3),
    dtype=compressed_images.dtype,
    compressor=None  # Don't re-compress JPEG data
)

# Case 2: Random/encrypted data
encrypted_data = np.random.bytes(10000 * 10000)  # Uncompressible

arr_encrypted = zarr.open_array(
    "encrypted.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="u1",
    compressor=None  # No benefit, only overhead
)

# Case 3: Performance-critical writes
arr_fast_write = zarr.open_array(
    "fast_write.zarr",
    mode="w",
    shape=(100000, 1000),
    chunks=(1000, 1000),
    dtype="f4",
    compressor=None  # Maximum write speed
)

# Case 4: Testing/debugging
arr_debug = zarr.open_array(
    "debug.zarr",
    mode="w",
    shape=(1000, 1000),
    dtype="f4",
    compressor=None  # Easier to inspect raw data
)
```

**Conditional compression:**
```python
def create_array_smart_compression(path, data_type='scientific'):
    """Create array with appropriate compression for data type."""

    if data_type == 'scientific':
        # Floating-point scientific data
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    elif data_type == 'images':
        # Already compressed images
        compressor = None
    elif data_type == 'integers':
        # Integer data
        compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)
    elif data_type == 'text':
        # Text/string data
        compressor = zarr.GZip(level=6)
    else:
        # Default
        compressor = zarr.Blosc(cname='zstd', clevel=3)

    return zarr.open_array(
        path,
        mode="w",
        shape=(1000, 1000),
        dtype="f4",
        compressor=compressor
    )
```
