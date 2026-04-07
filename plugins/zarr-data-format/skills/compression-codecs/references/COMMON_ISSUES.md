# Common Compression Issues and Solutions

## Issue 1: Poor Compression Ratio Despite Enabling Compression

**Problem:** Compression ratio is less than 1.5x despite using Blosc or other codecs.

**Symptoms:**
```python
arr = zarr.array(data, compressor=zarr.Blosc(cname='zstd', clevel=9))
ratio = arr.nbytes / arr.nbytes_stored
print(f"Ratio: {ratio:.2f}x")  # Only 1.2x - expected much better
```

**Root Causes:**
1. Data is inherently incompressible (random, encrypted, already compressed)
2. Wrong shuffle mode for data type
3. Chunks too small (overhead dominates)
4. Data type not suitable for compression

**Solutions:**

```python
import zarr
from numcodecs import Blosc
import numpy as np

# Diagnostic: Check data compressibility
def diagnose_compressibility(data):
    """Test if data is compressible."""

    # Test 1: Entropy check (rough estimate)
    flat_data = data.flat
    unique_ratio = len(np.unique(flat_data)) / len(flat_data)

    print(f"Unique value ratio: {unique_ratio:.3f}")
    if unique_ratio > 0.9:
        print("⚠️ High entropy - data may be incompressible")

    # Test 2: Quick compression test
    test = zarr.array(data[:100, :100] if len(data.shape) > 1 else data[:1000],
                      compressor=Blosc(cname='lz4', clevel=1))
    ratio = test.nbytes / test.nbytes_stored

    print(f"Test compression ratio: {ratio:.2f}x")

    if ratio < 1.2:
        print("⚠️ Poor compression - consider disabling compression")
        return None  # Disable compression
    elif ratio < 2.0:
        print("✓ Moderate compression - use fast codec")
        return Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    else:
        print("✓ Good compression - use balanced codec")
        return Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Example: Random vs structured data
random_data = np.random.randn(1000, 1000).astype('f4')
structured_data = np.cumsum(np.cumsum(random_data, axis=0), axis=1)

print("Random data:")
codec_random = diagnose_compressibility(random_data)

print("\nStructured data:")
codec_structured = diagnose_compressibility(structured_data)

# Use appropriate codec or disable compression
arr_random = zarr.array(random_data, compressor=codec_random)
arr_structured = zarr.array(structured_data, compressor=codec_structured)

print(f"\nRandom ratio: {arr_random.nbytes / arr_random.nbytes_stored:.2f}x")
print(f"Structured ratio: {arr_structured.nbytes / arr_structured.nbytes_stored:.2f}x")
```

**Fix shuffle mode for float data:**
```python
# WRONG for floating-point data
compressor_wrong = Blosc(cname='zstd', clevel=3, shuffle=Blosc.NOSHUFFLE)

# CORRECT for floating-point data
compressor_correct = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Test difference
data_float = np.linspace(0, 100, 1000000).astype('f4')

arr_wrong = zarr.array(data_float, chunks=(100000,), compressor=compressor_wrong)
arr_correct = zarr.array(data_float, chunks=(100000,), compressor=compressor_correct)

print(f"NOSHUFFLE: {arr_wrong.nbytes / arr_wrong.nbytes_stored:.2f}x")
print(f"BITSHUFFLE: {arr_correct.nbytes / arr_correct.nbytes_stored:.2f}x")
# BITSHUFFLE should give much better results for floating-point
```

## Issue 2: Silent Data Corruption in Multi-Process Code

**Problem:** Data corruption occurs when using Blosc with multiprocessing, with NO error messages.

**Symptoms:**
- Checksum mismatches
- Inconsistent read values
- Wrong data retrieved
- No exceptions or warnings

**Root Cause:**
Blosc internal threading conflicts with multi-process writes, causing race conditions that corrupt data silently.

**Solution:**

```python
import blosc
import zarr
from zarr import ProcessSynchronizer
from multiprocessing import Process
import numpy as np

# CRITICAL: This MUST be in every process that uses Blosc
blosc.use_threads = False

def unsafe_worker(store_path):
    """UNSAFE: Will cause data corruption!"""
    # Missing: blosc.use_threads = False
    arr = zarr.open_array(store_path, mode="r+")
    arr[0:1000, :] = np.random.randn(1000, 1000)
    # Data may be corrupted!

def safe_worker(store_path, sync_path, rank):
    """SAFE: Blosc threading disabled."""
    # CRITICAL: Disable Blosc threading
    blosc.use_threads = False

    arr = zarr.open_array(
        store_path,
        mode="r+",
        synchronizer=ProcessSynchronizer(sync_path)
    )
    arr[rank*1000:(rank+1)*1000, :] = np.random.randn(1000, 1000)

# Correct usage
blosc.use_threads = False  # Also in main process

arr = zarr.open_array(
    "safe.zarr",
    mode="w",
    shape=(10000, 1000),
    chunks=(1000, 1000),
    dtype='f4',
    compressor=zarr.Blosc(cname='zstd', clevel=3),
    synchronizer=ProcessSynchronizer("safe.sync")
)

workers = [Process(target=safe_worker, args=("safe.zarr", "safe.sync", i))
           for i in range(10)]
for w in workers:
    w.start()
for w in workers:
    w.join()
```

**Verification test:**
```python
def verify_no_corruption(arr):
    """Verify data integrity after multi-process write."""

    # Check for NaN or inf (signs of corruption)
    if np.any(np.isnan(arr[:])) or np.any(np.isinf(arr[:])):
        print("⚠️ CORRUPTION DETECTED: NaN or Inf values present")
        return False

    # Check statistics are reasonable
    data = arr[:]
    if abs(data.mean()) > 100 or data.std() > 100:
        print("⚠️ CORRUPTION SUSPECTED: Unusual statistics")
        return False

    print("✓ Data integrity check passed")
    return True

# Always verify after multi-process writes
verify_no_corruption(arr)
```

**Prevention checklist:**
```python
# Add to top of every Python file using Blosc with multiprocessing
import blosc
blosc.use_threads = False

# Verify setting
assert blosc.use_threads == False, "Blosc threading MUST be disabled!"

# Document in code
"""
CRITICAL: blosc.use_threads = False is REQUIRED for multi-process safety.
Failure to set this causes SILENT data corruption.
"""
```

## Issue 3: Compression Slower Than Expected

**Problem:** Compression overhead makes writes unacceptably slow.

**Symptoms:**
- Write operations much slower than uncompressed
- Compression time >> computation time
- High CPU usage during writes

**Root Causes:**
1. Compression level too high (diminishing returns)
2. Wrong codec for data type
3. Blosc threading disabled when single-process (unnecessary)
4. Small chunks causing excessive overhead

**Solutions:**

```python
import zarr
from numcodecs import Blosc, LZ4
import numpy as np
import time

def benchmark_compression_speed(data, compressors):
    """Compare compression speeds."""

    results = []

    for name, compressor in compressors.items():
        start = time.time()
        arr = zarr.array(data, chunks=data.shape, compressor=compressor)
        write_time = time.time() - start

        ratio = arr.nbytes / arr.nbytes_stored

        results.append({
            'name': name,
            'time': write_time,
            'ratio': ratio,
            'throughput_mb_s': (data.nbytes / 1024**2) / write_time
        })

        print(f"{name:20s}: {write_time:.3f}s ({ratio:.2f}x) - {results[-1]['throughput_mb_s']:.0f} MB/s")

    return results

# Test data
data = np.random.randn(5000, 5000).astype('f4')

compressors = {
    'No compression': None,
    'Blosc+LZ4 level 1': Blosc(cname='lz4', clevel=1),
    'Blosc+LZ4 level 5': Blosc(cname='lz4', clevel=5),
    'Blosc+Zstd level 1': Blosc(cname='zstd', clevel=1),
    'Blosc+Zstd level 3': Blosc(cname='zstd', clevel=3),
    'Blosc+Zstd level 9': Blosc(cname='zstd', clevel=9),
    'LZ4 standalone': LZ4(acceleration=1),
}

print("Compression speed comparison:")
print("-" * 80)
results = benchmark_compression_speed(data, compressors)

# Recommendation based on results
fastest = min(results, key=lambda x: x['time'])
best_balance = max(results, key=lambda x: x['ratio'] / x['time'])

print(f"\nFastest: {fastest['name']} ({fastest['time']:.3f}s)")
print(f"Best balance: {best_balance['name']} (ratio/time = {best_balance['ratio']/best_balance['time']:.2f})")
```

**Speed optimization strategies:**
```python
# Strategy 1: Use lower compression level
# High levels give diminishing returns
compressor_fast = Blosc(cname='zstd', clevel=1)  # Much faster, still good ratio

# Strategy 2: Use faster algorithm
compressor_fastest = Blosc(cname='lz4', clevel=5)  # Fastest Blosc option

# Strategy 3: Enable Blosc threading (if single-process)
import blosc
if not using_multiprocessing:
    blosc.use_threads = True
    blosc.set_nthreads(4)  # Use 4 threads

# Strategy 4: Increase chunk size (reduces overhead)
# Larger chunks = fewer compression operations
arr = zarr.open_array(
    "optimized.zarr",
    mode="w",
    shape=(100000, 1000),
    chunks=(10000, 1000),  # Larger chunks
    dtype='f4',
    compressor=Blosc(cname='lz4', clevel=5)
)

# Strategy 5: Disable compression if not needed
# If compression ratio < 1.3x, overhead may not be worth it
compressor = None  # No compression, maximum speed
```

## Issue 4: Filter Configuration Errors

**Problem:** Filters fail with cryptic error messages or don't improve compression.

**Symptoms:**
```python
ValueError: filter output has unexpected dtype
TypeError: Cannot apply filter to this dtype
# Or: Filter applied but no compression improvement
```

**Root Causes:**
1. Filter dtype mismatch with array dtype
2. Wrong filter for data characteristics
3. Filter order incorrect
4. Filter not compatible with codec

**Solutions:**

```python
import zarr
from numcodecs import Delta, Quantize, Blosc
import numpy as np

# Problem 1: Dtype mismatch
# WRONG: Filter dtype doesn't match array dtype
try:
    arr_wrong = zarr.open_array(
        "wrong_dtype.zarr",
        mode="w",
        shape=(1000, 1000),
        dtype='f4',  # Array is float32
        filters=[Delta(dtype='f8')],  # Filter is float64 - MISMATCH!
        compressor=Blosc(cname='zstd', clevel=3)
    )
except Exception as e:
    print(f"Error: {e}")

# CORRECT: Match filter dtype to array dtype
arr_correct = zarr.open_array(
    "correct_dtype.zarr",
    mode="w",
    shape=(1000, 1000),
    dtype='f4',
    filters=[Delta(dtype='f4')],  # Match array dtype
    compressor=Blosc(cname='zstd', clevel=3)
)

# Problem 2: Wrong filter for data type
# Delta filter only helps with correlated data
random_data = np.random.randn(1000, 1000).astype('f4')
correlated_data = np.cumsum(np.cumsum(random_data, axis=0), axis=1).astype('f4')

# Delta on random data (no benefit)
arr_random_delta = zarr.array(random_data, filters=[Delta(dtype='f4')],
                               compressor=Blosc(cname='zstd', clevel=3))

# Delta on correlated data (significant benefit)
arr_corr_delta = zarr.array(correlated_data, filters=[Delta(dtype='f4')],
                             compressor=Blosc(cname='zstd', clevel=3))

print("Random data + Delta:")
print(f"  Ratio: {arr_random_delta.nbytes / arr_random_delta.nbytes_stored:.2f}x")

print("Correlated data + Delta:")
print(f"  Ratio: {arr_corr_delta.nbytes / arr_corr_delta.nbytes_stored:.2f}x")
# Correlated should show much better improvement

# Problem 3: Filter order matters
# WRONG order: Quantize after Delta
filters_wrong = [
    Delta(dtype='f4'),
    Quantize(digits=2, dtype='f4')
]

# CORRECT order: Quantize before Delta
filters_correct = [
    Quantize(digits=2, dtype='f4'),  # First: reduce precision
    Delta(dtype='f4')                 # Then: delta encoding
]

# Quantize creates patterns that Delta can exploit
```

**Filter validation:**
```python
def validate_filter_config(dtype, filters):
    """Validate filter configuration."""

    if filters is None:
        return True

    for i, f in enumerate(filters):
        # Check dtype compatibility
        if hasattr(f, 'dtype'):
            if f.dtype != dtype:
                print(f"⚠️ Filter {i} dtype {f.dtype} doesn't match array dtype {dtype}")
                return False

        # Check filter type
        filter_name = type(f).__name__
        if filter_name == 'Delta':
            print(f"✓ Delta filter - ensure data is correlated")
        elif filter_name == 'Quantize':
            print(f"✓ Quantize filter - precision will be reduced")

    return True

# Usage
filters = [Delta(dtype='f4')]
if validate_filter_config('f4', filters):
    arr = zarr.zeros((1000, 1000), dtype='f4', filters=filters)
```

## Issue 5: Codec Compatibility Across Platforms

**Problem:** Array created on one system can't be read on another.

**Symptoms:**
```python
ValueError: codec not available: 'blosc'
ImportError: No module named 'numcodecs'
```

**Root Causes:**
1. Codec not installed on reading system
2. Different numcodecs versions
3. Platform-specific codec (Blosc variants)

**Solutions:**

```python
import zarr
from numcodecs import GZip, Zstd
import numpy as np

# Problem: Blosc may not be available everywhere
# Solution: Use universally available codecs for portability

# Most portable: Gzip (available everywhere)
arr_portable = zarr.open_array(
    "portable.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype='f4',
    compressor=GZip(level=6)  # Universally supported
)

# Good portability: Zstd (widely available, better than Gzip)
arr_zstd = zarr.open_array(
    "zstd.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype='f4',
    compressor=Zstd(level=3)
)

# Check codec availability
def check_codec_available(codec_name):
    """Check if codec is available."""
    try:
        from numcodecs import get_codec
        codec = get_codec({'id': codec_name})
        return True
    except Exception:
        return False

codecs_to_check = ['gzip', 'zstd', 'blosc', 'lz4']
for codec in codecs_to_check:
    available = check_codec_available(codec)
    print(f"{codec:10s}: {'✓ Available' if available else '✗ Not available'}")

# Document codec requirements
arr_portable.attrs['codec_requirements'] = 'gzip (standard library)'
arr_zstd.attrs['codec_requirements'] = 'zstd (pip install zstandard)'
```

**Fallback strategy:**
```python
def create_array_with_fallback(path, shape, dtype):
    """Create array with fallback codec selection."""
    from numcodecs import Blosc, Zstd, GZip

    # Try codecs in order of preference
    codecs = [
        ('Blosc+Zstd', Blosc(cname='zstd', clevel=3)),
        ('Zstd', Zstd(level=3)),
        ('Gzip', GZip(level=6)),
        ('None', None)
    ]

    for name, codec in codecs:
        try:
            arr = zarr.open_array(
                path,
                mode="w",
                shape=shape,
                dtype=dtype,
                compressor=codec
            )
            print(f"Using codec: {name}")
            return arr
        except Exception as e:
            print(f"Codec {name} failed: {e}")
            continue

    raise RuntimeError("No compatible codec found")

# Usage
arr = create_array_with_fallback("fallback.zarr", (1000, 1000), 'f4')
```

## Issue 6: Metadata Indicates Wrong Compressor

**Problem:** Array metadata shows compressor but data isn't actually compressed.

**Symptoms:**
```python
print(arr.compressor)  # Shows: Blosc(...)
print(arr.nbytes_stored == arr.nbytes)  # True - no compression!
```

**Root Cause:**
Compressor set but `compressor=None` was passed to write operation, or chunks never written.

**Solution:**

```python
import zarr
from numcodecs import Blosc

# Check actual compression status
def verify_compression(arr):
    """Verify array is actually compressed."""

    print(f"Compressor in metadata: {arr.compressor}")
    print(f"Uncompressed size: {arr.nbytes / 1024**2:.2f} MB")
    print(f"Stored size: {arr.nbytes_stored / 1024**2:.2f} MB")

    if arr.compressor is None:
        print("✓ No compressor (expected)")
        return

    ratio = arr.nbytes / arr.nbytes_stored if arr.nbytes_stored > 0 else 0

    if ratio < 1.01:
        print("⚠️ WARNING: Compressor set but data not compressed!")
        print("Possible causes:")
        print("  1. Chunks not initialized (no data written)")
        print("  2. Compression disabled during write")
        print("  3. Data truly incompressible")
    else:
        print(f"✓ Data compressed ({ratio:.2f}x ratio)")

# Test
arr = zarr.open_array(
    "test.zarr",
    mode="w",
    shape=(1000, 1000),
    dtype='f4',
    compressor=Blosc(cname='zstd', clevel=3)
)

verify_compression(arr)  # Warning: no data written yet

# Write data
arr[:] = np.random.randn(1000, 1000).astype('f4')

verify_compression(arr)  # Now should show compression
```

## Issue 7: Excessive Memory Usage During Compression

**Problem:** Compression uses more memory than expected, causing OOM errors.

**Symptoms:**
```python
MemoryError: Unable to allocate X GB for array
```

**Root Causes:**
1. Compression buffer larger than chunk
2. All chunks loaded into memory
3. Blosc internal buffers

**Solutions:**

```python
import zarr
from numcodecs import Blosc
import numpy as np

# Problem: Chunks too large
# Each chunk is compressed in memory

# WRONG: 100 MB chunks
arr_large_chunks = zarr.open_array(
    "large_chunks.zarr",
    mode="w",
    shape=(100000, 10000),
    chunks=(10000, 10000),  # 400 MB per chunk!
    dtype='f4',
    compressor=Blosc(cname='zstd', clevel=3)
)

# CORRECT: Reasonable chunk size (2-5 MB)
arr_good_chunks = zarr.open_array(
    "good_chunks.zarr",
    mode="w",
    shape=(100000, 10000),
    chunks=(500, 1000),  # 2 MB per chunk
    dtype='f4',
    compressor=Blosc(cname='zstd', clevel=3)
)

# Write chunk by chunk to control memory
chunk_size = arr_good_chunks.chunks[0]

for i in range(0, arr_good_chunks.shape[0], chunk_size):
    end = min(i + chunk_size, arr_good_chunks.shape[0])
    chunk_data = np.random.randn(end - i, arr_good_chunks.shape[1]).astype('f4')
    arr_good_chunks[i:end, :] = chunk_data
    # Each chunk compressed and written before next chunk loaded

print("Memory-efficient compression complete")
```

**Monitor memory usage:**
```python
import psutil
import os

def monitor_compression_memory(arr, data):
    """Monitor memory usage during compression."""

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2

    arr[:] = data

    mem_after = process.memory_info().rss / 1024**2
    mem_used = mem_after - mem_before

    print(f"Memory before: {mem_before:.0f} MB")
    print(f"Memory after: {mem_after:.0f} MB")
    print(f"Memory used: {mem_used:.0f} MB")
    print(f"Data size: {data.nbytes / 1024**2:.0f} MB")

    if mem_used > data.nbytes / 1024**2 * 2:
        print("⚠️ Memory usage > 2x data size - consider smaller chunks")
```
