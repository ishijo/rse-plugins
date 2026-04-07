# Common Issues and Solutions

## Issue 1: Performance Issues with Small Chunks

**Problem:** Slow read/write operations or high memory usage despite chunking.

**Symptoms:**
- Operations slower than expected
- High overhead for small data reads
- Excessive memory usage
- Cloud storage costs high due to many small objects

**Root Cause:**
Chunks are too small, leading to:
- High metadata overhead (many chunk files)
- Frequent I/O operations
- Poor compression ratios
- Inefficient cloud object storage usage

**Solution:**

```python
import numpy as np
import zarr

# BAD: Chunks too small
arr_bad = zarr.open_array(
    "data_bad.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(10, 10),  # Only 400 bytes per chunk!
    dtype="f4"
)

# Calculate chunk size
chunk_bytes = np.prod(arr_bad.chunks) * np.dtype(arr_bad.dtype).itemsize
print(f"Chunk size: {chunk_bytes / 1024:.2f} KB")  # Too small!

# GOOD: Target 1-3 MB chunks
target_mb = 2.0
shape = (10000, 10000)
dtype = np.float32

# Calculate chunk size to hit target
itemsize = np.dtype(dtype).itemsize
target_bytes = target_mb * 1024**2
items_per_chunk = target_bytes / itemsize

# For 2D array, use square chunks
chunk_size = int(np.sqrt(items_per_chunk))

arr_good = zarr.open_array(
    "data_good.zarr",
    mode="w",
    shape=shape,
    chunks=(chunk_size, chunk_size),
    dtype=dtype
)

print(f"Optimal chunk size: {chunk_size} × {chunk_size}")
print(f"Chunk size: {arr_good.nbytes / 1024**2:.2f} MB per chunk")  # ~2 MB

# Verify with realistic access
data = np.random.randn(*shape).astype(dtype)
arr_good[:] = data
```

**Best Practices:**
- Target 1-3 MB uncompressed chunk size
- Align chunks with access patterns (time-series vs spatial)
- Use larger chunks for better compression
- Consider cloud object storage limits (typical min: 128 KB)

**Diagnostic:**
```python
def diagnose_chunking(arr):
    """Diagnose chunking configuration."""
    chunk_bytes = np.prod(arr.chunks) * np.dtype(arr.dtype).itemsize
    chunk_mb = chunk_bytes / 1024**2
    total_chunks = int(np.prod([s / c for s, c in zip(arr.shape, arr.chunks)]))

    print(f"Shape: {arr.shape}")
    print(f"Chunks: {arr.chunks}")
    print(f"Chunk size: {chunk_mb:.2f} MB")
    print(f"Total chunks: {total_chunks:,}")

    if chunk_mb < 0.1:
        print("⚠️  WARNING: Chunks very small (< 0.1 MB). Consider larger chunks.")
    elif chunk_mb < 0.5:
        print("⚠️  WARNING: Chunks small (< 0.5 MB). May have high overhead.")
    elif chunk_mb > 100:
        print("⚠️  WARNING: Chunks very large (> 100 MB). May cause memory issues.")
    else:
        print("✓ Chunk size appears reasonable.")

    if total_chunks > 100000:
        print(f"⚠️  WARNING: Many chunks ({total_chunks:,}). Consider sharding (v3) or larger chunks.")

diagnose_chunking(arr_good)
```

## Issue 2: Compression Codec Selection

**Problem:** Poor compression ratio or slow I/O despite enabling compression.

**Symptoms:**
- Compressed size nearly same as uncompressed
- Write operations very slow
- Compression doesn't help performance

**Root Cause:**
- Data is inherently incompressible (random noise)
- Wrong codec for data characteristics
- Compression level too high (diminishing returns)

**Solution:**

```python
import zarr
import numpy as np
import time

# Test different compressors
data = np.random.randn(1000, 1000).astype("f4")  # 4 MB

def test_compressor(name, compressor, data):
    """Test compression performance."""
    arr = zarr.open_array(
        f"test_{name}.zarr",
        mode="w",
        shape=data.shape,
        chunks=(100, 100),
        dtype=data.dtype,
        compressor=compressor
    )

    # Time write
    start = time.time()
    arr[:] = data
    write_time = time.time() - start

    # Time read
    start = time.time()
    _ = arr[:]
    read_time = time.time() - start

    # Stats
    uncompressed = arr.nbytes
    compressed = arr.nbytes_stored
    ratio = uncompressed / compressed if compressed > 0 else 0

    print(f"{name:15} | Ratio: {ratio:4.2f}x | Write: {write_time:.3f}s | Read: {read_time:.3f}s | Size: {compressed/1024**2:.2f} MB")

    return ratio, write_time, read_time

print("Compressor      | Ratio | Write Time | Read Time | Stored Size")
print("-" * 70)

# Test various compressors
test_compressor("No compression", None, data)
test_compressor("Blosc+Zstd-1", zarr.Blosc(cname="zstd", clevel=1), data)
test_compressor("Blosc+Zstd-3", zarr.Blosc(cname="zstd", clevel=3), data)
test_compressor("Blosc+Zstd-9", zarr.Blosc(cname="zstd", clevel=9), data)
test_compressor("Blosc+LZ4", zarr.Blosc(cname="lz4", clevel=5), data)
test_compressor("Gzip-6", zarr.codecs.GZip(level=6), data)

# For scientific data (patterns, not random)
scientific_data = np.zeros((1000, 1000), dtype="f4")
# Add spatial structure
for i in range(1000):
    for j in range(1000):
        scientific_data[i, j] = 15 * np.cos(2 * np.pi * i / 100) + 5 * np.sin(2 * np.pi * j / 50)

print("\nScientific data (structured patterns):")
print("-" * 70)
test_compressor("Blosc+Zstd-3", zarr.Blosc(cname="zstd", clevel=3), scientific_data)
```

**Recommendations:**
- **Random data**: Compression won't help - disable it (`compressor=None`)
- **Scientific data**: Blosc+Zstd level 3 is good default
- **Maximum compression**: Use LZMA or Blosc+Zstd level 9 (but slow)
- **Speed priority**: Blosc+LZ4
- **Universal compatibility**: Gzip

**When compression fails:**
```python
# Check if data is compressible
ratio = arr.nbytes / arr.nbytes_stored
if ratio < 1.2:
    print("⚠️ Compression ineffective (< 1.2x)")
    print("Consider:")
    print("  1. Disable compression for random/encrypted data")
    print("  2. Try different codec (data may have patterns)")
    print("  3. Apply filters before compression (Delta, Quantize)")
```

## Issue 3: Concurrent Access Errors

**Problem:** Data corruption or errors when multiple threads/processes write to same array.

**Symptoms:**
```
RuntimeError: Concurrent writes detected
ValueError: Inconsistent chunk data
```

**Root Cause:**
Multiple writers accessing same chunks without synchronization.

**Solution:**

```python
import zarr
from zarr import ThreadSynchronizer, ProcessSynchronizer
import threading
import numpy as np

# WRONG: No synchronization
arr_unsafe = zarr.open_array(
    "unsafe.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4"
)

# This can cause corruption!
# def unsafe_write(i):
#     arr_unsafe[i*1000:(i+1)*1000, :] = np.random.randn(1000, 10000)

# CORRECT: Use synchronizer for threads
arr_safe_thread = zarr.open_array(
    "safe_thread.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    synchronizer=ThreadSynchronizer()  # Thread-safe
)

def safe_write_thread(arr, i):
    """Thread-safe write."""
    arr[i*1000:(i+1)*1000, :] = np.random.randn(1000, 10000).astype("f4")

threads = [threading.Thread(target=safe_write_thread, args=(arr_safe_thread, i)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("✓ Thread-safe writes complete")

# CORRECT: Use synchronizer for processes
from multiprocessing import Process

def safe_write_process(store_path, sync_path, i):
    """Process-safe write - must reopen array in each process."""
    arr = zarr.open_array(
        store_path,
        mode="r+",
        synchronizer=ProcessSynchronizer(sync_path)  # Same synchronizer
    )
    arr[i*1000:(i+1)*1000, :] = np.random.randn(1000, 10000).astype("f4")

store_path = "safe_process.zarr"
sync_path = "safe_process.sync"

# Create in main process
arr_safe_process = zarr.open_array(
    store_path,
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    synchronizer=ProcessSynchronizer(sync_path)
)

processes = [Process(target=safe_write_process, args=(store_path, sync_path, i)) for i in range(10)]
for p in processes:
    p.start()
for p in processes:
    p.join()

print("✓ Process-safe writes complete")

# Clean up lock file
import os
if os.path.exists(sync_path):
    os.remove(sync_path)
```

**Best Practices:**
- Always use `ThreadSynchronizer()` for multithreading
- Always use `ProcessSynchronizer(path)` for multiprocessing
- Each process must reopen array with same synchronizer
- Writes to different chunks are safe even without synchronization
- Synchronizer adds overhead - only use when needed

## Issue 4: Format Compatibility (v2 vs v3)

**Problem:** Array created with v3 cannot be read by tools expecting v2.

**Symptoms:**
```
ValueError: Zarr format 3 not supported
FileNotFoundError: .zarray not found
```

**Root Cause:**
Metadata format differs between v2 (.zarray) and v3 (zarr.json).

**Solution:**

```python
import zarr
import numpy as np

# Check array format
def check_zarr_format(path):
    """Determine Zarr format version."""
    import os

    zarray_path = os.path.join(path, ".zarray")
    zarr_json_path = os.path.join(path, "zarr.json")

    if os.path.exists(zarr_json_path):
        return 3
    elif os.path.exists(zarray_path):
        return 2
    else:
        return None

# Explicitly specify format
arr_v2 = zarr.open_array(
    "explicit_v2.zarr",
    mode="w",
    shape=(1000, 1000),
    chunks=(100, 100),
    dtype="f4",
    zarr_format=2  # Explicit v2
)

arr_v3 = zarr.open_array(
    "explicit_v3.zarr",
    mode="w",
    shape=(1000, 1000),
    chunks=(100, 100),
    dtype="f4",
    zarr_format=3  # Explicit v3
)

print(f"Array v2 format: {check_zarr_format('explicit_v2.zarr')}")
print(f"Array v3 format: {check_zarr_format('explicit_v3.zarr')}")

# Convert v3 to v2 for compatibility
def convert_v3_to_v2(v3_path, v2_path):
    """Convert Zarr v3 to v2."""
    src = zarr.open_array(v3_path, mode="r")
    dst = zarr.open_array(
        v2_path,
        mode="w",
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        zarr_format=2,  # Force v2
        compressor=zarr.Blosc(cname="zstd", clevel=3)  # v2 compressor
    )
    # Copy data
    dst[:] = src[:]
    # Copy metadata
    for key, value in src.attrs.items():
        dst.attrs[key] = value

    print(f"Converted {v3_path} (v3) → {v2_path} (v2)")

convert_v3_to_v2("explicit_v3.zarr", "converted_v2.zarr")
```

**Compatibility guidelines:**
- Default is v2 - most compatible
- v3 requires Python 3.11+ and recent zarr-python
- Use v2 for maximum ecosystem compatibility
- Use v3 when you need sharding or async I/O
- Document format version in project requirements

## Issue 5: Cloud Storage Configuration Issues

**Problem:** Errors reading/writing Zarr on S3, GCS, or Azure.

**Symptoms:**
```
PermissionError: Access Denied
FileNotFoundError: Bucket not found
botocore.exceptions.NoCredentialsError
```

**Root Cause:**
- Incorrect authentication
- Wrong bucket/container name
- Missing permissions
- Storage options not configured

**Solution:**

```python
import zarr
import numpy as np

# Local filesystem (works)
arr_local = zarr.open_array(
    "local.zarr",
    mode="w",
    shape=(1000, 1000),
    chunks=(100, 100),
    dtype="f4"
)
arr_local[:] = np.random.randn(1000, 1000).astype("f4")

# S3 storage with fsspec
try:
    import s3fs

    # Method 1: Direct URL (uses environment credentials)
    # Set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    arr_s3_url = zarr.open_array(
        "s3://my-bucket/data.zarr",
        mode="w",
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4"
    )

    # Method 2: Explicit credentials
    fs = s3fs.S3FileSystem(
        key="ACCESS_KEY",
        secret="SECRET_KEY",
        endpoint_url="https://s3.us-west-2.amazonaws.com"
    )

    store = s3fs.S3Map(root="my-bucket/data.zarr", s3=fs)
    arr_s3 = zarr.open_array(
        store,
        mode="w",
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4"
    )

    # Method 3: Anonymous access (public buckets)
    fs_anon = s3fs.S3FileSystem(anon=True)
    store_anon = s3fs.S3Map(root="public-bucket/data.zarr", s3=fs_anon)

except ImportError:
    print("s3fs not installed: pip install s3fs")

# GCS storage
try:
    import gcsfs

    # Uses Google Application Default Credentials
    fs_gcs = gcsfs.GCSFileSystem()
    store_gcs = gcsfs.GCSMap(root="my-bucket/data.zarr", gcs=fs_gcs)

    arr_gcs = zarr.open_array(
        store_gcs,
        mode="w",
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4"
    )

except ImportError:
    print("gcsfs not installed: pip install gcsfs")

# Azure Blob Storage
try:
    import adlfs

    # Connection string method
    fs_azure = adlfs.AzureBlobFileSystem(
        account_name="myaccount",
        account_key="KEY"  # Or use SAS token
    )

    store_azure = adlfs.AzureBlobFile(
        root="container/data.zarr",
        fs=fs_azure
    )

    arr_azure = zarr.open_array(
        store_azure,
        mode="w",
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4"
    )

except ImportError:
    print("adlfs not installed: pip install adlfs")
```

**Troubleshooting checklist:**
1. Check credentials are set (environment variables or config files)
2. Verify bucket/container exists and is accessible
3. Test with anonymous/public data first
4. Check IAM permissions (read/write/list)
5. Use explicit endpoints for non-default regions
6. Install required packages (`s3fs`, `gcsfs`, `adlfs`)

## Issue 6: Memory Errors Despite Chunking

**Problem:** `MemoryError` when processing Zarr arrays even with chunking enabled.

**Symptoms:**
```
MemoryError: Unable to allocate X GB
```

**Root Cause:**
- Loading entire array into memory with `arr[:]`
- Operations that materialize full array
- Chunks still too large for available memory

**Solution:**

```python
import zarr
import numpy as np

# Create large array
arr = zarr.open_array(
    "large.zarr",
    mode="w",
    shape=(100000, 10000),  # 4 GB uncompressed
    chunks=(1000, 1000),
    dtype="f4"
)

# Fill with data
for i in range(0, arr.shape[0], arr.chunks[0]):
    end = min(i + arr.chunks[0], arr.shape[0])
    arr[i:end, :] = np.random.randn(end - i, arr.shape[1]).astype("f4")

# BAD: Loads entire 4 GB into memory!
# data = arr[:]
# result = data.mean()  # Memory error!

# GOOD: Process chunk by chunk
chunk_sums = []
chunk_counts = []

for i in range(0, arr.shape[0], arr.chunks[0]):
    end = min(i + arr.chunks[0], arr.shape[0])
    chunk = arr[i:end, :]  # Only loads ~4 MB
    chunk_sums.append(chunk.sum())
    chunk_counts.append(chunk.size)

# Combine results
total_sum = np.sum(chunk_sums)
total_count = np.sum(chunk_counts)
mean = total_sum / total_count

print(f"Mean (out-of-core): {mean:.6f}")

# BETTER: Use Dask for automatic chunking
import dask.array as da

# Wrap Zarr array as Dask array
darr = da.from_zarr("large.zarr")

# Lazy computation
result = darr.mean()

# Compute (Dask handles chunking automatically)
mean_dask = result.compute()

print(f"Mean (Dask): {mean_dask:.6f}")

# BEST PRACTICE: Use Dask from the start
darr_new = da.random.random((100000, 10000), chunks=(1000, 1000))
da.to_zarr(darr_new, "large_dask.zarr", overwrite=True)
```

**Guidelines:**
- Never use `arr[:]` on large arrays
- Process in chunk-sized batches
- Use Dask for automatic out-of-core computation
- Monitor memory usage during development
- Reduce chunk size if memory errors persist

## Issue 7: Slow Cloud Access

**Problem:** Reading Zarr from cloud storage much slower than expected.

**Symptoms:**
- High latency for small reads
- Many small requests to cloud storage
- Slow metadata operations

**Root Cause:**
- Metadata not consolidated (v2)
- Many small chunks require many requests
- No caching enabled

**Solution:**

```python
import zarr

# Create array on S3
arr = zarr.open_array(
    "s3://my-bucket/data.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4"
)

# Fill with data
# ...

# CRITICAL for v2: Consolidate metadata
# Without this, every array access requires reading .zarray and .zattrs separately
zarr.consolidate_metadata("s3://my-bucket/data.zarr")

# Read with consolidated metadata
arr_fast = zarr.open_consolidated("s3://my-bucket/data.zarr")

# Enable caching for repeated access
import fsspec

# Local cache
cached_fs = fsspec.filesystem(
    "filecache",
    target_protocol="s3",
    cache_storage="/tmp/zarr_cache"
)

arr_cached = zarr.open_array(
    "s3://my-bucket/data.zarr",
    storage_options={"fs": cached_fs}
)

# First access: slow (cache miss)
data1 = arr_cached[0, :]

# Second access: fast (cache hit)
data2 = arr_cached[0, :]
```

**Optimization checklist:**
1. **Consolidate metadata** (v2): Reduces N+1 reads to 1 read
2. **Use larger chunks**: Fewer objects = fewer requests
3. **Enable caching**: Local cache for repeated reads
4. **Use sharding** (v3): Reduces object count
5. **Tune concurrency**: Adjust `zarr.config.set({'async.concurrency': 64})`
6. **Consider Icechunk**: Optimized for cloud with ACID guarantees

**v2 metadata consolidation:**
```python
# After creating/modifying arrays
zarr.consolidate_metadata("data.zarr")

# When reading
root = zarr.open_consolidated("data.zarr", mode="r")

# NOTE: Re-consolidate after adding/modifying arrays
```

## Issue 8: .info_complete() Slow on Large Arrays

**Problem:** Calling `.info_complete()` takes a very long time on large Zarr arrays, especially with many chunks.

**Symptoms:**
- `.info_complete()` hangs or takes minutes/hours
- High I/O activity when inspecting array metadata
- Timeouts when trying to get array information

**Root Cause:**
`.info_complete()` scans all chunks to provide detailed statistics (actual storage size, compression ratio per chunk, etc.). For arrays with thousands or millions of chunks, this requires reading metadata for every chunk.

**Solution:**

```python
import zarr
import numpy as np

# Create large array with many chunks
arr = zarr.open_array(
    "large.zarr",
    mode="w",
    shape=(100000, 10000),
    chunks=(100, 100),  # 1,000,000 chunks!
    dtype="f4"
)

# Fill with data
for i in range(0, arr.shape[0], arr.chunks[0]):
    end = min(i + arr.chunks[0], arr.shape[0])
    arr[i:end, :] = np.random.randn(end - i, arr.shape[1]).astype("f4")

# SLOW: This will scan all 1M chunks
# print(arr.info_complete())  # Don't do this!

# FAST: Use .info for quick metadata only
print(arr.info)

# .info provides:
# - Array shape and dtype
# - Chunk configuration
# - Compressor settings
# - Does NOT scan all chunks
```

**Output comparison:**

```python
# arr.info (fast - milliseconds)
print(arr.info)
# Output:
# Name               : /
# Type               : zarr.Array
# Data type          : float32
# Shape              : (100000, 10000)
# Chunk shape        : (100, 100)
# Order              : C
# Read-only          : False
# Compressor         : Blosc(cname='zstd', clevel=3, shuffle=SHUFFLE)
# Store type         : zarr.storage.DirectoryStore
# No. bytes          : 4000000000 (3.7G)
# Chunks initialized : unknown

# arr.info_complete() (slow - minutes/hours for large arrays)
# Scans every chunk to report:
# - Exact number of initialized chunks
# - Per-chunk storage size
# - Actual compression ratios
```

**When to use each:**

```python
# Quick inspection (always safe)
arr.info  # Use this by default

# Detailed analysis (use sparingly)
# Only call on small arrays or when you need exact storage stats
if arr.nchunks < 1000:  # Safe threshold
    arr.info_complete()
else:
    print(f"⚠️ Array has {arr.nchunks} chunks - .info_complete() will be slow")
    print("Use .info instead for quick metadata")
```

**Alternative approaches for large arrays:**

```python
# Get chunk count without scanning all
n_chunks = int(np.prod([s // c for s, c in zip(arr.shape, arr.chunks)]))
print(f"Total chunks: {n_chunks:,}")

# Estimate storage size
uncompressed_size = arr.nbytes
chunk_size_uncompressed = np.prod(arr.chunks) * arr.dtype.itemsize

print(f"Uncompressed size: {uncompressed_size / 1024**3:.2f} GB")
print(f"Chunk size: {chunk_size_uncompressed / 1024**2:.2f} MB")

# Sample chunks to estimate compression ratio
sample_size = min(100, n_chunks)
sample_indices = np.random.choice(n_chunks, sample_size, replace=False)

# This is still slow but faster than scanning all chunks
# Only do this if you really need compression stats
```

**Best practices:**
- Use `.info` for routine inspection (fast, always safe)
- Use `.info_complete()` only on small arrays (< 1000 chunks)
- For large arrays, estimate metrics rather than computing exactly
- Document chunk configuration in array attributes
- Monitor chunk count during array design - excessive chunks indicate poor chunking strategy

**Diagnostic helper:**

```python
def safe_info(arr, max_chunks=1000):
    """
    Safely print array info, avoiding slow .info_complete() on large arrays.
    """
    print(arr.info)

    n_chunks = arr.nchunks
    print(f"\nTotal chunks: {n_chunks:,}")

    if n_chunks > max_chunks:
        print(f"⚠️ Too many chunks for .info_complete() ({n_chunks:,} > {max_chunks:,})")
        print("Skipping detailed scan for performance")

        # Provide estimates instead
        print(f"\nEstimates:")
        print(f"  Uncompressed: {arr.nbytes / 1024**3:.2f} GB")
        print(f"  Chunk size: {np.prod(arr.chunks) * arr.dtype.itemsize / 1024**2:.2f} MB")
    else:
        print("\nDetailed info (scanning all chunks):")
        print(arr.info_complete())

# Usage
safe_info(arr)
```

