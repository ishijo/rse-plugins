---
name: compression-codecs
description: This skill should be used when the user asks to "configure zarr compression", "choose a compressor", "compare compression codecs", "use blosc with zarr", "add filters to zarr", or needs guidance on compression codec selection, Blosc configuration, filter pipelines, or optimizing storage efficiency for Zarr arrays.
---

# Compression Codecs for Zarr

Master compression strategies for Zarr arrays including codec selection, Blosc configuration, pre-compression filters, and the critical trade-offs between compression ratio, speed, and compatibility.

## Quick Reference Card

### Installation & Setup

See the plugin README for installation instructions.

### Codec Selection Table

| Codec | Speed | Ratio | Best For |
|-------|-------|-------|----------|
| **Blosc+LZ4** | Fastest | Good | Hot data, frequent access, speed priority |
| **Blosc+Zstd** (v2 default) | Fast | Excellent | General purpose, balanced performance |
| **Zstd** (v3 default) | Fast | Excellent | Zarr v3, broad compatibility |
| **Gzip** | Slow | Good | Universal compatibility, archival |
| **LZ4** | Very Fast | Moderate | Speed-critical applications |
| **LZMA** | Very Slow | Best | Maximum compression, cold storage |
| **BZ2** | Slow | Good | Legacy compatibility |
| **Zlib** | Moderate | Moderate | Wide compatibility |

**Decision tree:**
- **Fastest** → Blosc+LZ4 or standalone LZ4
- **Best ratio** → LZMA or Blosc+Zstd (level 9)
- **Universal** → Gzip (widest tool support)
- **Default** → Zstd (v3) or Blosc+Zstd (v2)
- **Integer deltas** → Delta filter + any codec
- **Limited float precision** → Quantize + Zstd

## When to Use This Skill

Use compression codec knowledge when:

- **Selecting compression** for new Zarr arrays
- **Optimizing storage size** for cloud or archival data
- **Balancing I/O performance** with compression overhead
- **Troubleshooting compression issues** (poor ratios, slow writes)
- **Migrating between codecs** or Zarr versions
- **Applying filters** for scientific data (Delta, Quantize)
- **Working with multi-process** applications (Blosc thread safety)

## Core Concepts

### 1. Blosc Configuration

Blosc is a **meta-compressor** that wraps other algorithms with multithreading and bit-shuffling optimizations.

**Complete Blosc configuration:**
```python
from numcodecs import Blosc

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE, blocksize=0)
arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), chunks=(1000, 1000), dtype="f4", compressor=compressor)
```

**Blosc compression algorithms (cname):**
- `'blosclz'` - Blosc's internal algorithm (fast)
- `'lz4'` - Very fast, moderate compression
- `'lz4hc'` - LZ4 high compression (slower than lz4, better ratio)
- `'snappy'` - Fast, moderate compression
- `'zlib'` - Moderate speed and ratio
- `'zstd'` - **Recommended** - fast with excellent compression

**Compression levels (clevel):**
```python
Blosc(cname='zstd', clevel=1)  # Fastest
Blosc(cname='zstd', clevel=3)  # Recommended balance
Blosc(cname='zstd', clevel=9)  # Maximum compression
```

**Shuffle modes:**
```python
Blosc(cname='zstd', clevel=3, shuffle=Blosc.NOSHUFFLE)    # No shuffling
Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)      # Byte-wise (default)
Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)   # Bit-wise (best for floats)
```

**Shuffle effectiveness test:**
```python
data = np.linspace(0, 100, 10000000).astype('f4')
arr_bitshuffle = zarr.array(data, chunks=(100000,), compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
print(f"Compressed: {arr_bitshuffle.nbytes_stored / 1024**2:.2f} MB")
```

### 2. ⚠️ CRITICAL: Blosc Thread Safety

**WARNING: Blosc multi-threading causes silent data corruption in multi-process environments!**

```python
import blosc
import zarr
from multiprocessing import Process

# CRITICAL: Disable Blosc internal threading for multi-process safety
blosc.use_threads = False

# This setting MUST be in every process that uses Blosc
# Silent data corruption will occur otherwise - no error messages!
```

**Why this matters:**
- Blosc uses internal multithreading for compression
- Multiple processes + Blosc threads = race conditions
- Corruption is **silent** - no exceptions, wrong data written
- Symptoms: inconsistent reads, checksum failures, corrupted values

**Safe multi-process pattern:**
```python
import blosc
from zarr import ProcessSynchronizer
from multiprocessing import Process

def worker(rank):
    blosc.use_threads = False  # CRITICAL in each process
    arr = zarr.open_array("data.zarr", mode="r+", synchronizer=ProcessSynchronizer("data.sync"))
    arr[rank*1000:(rank+1)*1000, :] = np.random.randn(1000, 1000).astype("f4")

blosc.use_threads = False
arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 1000), chunks=(1000, 1000), dtype="f4",
                       compressor=zarr.Blosc(cname='zstd', clevel=3), synchronizer=ProcessSynchronizer("data.sync"))
[Process(target=worker, args=(i,)).start() for i in range(10)]
```

**Thread control:**
```python
blosc.use_threads = False    # Disable (safest for multi-process)
blosc.set_nthreads(4)         # Or limit threads (single-process)
```

### 3. Standalone Codecs

Use standalone codecs when Blosc overhead isn't needed or for Zarr v3:

**Standalone codecs:**
```python
from numcodecs import Zstd, LZ4, GZip, LZMA, BZ2, Zlib

Zstd(level=3)         # Default for v3, level 1-22
LZ4(acceleration=1)   # Very fast, acceleration 1-10
GZip(level=6)         # Universal compatibility, level 1-9
LZMA(preset=6)        # Maximum compression, very slow
BZ2(level=9)          # Level 1-9
Zlib(level=6)         # Level 1-9
```

### 4. Disabling Compression

```python
arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), dtype="f4", compressor=None)  # v2
arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), dtype="f4", zarr_format=3, compressors=None)  # v3
```

**When to disable:** Already compressed data (JPEG, PNG), encrypted/random data, compression ratio < 1.1x, write speed critical.

### 5. Global Compression Override

```python
from numcodecs import Zstd, Blosc

zarr.storage.default_compressor = Zstd(level=1)  # Override for all new arrays
zarr.storage.default_compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)  # Restore default
```

### 6. Pre-Compression Filters

```python
from numcodecs import Delta, Quantize, FixedScaleOffset, PackBits, Categorize, Blosc

# Delta - stores differences (excellent for time-series/correlated data)
filters = [Delta(dtype='f4')]

# Quantize - round floats to N decimal places (lossy but highly compressible)
filters = [Quantize(digits=2, dtype='f4')]

# FixedScaleOffset - store floats as scaled integers
filters = [FixedScaleOffset(offset=0, scale=100, dtype='f4', astype='i2')]

# PackBits - run-length encoding (good for masks, boolean arrays)
filters = [PackBits()]

# Categorize - encode repeated values as integers
filters = [Categorize(labels=["low", "medium", "high"], dtype=object, astype='u1')]

arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), dtype="f4",
                       filters=filters, compressor=Blosc(cname='zstd', clevel=3))
```

### 7. Integrity Checks

```python
from numcodecs import CRC32, Blosc

filters = [CRC32()]  # Add checksum to detect corruption
arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), dtype="f4",
                       filters=filters, compressor=Blosc(cname='zstd', clevel=3))
# Detects corruption but adds ~4 bytes per chunk
```

**Adler32 checksum:**
```python
from numcodecs import Adler32

# Faster than CRC32, slightly less reliable
filters = [Adler32()]
```

### 8. Zarr v3 Codec Pipeline

Zarr v3 uses a different API for codecs:

**v3 codec configuration:**
```python
import zarr

# Zarr v3 uses 'codecs' or 'compressors', not 'compressor'
arr_v3 = zarr.open_array(
    "data_v3.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    zarr_format=3,
    compressors="zstd"  # String shorthand for Zstd(level=3)
)

# Or use explicit codec objects
from zarr.codecs import BloscCodec

arr_v3_blosc = zarr.open_array(
    "data_v3_blosc.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    zarr_format=3,
    compressors=BloscCodec(cname="zstd", clevel=3)
)
```

**v2 vs v3 comparison:**
```python
# Zarr v2
arr_v2 = zarr.open_array(
    "v2.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    compressor=zarr.Blosc(cname='zstd', clevel=3)  # 'compressor' parameter
)

# Zarr v3
arr_v3 = zarr.open_array(
    "v3.zarr",
    mode="w",
    shape=(10000, 10000),
    dtype="f4",
    zarr_format=3,
    compressors="zstd"  # 'compressors' parameter
)
```

### 9. Custom Codec Registration

Register custom codecs with numcodecs:

```python
from numcodecs.abc import Codec
from numcodecs.registry import register_codec
import numpy as np

class MyCustomCodec(Codec):
    codec_id = 'my_custom'

    def encode(self, buf):
        # Custom compression logic
        return buf  # Return compressed bytes

    def decode(self, buf, out=None):
        # Custom decompression logic
        return np.frombuffer(buf, dtype=out.dtype).reshape(out.shape)

# Register codec
register_codec(MyCustomCodec)

# Use in Zarr
from numcodecs import get_codec
compressor = get_codec({'id': 'my_custom'})
```

## Compression and Chunking

**Research finding (Nguyen et al. 2023):** Larger chunks achieve better compression ratios because there are more repeated patterns for codecs to exploit.

```python
import zarr
import numpy as np

# Same data, different chunk sizes
data = np.random.randn(10000, 10000).astype("f4")

# Small chunks
arr_small = zarr.array(data, chunks=(100, 100), compressor=zarr.Blosc(cname='zstd', clevel=3))

# Large chunks
arr_large = zarr.array(data, chunks=(1000, 1000), compressor=zarr.Blosc(cname='zstd', clevel=3))

ratio_small = arr_small.nbytes / arr_small.nbytes_stored
ratio_large = arr_large.nbytes / arr_large.nbytes_stored

print(f"Small chunks (100x100): {ratio_small:.2f}x compression")
print(f"Large chunks (1000x1000): {ratio_large:.2f}x compression")
# Larger chunks typically compress better
```

**Trade-off:** Larger chunks = better compression but less flexible access.

## Best Practices Checklist

### Codec Selection
- Use Blosc+Zstd or Zstd for general-purpose compression
- Use Blosc+LZ4 or LZ4 when speed is critical
- Use LZMA for archival/cold storage (maximum compression)
- Use Gzip for maximum compatibility across tools
- Disable compression for random/encrypted data

### Blosc Configuration
- **CRITICAL**: Set `blosc.use_threads = False` in multi-process code
- Use BITSHUFFLE for floating-point scientific data
- Use clevel=3 as default balance (speed vs ratio)
- Limit threads with `blosc.set_nthreads(n)` if needed

### Filters
- Apply Delta filter for time-series or spatially correlated data
- Use Quantize for data where precision can be sacrificed
- Test filter effectiveness before production deployment
- Document filter configuration in array metadata

### Performance
- Benchmark compression on representative data samples
- Monitor compression ratio: ratio < 1.2x may not be worth overhead
- Consider chunk size impact on compression (larger = better ratio)
- Test read/write speeds with different codecs for your workflow

### Compatibility
- Use Gzip or Zstd for maximum cross-tool compatibility
- Document codec requirements in dataset metadata
- Test codec availability in target environments
- Provide fallback options for legacy systems

## Resources and References

### Official Documentation
- **Zarr Compression**: https://zarr.readthedocs.io/en/stable/tutorial.html#compressors
- **numcodecs**: https://numcodecs.readthedocs.io/
- **Blosc**: https://www.blosc.org/

### Research
- **Nguyen et al. (2023)**: Larger chunks provide better compression ratios due to more repeated data patterns

### Related Libraries
- **Blosc2**: Next-generation Blosc (https://github.com/Blosc/python-blosc2)
- **Zstandard**: https://facebook.github.io/zstd/
- **LZ4**: https://lz4.github.io/lz4/

### Codec Benchmarks
- **Blosc benchmark suite**: https://github.com/Blosc/c-blosc/tree/master/bench
- **Squash benchmark**: https://quixdb.github.io/squash-benchmark/
