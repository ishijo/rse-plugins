---
name: zarr-fundamentals
description: This skill should be used when the user asks to "create a zarr array", "open a zarr store", "read zarr data", "write zarr arrays", "manage zarr groups", "set zarr metadata", "use zarr indexing", "understand zarr format", "work with zarr v3", or needs guidance on Zarr array storage format, chunking strategies, compression configuration, metadata management, or transitioning between Zarr v2 and v3.
---

# Zarr Fundamentals

Master Zarr, the cloud-optimized array storage format for chunked, compressed N-dimensional arrays, with support for both v2 and v3 specifications. This skill covers array creation, chunking strategies, compression configuration, metadata management, and efficient data pipelines for scientific computing.

## Quick Reference Card

### Installation & Setup

See the plugin README for installation instructions.

### Essential Zarr Concepts (v2 and v3 Side-by-Side)

```python
import zarr
import numpy as np

# Array creation - Zarr v2 (default until explicitly set)
z_v2 = zarr.open_array(
    "data_v2.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    compressor=zarr.storage.default_compressor  # Blosc by default
)

# Array creation - Zarr v3 (explicitly specify)
z_v3 = zarr.open_array(
    "data_v3.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    zarr_format=3,  # Explicit v3
    compressors="zstd"  # Note: compressors, not compressor
)

# Write data (same for both versions)
z_v2[:] = np.random.randn(10000, 10000)
z_v3[:] = np.random.randn(10000, 10000)

# Read data (same for both versions)
data_v2 = z_v2[:]
data_v3 = z_v3[5000:6000, :]  # Partial read
```

### Quick Decision Tree

```
Need to store large N-dimensional arrays?
├─ YES → Zarr is designed for this
└─ NO → Consider simpler formats (CSV, JSON)

Data larger than memory?
├─ YES → Use chunking (Zarr's core feature)
└─ NO → Zarr still useful for compression

Need cloud storage (S3, GCS, Azure)?
├─ YES → Zarr is cloud-native
└─ NO → Local filesystem works fine

Concurrent writes required?
├─ YES → Use synchronizers (Thread/Process)
└─ NO → Default behavior is safe

Which Zarr version to use?
├─ Existing v2 data → Stay with v2 for compatibility
├─ Need sharding → Use v3
├─ Need async I/O → Use v3
├─ Python 3.8-3.10 → Use v2
└─ Starting fresh + Python 3.11+ → Consider v3

Working with Xarray?
├─ YES → Zarr integrates seamlessly as storage backend
└─ NO → Use Zarr directly via zarr-python
```

## When to Use This Skill

Use Zarr when working with:

- **Large-scale N-dimensional arrays** that don't fit in memory
- **Cloud-native workflows** requiring parallel I/O to S3, GCS, or Azure
- **Compressed scientific data** with flexible codec options
- **Chunked data access** where only portions of arrays are read
- **Distributed computing** with Dask or other frameworks
- **Multidimensional time series** with spatial and temporal dimensions
- **Data migration** from HDF5 or NetCDF to cloud-optimized formats
- **Collaborative workflows** where multiple processes write concurrently

## Core Concepts

### 1. Zarr v2 vs v3: Key Differences

Zarr has two specification versions with important differences:

| Feature | Zarr v2 | Zarr v3 |
|---------|---------|---------|
| **Metadata Files** | Separate `.zarray`, `.zattrs`, `.zgroup` | Unified `zarr.json` |
| **Default Compressor** | Blosc (multithreaded) | Zstd (single-threaded) |
| **Sharding** | Not supported | Native sharding support |
| **Async I/O** | Synchronous only | Async-capable stores |
| **Python Requirement** | Python 3.8+ | Python 3.11+ |
| **zarr_format param** | `zarr_format=2` (default) | `zarr_format=3` (explicit) |
| **Consolidated Metadata** | `consolidate_metadata()` | Built-in via `zarr.json` |
| **Store API** | `MutableMapping` interface | Abstract `Store` class |
| **Codec Configuration** | `compressor` parameter | `codecs` or `compressors` parameter |

**When to use each:**
- **v2**: Existing workflows, Python 3.8-3.10, broad ecosystem compatibility
- **v3**: New projects, sharding required, async I/O, Python 3.11+

**Note**: The zarr-python library supports both formats. Specify `zarr_format=3` to use v3.

### 2. Array Creation Functions

Zarr provides multiple functions for array creation, each with specific use cases:

**Create array with explicit parameters:**
```python
import zarr

# Most explicit - full control
arr = zarr.create_array(
    store="data.zarr",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="float32",
    fill_value=0.0,
    compressor=zarr.Blosc(cname="zstd", clevel=3),  # v2
    # compressors="zstd",  # v3 alternative
    overwrite=True
)
```

**Convenience functions (like NumPy):**
```python
z_zeros = zarr.zeros((5000, 5000), chunks=(500, 500), dtype="i4", store="zeros.zarr")
z_ones = zarr.ones((1000, 1000, 100), chunks=(100, 100, 10), dtype="f8", store="ones.zarr")
z_full = zarr.full((2000, 3000), fill_value=42.0, chunks=(200, 300), dtype="f4", store="full.zarr")
z_empty = zarr.empty((10000, 10000), chunks=(1000, 1000), dtype="f4", store="empty.zarr")
```

**Open existing or create new:**
```python
z = zarr.open_array("data.zarr", mode="a", shape=(10000, 10000), chunks=(1000, 1000), dtype="f4")
# Modes: 'r' (read), 'r+' (read/write), 'a' (read/write, create if needed), 'w' (overwrite), 'w-' (create, fail if exists)
```

**Key parameters explained:**

- **shape**: Tuple defining array dimensions `(dim0, dim1, ...)`
- **chunks**: Chunk size for each dimension `(chunk0, chunk1, ...)` - CRITICAL for performance
- **dtype**: NumPy data type (`'f4'`, `'i8'`, `'bool'`, etc.)
- **fill_value**: Default value for uninitialized elements
- **compressor** (v2) / **compressors** (v3): Compression codec configuration
- **shards** (v3 only): Shard size `(shard0, shard1, ...)` - must be multiples of chunks
- **zarr_format**: `2` (default) or `3` (explicit)

### 3. Group Management

Groups organize arrays hierarchically, like directories in a filesystem:

**Create and navigate groups:**
```python
root = zarr.open_group("data.zarr", mode="w")
observations = root.create_group("observations")
observations.create_array("temperature", shape=(365, 100, 100), chunks=(30, 10, 10), dtype="f4")
observations.create_array("humidity", shape=(365, 100, 100), chunks=(30, 10, 10), dtype="f4")

station_a = observations.create_group("station_a")
station_a.create_array("data", shape=(1000,), dtype="f4")

temp = root["observations/temperature"]  # Access via path
print(root.tree())  # Visualize hierarchy
```

**Group tree output example:**
```
/
 ├── observations
 │   ├── temperature (365, 100, 100) float32
 │   ├── humidity (365, 100, 100) float32
 │   └── station_a
 │       └── data (1000,) float32
 └── models
```

### 4. Metadata and Attributes

Zarr supports custom metadata via attributes, following CF conventions for scientific data:

**CF Convention metadata:**
```python
import zarr
import numpy as np

# Create array
arr = zarr.open_array(
    "temperature.zarr",
    mode="w",
    shape=(365, 180, 360),
    chunks=(30, 18, 36),
    dtype="f4"
)

# Set CF-compliant metadata
arr.attrs["long_name"] = "Air Temperature"
arr.attrs["units"] = "Celsius"
arr.attrs["standard_name"] = "air_temperature"
arr.attrs["coordinates"] = "time lat lon"

# Additional metadata
arr.attrs["_FillValue"] = -999.0
arr.attrs["valid_range"] = [-100.0, 60.0]
arr.attrs["source"] = "Climate Model v3.2"
arr.attrs["history"] = "2024-04-06: Created from raw model output"

# Read metadata
print(arr.attrs["long_name"])  # Air Temperature
print(arr.attrs["units"])  # Celsius

# Group metadata
group = zarr.open_group("data.zarr", mode="w")
group.attrs["project"] = "Climate Analysis 2024"
group.attrs["institution"] = "Research Center"
group.attrs["contact"] = "researcher@example.org"

# Access all attributes as dict
print(dict(arr.attrs))
```

**Structured metadata:**
```python
# Complex nested metadata
arr.attrs["processing"] = {
    "method": "kriging",
    "parameters": {"variogram": "spherical", "range": 100},
    "timestamp": "2024-04-06T12:00:00Z"
}

# Array metadata
arr.attrs["dimensions"] = {
    "time": {"long_name": "Time", "units": "days since 2024-01-01"},
    "lat": {"long_name": "Latitude", "units": "degrees_north"},
    "lon": {"long_name": "Longitude", "units": "degrees_east"}
}
```

### 5. Indexing Modes

Zarr supports 6 indexing modes for flexible data access:

#### Mode 1: Basic Slicing

```python
import zarr
import numpy as np

arr = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype="f4")

# Single element
val = arr[0, 0]

# Slices
row = arr[0, :]  # First row
col = arr[:, 0]  # First column
block = arr[100:200, 500:600]  # Sub-array

# Strided slices
every_10th = arr[::10, ::10]

# Negative indices
last_row = arr[-1, :]
```

#### Mode 2: Coordinate Indexing (vindex)
```python
# Select specific coordinates (non-contiguous)
indices = [0, 5, 10, 99, 500]
selected = arr.vindex[indices, :]

# 2D coordinate selection
row_indices = [0, 1, 2]
col_indices = [10, 20, 30]
selected = arr.vindex[row_indices, col_indices]  # Shape: (3,)
```

#### Mode 3: Mask Indexing (vindex with boolean)
```python
# Boolean mask - keep elements where mask is True
mask = np.random.rand(10000) > 0.5
selected = arr.vindex[mask, 0]  # Select rows where mask is True

# Condition-based selection
data = arr[:]
warm_pixels = arr.vindex[data > 20.0]  # All elements > 20
```

#### Mode 4: Orthogonal Indexing (oindex)
```python
# Outer product indexing - all combinations
rows = [0, 5, 10]
cols = [2, 7, 15, 100]
selected = arr.oindex[rows, cols]  # Shape: (3, 4) - all combinations

# Equivalent to:
# result[i, j] = arr[rows[i], cols[j]]
```

#### Mode 5: Block Indexing (blocks)
```python
# Access chunks directly as blocks
arr = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype="f4")

# Get first block (chunk)
block = arr.blocks[0, 0]  # Returns (1000, 1000) array

# Iterate over blocks
for i in range(arr.nchunks[0]):
    for j in range(arr.nchunks[1]):
        block = arr.blocks[i, j]
        # Process block
        arr.blocks[i, j] = block * 2.0
```

#### Mode 6: Structured Field Indexing
```python
# For structured/record arrays
dtype = np.dtype([
    ("time", "f8"),
    ("temperature", "f4"),
    ("pressure", "f4")
])

arr = zarr.zeros((10000,), chunks=(1000,), dtype=dtype)

# Access by field name
temps = arr["temperature"]
pressures = arr["pressure"]

# Assign to fields
arr["temperature"][:] = np.random.randn(10000) * 10 + 15
arr["pressure"][:] = np.random.randn(10000) * 5 + 1013
```

### 6. Data Types

Zarr supports all NumPy data types plus specialized types for variable-length data:

#### Standard NumPy dtypes
```python
zarr.zeros((100,), dtype="f4")    # float32, f8=float64
zarr.zeros((100,), dtype="i4")    # int32, i8=int64, u2=uint16
zarr.zeros((100,), dtype="bool")  # boolean
zarr.zeros((100,), dtype="c8")    # complex64, c16=complex128
```

#### Variable-length types
```python
from numcodecs import VLenUTF8, VLenBytes

arr = zarr.open_array("strings.zarr", mode="w", shape=(1000,), dtype=object, object_codec=VLenUTF8())
arr[:] = ["short", "a much longer string", "medium"]

arr_bytes = zarr.open_array("bytes.zarr", mode="w", shape=(1000,), dtype=object, object_codec=VLenBytes())
```

#### Structured/record arrays
```python
dtype = np.dtype([("name", "U20"), ("age", "i4"), ("height", "f4"), ("active", "bool")])
arr = zarr.zeros((10000,), chunks=(1000,), dtype=dtype)
arr[0] = ("Alice", 30, 165.5, True)
```

#### Datetime types
```python
arr_time = zarr.zeros((365,), dtype="datetime64[D]")
arr_time[:] = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]")
arr_delta = zarr.zeros((100,), dtype="timedelta64[s]")
```

#### Object codecs for complex types
```python
from numcodecs import JSON, MsgPack, Pickle

arr = zarr.open_array("json.zarr", mode="w", shape=(100,), dtype=object, object_codec=JSON())
arr[0] = {"key": "value", "number": 42, "list": [1, 2, 3]}
# Also available: MsgPack() (more efficient), Pickle() (not recommended for long-term storage)
```

#### Ragged arrays (variable-length arrays)
```python
from numcodecs import VLenArray

arr = zarr.open_array("ragged.zarr", mode="w", shape=(1000,), dtype=object, object_codec=VLenArray(dtype="f4"))
arr[0] = np.array([1.0, 2.0, 3.0], dtype="f4")      # Length 3
arr[1] = np.array([4.0, 5.0], dtype="f4")           # Length 2
arr[2] = np.array([6.0, 7.0, 8.0, 9.0], dtype="f4") # Length 4
```

#### Categorical data
```python
from numcodecs import Categorize

categories = ["low", "medium", "high", "very_high"]
arr = zarr.open_array("categorical.zarr", mode="w", shape=(10000,), chunks=(1000,), dtype=object,
                       object_codec=Categorize(labels=categories, dtype="u1"))
arr[:100] = "low"
arr[100:500] = "medium"
# Stores as integers 0,1,2,3 internally; reads back as category labels
```

### 7. Concurrency and Synchronization

Zarr supports concurrent access with proper synchronization:

**Thread-safe access:**
```python
from zarr import ThreadSynchronizer
import threading

arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), chunks=(1000, 1000),
                       synchronizer=ThreadSynchronizer())

def write_chunk(arr, i):
    arr[i*1000:(i+1)*1000, :] = np.random.randn(1000, 10000)

threads = [threading.Thread(target=write_chunk, args=(arr, i)) for i in range(10)]
[t.start() for t in threads]
[t.join() for t in threads]
```

**Process-safe access:**
```python
from zarr import ProcessSynchronizer
from multiprocessing import Process

arr = zarr.open_array("data.zarr", mode="w", shape=(10000, 10000), chunks=(1000, 1000),
                       synchronizer=ProcessSynchronizer("data.sync"))

def write_chunk(i):
    arr = zarr.open_array("data.zarr", mode="r+", synchronizer=ProcessSynchronizer("data.sync"))
    arr[i*1000:(i+1)*1000, :] = np.random.randn(1000, 10000)

processes = [Process(target=write_chunk, args=(i,)) for i in range(10)]
[p.start() for p in processes]
[p.join() for p in processes]
```

### 8. Sharding (Zarr v3)

Sharding reduces metadata overhead by grouping multiple chunks into shards:

**When to use sharding:**
- Very large arrays with millions of small chunks
- Cloud storage where metadata operations are slow/expensive
- Need to reduce number of objects in object storage

**Sharding configuration:**
```python
arr = zarr.open_array("data_v3.zarr", mode="w", shape=(100000, 100000, 100),
                       chunks=(100, 100, 100), shards=(1000, 1000, 100), dtype="f4", zarr_format=3)
# Each shard contains 10×10×1 = 100 chunks
# Shards = (chunks[0] * 10, chunks[1] * 10, chunks[2]) is typical pattern
```

## Chunking Strategy and Performance

Chunking is the most critical parameter for Zarr performance. Based on research (Nguyen et al. 2023):

**Key findings:**
- **Chunk size impacts both performance AND memory** - entire chunks must be read into memory
- **Access pattern determines optimal chunking** - time-series vs spatial access require opposite strategies
- **Optimal chunk size: ~1-3 MB uncompressed** - balances I/O overhead and memory usage
- **Larger chunks = better compression** - more data patterns for codecs to exploit
- **Smaller chunks = more flexible access** - but increased overhead

**Chunking guidelines:**

```python
# Time-series access → larger time chunks, smaller spatial
arr_ts = zarr.create_array("timeseries.zarr", shape=(10000, 1000, 1000), chunks=(1000, 100, 100), dtype="f4")

# Spatial/map access → smaller time chunks, larger spatial
arr_sp = zarr.create_array("maps.zarr", shape=(10000, 1000, 1000), chunks=(10, 500, 500), dtype="f4")

# Balanced access (both patterns)
arr_bal = zarr.create_array("balanced.zarr", shape=(10000, 1000, 1000), chunks=(48, 100, 100), dtype="f4")
```

**Estimate chunk size:**
```python
import numpy as np

shape = (10000, 1000, 1000)
chunks = (100, 100, 100)
dtype = np.float32

# Calculate uncompressed chunk size
chunk_bytes = np.prod(chunks) * np.dtype(dtype).itemsize
chunk_mb = chunk_bytes / (1024**2)

print(f"Chunk size: {chunk_mb:.2f} MB")  # Aim for 1-3 MB
```

## Patterns

See [references/PATTERNS.md](references/PATTERNS.md) for detailed patterns including:
- Array creation and initialization
- Reading and writing data
- Group hierarchies and organization
- Metadata management
- Concurrent access patterns
- Format conversion (v2 ↔ v3)

## Real-World Examples

See [references/EXAMPLES.md](references/EXAMPLES.md) for complete examples including:
- Creating cloud-native scientific datasets
- Migrating from HDF5 to Zarr
- Concurrent writes with Dask
- Time-series data with optimal chunking

## Common Issues and Solutions

See [references/COMMON_ISSUES.md](references/COMMON_ISSUES.md) for solutions to:
- Performance issues with chunking
- Compression codec selection
- Concurrent access errors
- Format compatibility (v2/v3)
- Cloud storage configuration

## Best Practices Checklist

### Array Design
- Choose chunk size based on access patterns (1-3 MB uncompressed target)
- Use meaningful metadata (long_name, units, standard_name)
- Select appropriate data types (float32 vs float64)
- Consider sharding for arrays with millions of chunks (v3)
- Document chunk rationale in array attributes

### Performance
- Align chunks with access patterns (time-series vs spatial)
- Enable compression unless data is random/incompressible
- Use appropriate compressor (see compression-codecs skill)
- Consolidate metadata for cloud access (v2: `consolidate_metadata()`)
- Test chunking strategy with realistic access patterns

### Concurrency
- Use ThreadSynchronizer for multithreaded access
- Use ProcessSynchronizer with lock file for multiprocess access
- Ensure processes open arrays with same synchronizer
- Avoid simultaneous writes to same chunks

### Format Selection
- Use v2 for maximum compatibility and Python 3.8-3.10
- Use v3 for sharding, async I/O, Python 3.11+
- Document zarr_format in project requirements
- Test migration path before committing to v3

### Code Quality
- Explicitly specify zarr_format for clarity
- Include units and metadata in attrs
- Use context managers where applicable
- Add type hints for array parameters
- Document expected dimensions and dtypes

## Resources and References

### Official Documentation
- **Zarr Specification**: https://zarr.dev/
- **zarr-python Documentation**: https://zarr.readthedocs.io/
- **Zarr v3 Specification**: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
- **API Reference**: https://zarr.readthedocs.io/en/stable/api.html

### Research
- **Nguyen et al. (2023)** - Chunking optimization for time-series and spatial access patterns

### Related Libraries
- **numcodecs**: https://numcodecs.readthedocs.io/ (compression codecs)
- **fsspec**: https://filesystem-spec.readthedocs.io/ (filesystem abstraction)
- **Xarray**: https://docs.xarray.dev/ (labeled arrays with Zarr backend)
- **Dask**: https://docs.dask.org/ (parallel computing)

### Community
- **Zarr GitHub Discussions**: https://github.com/zarr-developers/zarr-python/discussions
- **Pangeo**: https://pangeo.io/ (cloud-native geoscience)
