---
name: zarr-xarray-integration
description: |
  Use this skill when the user asks to "read zarr with xarray", "write xarray to zarr", "append data to zarr store", "use region writes", "configure zarr encoding in xarray", "use dask with zarr", "open zarr dataset", "distributed zarr writes", or needs guidance on xarray's Zarr backend, write modes, encoding control, Dask lazy loading, region writes for parallel filling, or cloud dataset access via xarray.
---

# Zarr-Xarray Integration

Using Zarr as a storage backend for Xarray datasets, covering all write modes (overwrite, append, region writes), encoding control, distributed writes with Dask, cloud storage integration, and consolidated metadata. For Xarray fundamentals, see the xarray-for-multidimensional-data skill in the scientific-domain-applications plugin.

## Quick Reference: Reading Zarr with Xarray

### Basic Read
```python
import xarray as xr

# Preferred: Explicit Zarr engine
ds = xr.open_zarr('path/to/dataset.zarr')

# Alternative: Explicit engine parameter
ds = xr.open_dataset('path/to/dataset.zarr', engine='zarr')
```

### Cloud Read (S3)
```python
import xarray as xr

# With consolidated metadata (recommended for cloud)
ds = xr.open_zarr('s3://bucket/dataset.zarr', consolidated=True)

# With storage options (authentication)
ds = xr.open_zarr(
    's3://bucket/dataset.zarr',
    consolidated=True,
    storage_options={'anon': False}
)
```

### Chunking Control
```python
# Option 1: Dask auto-chunking (for computation)
ds = xr.open_zarr('dataset.zarr', chunks='auto')

# Option 2: Preserve Zarr chunks exactly (for I/O optimization)
ds = xr.open_zarr('dataset.zarr', chunks={})

# Option 3: Custom chunks
ds = xr.open_zarr('dataset.zarr', chunks={'time': 10, 'lat': 90, 'lon': 180})
```

## Reading Zarr with Xarray: Details

### Chunks Parameter Behavior

| Parameter | Behavior | Use Case |
|-----------|----------|----------|
| `chunks='auto'` | Dask auto-chunks based on array size | General computation |
| `chunks={}` | Preserves exact Zarr chunk structure | I/O-bound workflows |
| `chunks={'dim': N}` | Custom chunks per dimension | Specific access patterns |
| `chunks=None` | Load entire arrays into memory | Small datasets only |

**Recommendation:**
- **Cloud access:** `chunks={}` (preserves optimized Zarr chunks)
- **Computation:** `chunks='auto'` (Dask optimizes for parallelism)
- **Custom:** Specify chunks to match your access pattern

### Consolidated Metadata (Cloud Performance)

See the cloud-storage-backends skill for complete metadata consolidation details. Always use `consolidated=True` for cloud storage to achieve 10-100x faster dataset opening (1 network request vs 100+).

### Decode Times and Missing Values

```python
import xarray as xr

# Decode CF time units automatically
ds = xr.open_zarr('dataset.zarr', decode_times=True)  # Default

# Disable time decoding (keep as integers)
ds = xr.open_zarr('dataset.zarr', decode_times=False)

# Custom fill value handling
ds = xr.open_zarr('dataset.zarr', mask_and_scale=True)  # Default: decode fill_value
```

## Writing Zarr with Xarray: All Modes

Xarray provides 6 write modes for different use cases.

### Mode Table

| Mode | Append Dim | Region | Behavior | Use Case |
|------|-----------|--------|----------|----------|
| `mode='w'` | - | - | Overwrite entire store | Initial write |
| `mode='a'` | - | - | Append new variables | Add variables to existing store |
| `mode='a'` | `'time'` | - | Append along dimension | Time-series updates |
| `mode='a'` | - | `'auto'` | Auto-detect region | Distributed parallel writes |
| `mode='a'` | - | `{'x': slice(10,20)}` | Update specific region | Partial updates |
| `compute=False` | - | - | Return delayed write task | Manual scheduling |

### Mode 1: Overwrite (`mode='w'`)

**When to use:** Initial dataset creation, complete replacement.

```python
import xarray as xr
import numpy as np

# Create dataset
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360)),
    'precipitation': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360))
})

# Write to Zarr (overwrites existing store)
ds.to_zarr('dataset.zarr', mode='w')
```

**⚠️ Warning:** `mode='w'` deletes existing store. Use with caution.

### Mode 2: Append Variables (`mode='a'`)

**When to use:** Adding new variables to existing store.

```python
import xarray as xr
import numpy as np

# Open existing store
ds_existing = xr.open_zarr('dataset.zarr')

# Create new variable
ds_new = xr.Dataset({
    'humidity': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360))
})

# Append new variable (mode='a')
ds_new.to_zarr('dataset.zarr', mode='a')

# Result: dataset.zarr now contains temperature, precipitation, AND humidity
```

**Must re-consolidate metadata:**
```python
import zarr
zarr.consolidate_metadata('dataset.zarr')
```

### Mode 3: Append Along Dimension (`append_dim='time'`)

**When to use:** Time-series updates, incremental data ingestion.

```python
import xarray as xr
import numpy as np

# Initial write (days 0-99)
ds_initial = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(100, 180, 360))
}, coords={'time': range(100)})
ds_initial.to_zarr('dataset.zarr', mode='w')

# Append new data (days 100-199)
ds_append = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(100, 180, 360))
}, coords={'time': range(100, 200)})
ds_append.to_zarr('dataset.zarr', mode='a', append_dim='time')

# Result: dataset.zarr now has 200 time steps
```

**Must re-consolidate metadata after append:**
```python
import zarr
zarr.consolidate_metadata('dataset.zarr')
```

### Mode 4: Region Write with Auto-Detection (`region='auto'`)

**When to use:** Distributed parallel writes (each worker writes different region).

```python
import xarray as xr
import numpy as np

# Worker 1: Writes time steps 0-99
ds_worker1 = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(100, 180, 360))
}, coords={'time': range(0, 100)})
ds_worker1.to_zarr('dataset.zarr', region='auto')

# Worker 2: Writes time steps 100-199 (in parallel)
ds_worker2 = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(100, 180, 360))
}, coords={'time': range(100, 200)})
ds_worker2.to_zarr('dataset.zarr', region='auto')

# Xarray auto-detects non-overlapping regions
```

**Critical: Target store must be initialized first:**
```python
# Initialize store with full shape
ds_template = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.empty((200, 180, 360)))
}, coords={'time': range(200)})
ds_template.to_zarr('dataset.zarr', mode='w', compute=False)
```

### Mode 5: Explicit Region Write

**When to use:** Updating specific spatial or temporal regions.

```python
import xarray as xr
import numpy as np

# Update specific time range (days 50-59)
ds_update = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(10, 180, 360))
}, coords={'time': range(50, 60)})

ds_update.to_zarr(
    'dataset.zarr',
    mode='a',
    region={'time': slice(50, 60)}  # Explicit region
)

# Update spatial subset (specific lat/lon box)
ds_spatial = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(365, 20, 40))
}, coords={
    'time': range(365),
    'lat': range(80, 100),
    'lon': range(100, 140)
})

ds_spatial.to_zarr(
    'dataset.zarr',
    mode='a',
    region={'lat': slice(80, 100), 'lon': slice(100, 140)}
)
```

### Mode 6: Delayed Write (`compute=False`)

**When to use:** Manual control over computation scheduling.

```python
import xarray as xr
import dask.array as da

# Create large Dask-backed dataset
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], da.random.random((365, 180, 360), chunks=(10, 90, 180)))
})

# Return delayed task (doesn't execute yet)
write_task = ds.to_zarr('dataset.zarr', mode='w', compute=False)

# Execute manually
write_task.compute()

# Or: persist to memory first, then write
ds_persisted = ds.persist()
ds_persisted.to_zarr('dataset.zarr', mode='w')
```

## Encoding Control: Compression and Chunking

Xarray allows fine-grained control over Zarr encoding (chunks, compression, dtypes).

### Encoding Precedence

```
Highest priority → Manual encoding dict
                ↓
                 Dask array chunks
                ↓
Lowest priority  → Zarr defaults
```

### Manual Encoding (Highest Priority)

```python
import xarray as xr
import zarr

ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], data)
})

# Explicit encoding
encoding = {
    'temperature': {
        'chunks': (10, 90, 180),  # Override Dask chunks
        'compressor': zarr.codecs.Blosc(cname='zstd', clevel=3, shuffle=zarr.codecs.BITSHUFFLE),
        'dtype': 'float32',
        'fill_value': -9999.0
    }
}

ds.to_zarr('dataset.zarr', mode='w', encoding=encoding)
```

### Dask Array Chunks (Medium Priority)

```python
import xarray as xr
import dask.array as da

# Dask chunks influence Zarr chunks (if no manual encoding)
data = da.random.random((365, 180, 360), chunks=(10, 90, 180))
ds = xr.Dataset({'temperature': (['time', 'lat', 'lon'], data)})

# Zarr inherits Dask chunks: (10, 90, 180)
ds.to_zarr('dataset.zarr', mode='w')
```

### Zarr Defaults (Lowest Priority)

```python
import xarray as xr
import numpy as np

# No Dask chunks, no manual encoding → Zarr uses defaults
data = np.random.randn(365, 180, 360)
ds = xr.Dataset({'temperature': (['time', 'lat', 'lon'], data)})

# Zarr auto-chunks (varies by version, typically full array or 1 chunk per dim)
ds.to_zarr('dataset.zarr', mode='w')
```

### Encoding All Variables

```python
import xarray as xr
import zarr

ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], temp_data),
    'precipitation': (['time', 'lat', 'lon'], precip_data)
})

# Encoding for all variables
encoding = {
    'temperature': {
        'chunks': (10, 90, 180),
        'compressor': zarr.codecs.Blosc(cname='zstd', clevel=3),
        'dtype': 'float32'
    },
    'precipitation': {
        'chunks': (10, 90, 180),
        'compressor': zarr.codecs.Blosc(cname='lz4', clevel=5),
        'dtype': 'float32'
    }
}

ds.to_zarr('dataset.zarr', mode='w', encoding=encoding)
```

## Distributed Parallel Writes with Dask

**Critical pattern for large datasets:** Initialize store → region writes → consolidate metadata.

### Step-by-Step Distributed Write

```python
import xarray as xr
import dask
import dask.array as da
import zarr

# Step 1: Create Dask-backed dataset
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], da.random.random((365, 180, 360), chunks=(10, 90, 180)))
})

# Step 2: Initialize Zarr store with full structure (compute=False)
ds.to_zarr('dataset.zarr', mode='w', compute=False)

# Step 3: Write in parallel with region='auto'
# Dask scheduler distributes chunk writes across workers
write_task = ds.to_zarr('dataset.zarr', region='auto', compute=False)
write_task.compute()

# Step 4: Consolidate metadata (CRITICAL for cloud)
zarr.consolidate_metadata('dataset.zarr')
```

### Distributed Workers Pattern

```python
from dask.distributed import Client
import xarray as xr
import dask.array as da

# Setup Dask cluster
client = Client()

# Create large Dask dataset
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], da.random.random((10000, 720, 1440), chunks=(10, 360, 720)))
})

# Initialize store
ds.to_zarr('large-dataset.zarr', mode='w', compute=False)

# Distributed write (Dask workers write chunks in parallel)
ds.to_zarr('large-dataset.zarr', region='auto')

# Consolidate metadata
import zarr
zarr.consolidate_metadata('large-dataset.zarr')

client.close()
```

## Cloud Storage Integration

### S3 with Storage Options

```python
import xarray as xr

# Read from S3
ds = xr.open_zarr(
    's3://bucket/dataset.zarr',
    consolidated=True,
    storage_options={
        'anon': False,  # Use credentials
        'client_kwargs': {'region_name': 'us-west-2'}
    }
)

# Write to S3
ds.to_zarr(
    's3://bucket/output.zarr',
    mode='w',
    storage_options={
        'anon': False,
        'client_kwargs': {'region_name': 'us-west-2'}
    }
)

# Consolidate metadata (required for cloud)
import zarr
zarr.consolidate_metadata('s3://bucket/output.zarr')
```

### GCS with Service Account

```python
import xarray as xr

# GCS with service account
ds = xr.open_zarr(
    'gs://bucket/dataset.zarr',
    consolidated=True,
    storage_options={
        'project': 'my-project',
        'token': '/path/to/service-account.json'
    }
)
```

### Azure Blob Storage

```python
import xarray as xr

# Azure with connection string
ds = xr.open_zarr(
    'az://container/dataset.zarr',
    consolidated=True,
    storage_options={
        'connection_string': 'DefaultEndpointsProtocol=https;...'
    }
)
```

### Explicit Store for Cloud

```python
import xarray as xr
import zarr
from fsspec import filesystem

# Create cloud store explicitly
fs = filesystem('s3', anon=False)
store = zarr.storage.FsspecStore('bucket/dataset.zarr', fs=fs)

# Read with Xarray
ds = xr.open_zarr(store, consolidated=True)

# Write with Xarray
ds.to_zarr(store, mode='w')
zarr.consolidate_metadata(store)
```

## Dask Lazy Loading and Computation

### Lazy Loading Pattern

```python
import xarray as xr

# Open with Dask chunks (lazy loading)
ds = xr.open_zarr('large-dataset.zarr', chunks='auto')

# No data loaded yet
temp = ds['temperature']  # Lazy Dask array

# Compute subset (loads only necessary chunks)
subset = temp.isel(time=0).compute()  # Loads day 0 only

# Avoid .values on large arrays (loads entire array into memory)
# BAD: data = temp.values  # ❌ Loads all 100 GB into RAM
# GOOD: data = temp.compute()  # ✓ Still large, but explicit
```

### Load vs Compute

```python
import xarray as xr

ds = xr.open_zarr('dataset.zarr', chunks='auto')

# .compute() — Returns computed numpy array
data = ds['temperature'].compute()  # Returns numpy array

# .load() — Loads data into Dataset/DataArray (in-place)
ds.load()  # Now ds['temperature'].values is in memory
```

### Partial Loads (Efficient)

```python
import xarray as xr

ds = xr.open_zarr('large-dataset.zarr', chunks='auto')

# Load only specific time range
subset = ds.sel(time=slice('2024-01-01', '2024-01-31'))
subset.load()  # Loads only January data

# Load only specific spatial region
bbox = ds.sel(lat=slice(-90, -60), lon=slice(0, 40))
bbox.compute()
```

## Best Practices

### Writing to Cloud: Complete Workflow

```python
import xarray as xr
import zarr
import dask.array as da

# 1. Create Dask-backed dataset
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], da.random.random((365, 180, 360), chunks=(10, 90, 180)))
})

# 2. Configure encoding
encoding = {
    'temperature': {
        'compressor': zarr.codecs.Blosc(cname='zstd', clevel=3, shuffle=zarr.codecs.BITSHUFFLE),
        'chunks': (10, 90, 180)
    }
}

# 3. Write to cloud with storage_options
ds.to_zarr(
    's3://bucket/dataset.zarr',
    mode='w',
    encoding=encoding,
    storage_options={'anon': False}
)

# 4. Consolidate metadata (CRITICAL for cloud performance)
zarr.consolidate_metadata('s3://bucket/dataset.zarr')
```

### Appending to Existing Store: Complete Workflow

```python
import xarray as xr
import zarr

# 1. Append new time steps
ds_new = xr.Dataset({'temperature': (['time', 'lat', 'lon'], new_data)})
ds_new.to_zarr('dataset.zarr', mode='a', append_dim='time')

# 2. Re-consolidate metadata (REQUIRED after structural changes)
zarr.consolidate_metadata('dataset.zarr')
```

### Distributed Write: Complete Workflow

```python
import xarray as xr
import zarr
import dask.array as da

# 1. Create template with full shape
ds_template = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], da.zeros((1000, 180, 360), chunks=(10, 90, 180)))
})
ds_template.to_zarr('dataset.zarr', mode='w', compute=False)

# 2. Workers write regions in parallel
# Worker 1:
ds_worker1 = xr.Dataset({'temperature': (['time', 'lat', 'lon'], data1)})
ds_worker1.to_zarr('dataset.zarr', region={'time': slice(0, 100)})

# Worker 2:
ds_worker2 = xr.Dataset({'temperature': (['time', 'lat', 'lon'], data2)})
ds_worker2.to_zarr('dataset.zarr', region={'time': slice(100, 200)})

# 3. Consolidate metadata
zarr.consolidate_metadata('dataset.zarr')
```

## Common Pitfalls

### ❌ Forgetting to Consolidate Metadata

```python
# BAD: No consolidation (100+ cloud requests on read)
ds.to_zarr('s3://bucket/dataset.zarr', mode='w')

# GOOD: Always consolidate for cloud
ds.to_zarr('s3://bucket/dataset.zarr', mode='w')
zarr.consolidate_metadata('s3://bucket/dataset.zarr')
```

### ❌ Using .values on Large Dask Arrays

```python
# BAD: Loads entire 100 GB array into memory
ds = xr.open_zarr('large-dataset.zarr', chunks='auto')
data = ds['temperature'].values  # ❌ OOM error

# GOOD: Use .compute() explicitly or work with subsets
subset = ds['temperature'].isel(time=slice(0, 10))
data = subset.compute()  # ✓ Only loads 10 time steps
```

### ❌ Not Re-Consolidating After Append

```python
# BAD: Append without re-consolidating
ds_new.to_zarr('dataset.zarr', mode='a', append_dim='time')
# Old .zmetadata is now stale

# GOOD: Re-consolidate after structural changes
ds_new.to_zarr('dataset.zarr', mode='a', append_dim='time')
zarr.consolidate_metadata('dataset.zarr')
```

### ❌ Wrong chunks Parameter

```python
# BAD: chunks='auto' for I/O-heavy workflow (re-chunks inefficiently)
ds = xr.open_zarr('dataset.zarr', chunks='auto')

# GOOD: chunks={} preserves optimal Zarr chunks
ds = xr.open_zarr('dataset.zarr', chunks={})
```

## Related Skills and References

**Cross-references:**
- **xarray-for-multidimensional-data** (scientific-domain-applications) — Xarray fundamentals
- **zarr-fundamentals** — chunking strategies, array creation, Zarr basics
- **cloud-storage-backends** — S3/GCS/Azure configuration, storage_options
- **compression-codecs** — choosing compressors for encoding

**Reference files in this skill:**
- `references/PATTERNS.md` — 6+ Xarray-Zarr workflow patterns
- `references/EXAMPLES.md` — 4+ end-to-end examples (cloud writes, distributed, append)
- `references/COMMON_ISSUES.md` — 5+ troubleshooting guides

---

**This skill covers Zarr-Xarray integration. For Xarray basics, see xarray-for-multidimensional-data skill. For cloud storage details, see cloud-storage-backends skill.**
