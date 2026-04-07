# Common Issues and Solutions

Troubleshooting guide for Xarray-Zarr integration workflows.

## Issue 1: Missing Consolidated Metadata After Append

**Symptom:**
```python
ds.to_zarr('data.zarr', mode='a', append_dim='time')

# Later: Cloud read is VERY slow (100+ network requests)
ds_read = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)
# WARNING: Could not read consolidated metadata, reading non-consolidated metadata
```

**Cause:**

After `mode='a'` (append), Xarray does not automatically re-consolidate metadata. Cloud reads fetch each `.zarray`/`.zattrs` file individually (catastrophic for performance).

**Solution:**

Always consolidate metadata after append operations:

```python
import xarray as xr
import zarr

# Append data
ds_new.to_zarr('data.zarr', mode='a', append_dim='time')

# CRITICAL: Re-consolidate
zarr.consolidate_metadata('data.zarr')

# Now cloud reads are fast (1 request)
ds_read = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)
```

**Best practice:**

Wrap append operations in a helper function:

```python
def append_and_consolidate(ds, zarr_path, append_dim='time'):
    """Append data and re-consolidate metadata."""
    ds.to_zarr(zarr_path, mode='a', append_dim=append_dim)
    zarr.consolidate_metadata(zarr_path)
```

**Performance impact:**

- Without consolidation: 50-500+ requests (5-30 seconds)
- With consolidation: 1-2 requests (<1 second)

---

## Issue 2: Chunk Mismatch Between Xarray and Zarr

**Symptom:**
```python
# Write with Xarray chunking
ds.to_zarr('data.zarr', mode='w')

# Read back - chunks don't match expectations
ds_read = xr.open_zarr('data.zarr', chunks='auto')
print(ds_read['temperature'].chunks)
# Chunks: ((365,), (180,), (360,))  # Wrong! Expected (30, 90, 180)
```

**Cause:**

Xarray uses default chunking during `to_zarr()` if encoding not specified. The stored Zarr chunks may differ from what you expect.

**Solution:**

**Always specify encoding explicitly:**

```python
import zarr

# Define exact chunks
encoding = {
    'temperature': {
        'chunks': (30, 90, 180),  # 30 days, 90° lat, 180° lon
        'compressor': zarr.Blosc(cname='zstd', clevel=3)
    }
}

# Write with encoding
ds.to_zarr('data.zarr', mode='w', encoding=encoding)

# Read with preserved chunks
ds_read = xr.open_zarr('data.zarr', chunks={})  # Empty dict = preserve Zarr chunks
print(ds_read['temperature'].chunks)
# Chunks: ((30, 30, ...), (90, 90), (180, 180))  # Correct!
```

**Key parameters:**

- `chunks={}` → Preserve Zarr chunk structure (recommended for cloud)
- `chunks='auto'` → Let Dask auto-chunk (may differ from Zarr)
- `chunks=None` → Load entire array into memory (dangerous for large datasets)

**Verification:**

```python
import zarr

# Inspect stored chunks directly
z = zarr.open('data.zarr/temperature', mode='r')
print(f"Zarr chunks: {z.chunks}")

# Compare with Xarray
ds = xr.open_zarr('data.zarr', chunks={})
print(f"Xarray chunks: {ds['temperature'].chunks}")
```

---

## Issue 3: Encoding Not Applied During Append

**Symptom:**
```python
# Initial write with compression
encoding = {'temperature': {'compressor': zarr.Blosc(cname='zstd', clevel=3)}}
ds.to_zarr('data.zarr', mode='w', encoding=encoding)

# Append new data
ds_new.to_zarr('data.zarr', mode='a', append_dim='time')

# New chunks are UNCOMPRESSED!
z = zarr.open('data.zarr/temperature')
print(z.chunks[365])  # Larger than expected
```

**Cause:**

Xarray ignores `encoding` parameter in `mode='a'` - new data inherits Zarr store settings, but if settings aren't already in the store, data may be uncompressed.

**Solution:**

Encoding is **automatically inherited** from the existing Zarr store. No action needed if initial write had encoding.

**Verification:**

```python
import zarr

# Check compressor on existing array
z = zarr.open('data.zarr/temperature', mode='r')
print(f"Compressor: {z.compressor}")
# Output: Blosc(cname='zstd', clevel=3, shuffle=SHUFFLE, blocksize=0)

# Append (compressor inherited automatically)
ds_new.to_zarr('data.zarr', mode='a', append_dim='time')

# New chunks use same compressor
print(f"Compressor after append: {z.compressor}")  # Same
```

**If you need to change encoding:**

Can't change encoding during append. Options:
1. **Rechunk entire dataset** (slow, expensive)
2. **Create new store** with new encoding, copy data
3. **Accept inherited encoding** (recommended)

---

## Issue 4: Region Parameter Errors (Misaligned Regions)

**Symptom:**
```python
# Try to update specific time slice
ds_update.to_zarr('data.zarr', mode='a', region={'time': slice(50, 60)})

# Error: ValueError: region slice (50, 60) exceeds array shape (100,) along dimension 'time'
```

**Cause:**

Region slices must align with actual array indices, not coordinate values.

**Solutions:**

**Option 1: Use integer indices (not coordinate values)**

```python
import xarray as xr

ds = xr.open_zarr('data.zarr')

# WRONG: Using coordinate values
# ds_update.to_zarr('data.zarr', mode='a', region={'time': slice('2024-01-01', '2024-01-10')})

# CORRECT: Using integer indices
ds_update.to_zarr('data.zarr', mode='a', region={'time': slice(0, 10)})
```

**Option 2: Calculate indices from coordinates**

```python
import numpy as np

ds = xr.open_zarr('data.zarr')

# Find indices for coordinate range
target_start = np.datetime64('2024-01-01')
target_end = np.datetime64('2024-01-10')

start_idx = np.where(ds['time'].values == target_start)[0][0]
end_idx = np.where(ds['time'].values == target_end)[0][0] + 1

# Use calculated indices
ds_update.to_zarr(
    'data.zarr',
    mode='a',
    region={'time': slice(start_idx, end_idx)}
)
```

**Option 3: Helper function for coordinate-based regions**

```python
def zarr_update_by_coords(ds_update, zarr_path, dim, coord_start, coord_end):
    """Update Zarr region using coordinate values instead of indices."""
    import numpy as np
    import xarray as xr

    ds_existing = xr.open_zarr(zarr_path)

    # Find indices
    start_idx = np.where(ds_existing[dim].values == coord_start)[0][0]
    end_idx = np.where(ds_existing[dim].values == coord_end)[0][0] + 1

    # Update region
    ds_update.to_zarr(
        zarr_path,
        mode='a',
        region={dim: slice(start_idx, end_idx)}
    )

# Usage
zarr_update_by_coords(
    ds_update,
    'data.zarr',
    dim='time',
    coord_start=np.datetime64('2024-01-01'),
    coord_end=np.datetime64('2024-01-10')
)
```

**Dimension alignment check:**

```python
# Ensure update dataset matches region shape
ds_existing = xr.open_zarr('data.zarr')
region_slice = slice(50, 60)

# Check shape compatibility
region_shape = region_slice.stop - region_slice.start
update_shape = ds_update.sizes['time']

assert region_shape == update_shape, \
    f"Region shape ({region_shape}) != update shape ({update_shape})"
```

---

## Issue 5: S3 Credentials and Configuration Errors

**Symptom:**
```python
ds = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)

# Error: botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Cause:**

Xarray uses `s3fs` for S3 access, which requires AWS credentials configuration.

**Solutions:**

**Option 1: Environment variables**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

```python
import xarray as xr

# Now credentials auto-detected
ds = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)
```

**Option 2: AWS credentials file (`~/.aws/credentials`)**

```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key

[production]
aws_access_key_id = prod_access_key
aws_secret_access_key = prod_secret_key
```

```python
import xarray as xr

# Use default profile (implicit)
ds = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)

# Use specific profile
ds = xr.open_zarr(
    's3://bucket/data.zarr',
    consolidated=True,
    storage_options={'profile': 'production'}
)
```

**Option 3: Explicit credentials in `storage_options`**

```python
import xarray as xr

ds = xr.open_zarr(
    's3://bucket/data.zarr',
    consolidated=True,
    storage_options={
        'key': 'your_access_key',
        'secret': 'your_secret_key',
        'client_kwargs': {'region_name': 'us-west-2'}
    }
)
```

**Option 4: Anonymous access (public buckets)**

```python
import xarray as xr

# Public bucket (no credentials)
ds = xr.open_zarr(
    's3://public-climate-data/model.zarr',
    consolidated=True,
    storage_options={'anon': True}
)
```

**Writing to S3:**

```python
import xarray as xr

ds.to_zarr(
    's3://my-bucket/output.zarr',
    mode='w',
    storage_options={
        'key': 'your_access_key',
        'secret': 'your_secret_key'
    },
    consolidated=True
)
```

**Debugging credentials:**

```python
import s3fs

# Test S3 connection
s3 = s3fs.S3FileSystem(anon=False)
print(s3.ls('s3://my-bucket'))  # Should list bucket contents

# If this works, Xarray will work too
```

**Common storage_options parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `anon` | Anonymous access (public buckets) | `True` |
| `key` | AWS access key ID | `'AKIAIOSFODNN7EXAMPLE'` |
| `secret` | AWS secret access key | `'wJalrXUtnFEMI/K7MDENG...'` |
| `token` | Session token (temporary credentials) | `'token123...'` |
| `profile` | AWS credentials profile name | `'production'` |
| `client_kwargs` | Boto3 client configuration | `{'region_name': 'us-west-2'}` |

---

## Issue 6: Performance Degradation from Non-Consolidated Metadata

**Symptom:**
```python
# Cloud read is very slow
ds = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)
# Takes 30+ seconds, hundreds of S3 requests
```

**Cause:**

Consolidated metadata file (`.zmetadata`) is missing or outdated. Xarray falls back to reading individual metadata files for each array/variable.

**Diagnosis:**

```python
import s3fs

s3 = s3fs.S3FileSystem(anon=False)

# Check if .zmetadata exists
files = s3.ls('s3://bucket/data.zarr')
has_consolidated = any('.zmetadata' in f for f in files)

print(f"Consolidated metadata exists: {has_consolidated}")

if not has_consolidated:
    print("⚠️ WARNING: No consolidated metadata - performance will be poor")
```

**Solutions:**

**Option 1: Consolidate metadata locally, upload**

```python
import zarr
import s3fs

# Download Zarr store
s3 = s3fs.S3FileSystem(anon=False)
s3.get('s3://bucket/data.zarr', 'local_copy.zarr', recursive=True)

# Consolidate locally
zarr.consolidate_metadata('local_copy.zarr')

# Upload back (including .zmetadata)
s3.put('local_copy.zarr', 's3://bucket/data.zarr', recursive=True)
```

**Option 2: Consolidate directly on S3 (if writable)**

```python
import zarr
import s3fs

s3 = s3fs.S3FileSystem(anon=False)
mapper = s3.get_mapper('s3://bucket/data.zarr')

# Consolidate in place
zarr.consolidate_metadata(mapper)
```

**Option 3: Always consolidate when writing**

```python
import xarray as xr

# Write with auto-consolidation
ds.to_zarr(
    's3://bucket/data.zarr',
    mode='w',
    consolidated=True  # Automatically creates .zmetadata
)
```

**Verification:**

```python
import xarray as xr
import time

# Benchmark consolidated vs non-consolidated
start = time.time()
ds_consolidated = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)
time_consolidated = time.time() - start

start = time.time()
ds_non_consolidated = xr.open_zarr('s3://bucket/data.zarr', consolidated=False)
time_non_consolidated = time.time() - start

print(f"Consolidated: {time_consolidated:.2f}s")
print(f"Non-consolidated: {time_non_consolidated:.2f}s")
print(f"Speedup: {time_non_consolidated / time_consolidated:.1f}x")
```

**Expected speedup:**

- Small datasets (< 10 variables): 5-10x faster
- Medium datasets (10-100 variables): 10-50x faster
- Large datasets (100+ variables): 50-500x faster

---

## Issue 7: Coordinate Mismatch During Append

**Symptom:**
```python
ds_new.to_zarr('data.zarr', mode='a', append_dim='time')

# Error: ValueError: Existing coordinate 'lat' does not match new coordinate
```

**Cause:**

Coordinates for non-appended dimensions must match exactly (including dtype, precision).

**Solution:**

**Ensure coordinates match existing store:**

```python
import xarray as xr
import numpy as np

# Read existing coordinates
ds_existing = xr.open_zarr('data.zarr')

# Create new data with EXACT same coordinates
ds_new = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'],
                   np.random.randn(10, 180, 360).astype('f4'))
}, coords={
    'time': np.arange(365, 375),  # New time values
    'lat': ds_existing['lat'].values,  # EXACT match
    'lon': ds_existing['lon'].values   # EXACT match
})

# Now append works
ds_new.to_zarr('data.zarr', mode='a', append_dim='time')
```

**Helper function:**

```python
def create_appendable_dataset(zarr_path, new_time_values, **data_vars):
    """Create dataset with coordinates matching existing Zarr store."""
    import xarray as xr

    # Load existing coordinates
    ds_existing = xr.open_zarr(zarr_path)

    # Create new dataset with matching coords
    ds_new = xr.Dataset(
        data_vars,
        coords={
            'time': new_time_values,
            **{coord: ds_existing[coord].values
               for coord in ds_existing.coords
               if coord != 'time'}
        }
    )

    return ds_new

# Usage
ds_new = create_appendable_dataset(
    'data.zarr',
    new_time_values=np.arange(365, 375),
    temperature=(['time', 'lat', 'lon'], np.random.randn(10, 180, 360))
)

ds_new.to_zarr('data.zarr', mode='a', append_dim='time')
```

---

## Issue 8: Dask Compute Hangs on Large Zarr Writes

**Symptom:**
```python
ds.to_zarr('large.zarr', mode='w', compute=True)

# Hangs indefinitely or crashes with MemoryError
```

**Cause:**

Dask tries to compute entire dataset at once, exceeding memory.

**Solutions:**

**Option 1: Two-step write (structure, then data)**

```python
import xarray as xr

# Step 1: Initialize structure (no data)
ds.to_zarr('large.zarr', mode='w', compute=False)

# Step 2: Write data with region='auto' (Dask manages memory)
write_task = ds.to_zarr('large.zarr', region='auto', compute=False)
write_task.compute()  # Controlled parallel write
```

**Option 2: Use Dask distributed scheduler**

```python
from dask.distributed import Client
import xarray as xr

client = Client(n_workers=4, memory_limit='4GB')

# Dask distributes work across workers
ds.to_zarr('large.zarr', mode='w', compute=True)

client.close()
```

**Option 3: Manual chunked writes**

```python
import xarray as xr

# Write in batches
chunk_size = 100
for i in range(0, ds.sizes['time'], chunk_size):
    end = min(i + chunk_size, ds.sizes['time'])

    ds_chunk = ds.isel(time=slice(i, end))

    if i == 0:
        ds_chunk.to_zarr('large.zarr', mode='w')
    else:
        ds_chunk.to_zarr('large.zarr', mode='a', append_dim='time')

    print(f"Written {end}/{ds.sizes['time']} time steps")
```

---

## Cross-References

- **zarr-fundamentals** — Zarr chunking, consolidation basics
- **cloud-storage-backends** — S3/GCS storage_options details
- **compression-codecs** — Compressor configuration
- **xarray-for-multidimensional-data** (scientific-domain-applications) — Xarray troubleshooting
