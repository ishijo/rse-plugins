# Common Migration Issues and Solutions

## Issue 1: HDF5 Compression Filter Incompatibility

**Symptom:**
```
OSError: Can't read data (required filter 'deflate' is not registered)
ValueError: Unknown compression filter
```

**Cause:**
HDF5 uses different compression filters than Zarr. Some filters (gzip, LZF, szip) require decompression before copying.

**Solution:**

```python
import h5py
import zarr
import numpy as np

# Check HDF5 compression
h5f = h5py.File('data.h5', 'r')
dataset = h5f['temperature']

# Check for compression filter
compression = dataset.compression
compression_opts = dataset.compression_opts

print(f"HDF5 compression: {compression} (level {compression_opts})")

# Solution: zarr.copy() automatically decompresses and recompresses
dest_group = zarr.open_group('data.zarr', mode='w')

# Specify new Zarr-compatible compressor
zarr.copy(
    dataset,
    dest_group,
    name='temperature',
    compressor=zarr.Blosc(cname='zstd', clevel=3)  # Zarr compressor
)

# HDF5's compression is decompressed, data is recompressed with Zarr's codec
h5f.close()
```

**Alternative for reading-only issues:**

```python
# If you can't read HDF5 data due to missing filter
import h5py

# Try with different library backends
h5f = h5py.File('data.h5', 'r', driver='core')  # Load into memory

# Or use h5py's virtual datasets
source = h5py.VirtualSource('data.h5', 'temperature', shape=(1000, 1000))
```

**Prevention:**
- Always test migration on small subset first
- Check HDF5 compression before migration: `h5dump -H data.h5`
- Use `zarr.copy()` which handles decompression automatically

---

## Issue 2: Multi-File Dimension Conflicts

**Symptom:**
```
ValueError: conflicting values for variable 'time' on objects to be combined
MergeError: Cannot merge datasets with conflicting dimension coordinates
```

**Cause:**
Files have inconsistent dimension definitions (overlapping time ranges, different coordinate values, mismatched units).

**Solution:**

```python
import xarray as xr
import numpy as np

# Diagnose the issue
files = ['data_2020.nc', 'data_2021.nc']

for file in files:
    ds = xr.open_dataset(file)
    print(f"{file}: time range {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  shape: {ds.time.shape}, dtype: {ds.time.dtype}")
    ds.close()

# Solution 1: Fix overlapping coordinates with preprocess
def fix_time_coord(ds):
    """Ensure consistent time coordinates."""
    # Reset time to integer index if needed
    ds = ds.assign_coords(time=np.arange(len(ds.time)))
    return ds

ds = xr.open_mfdataset(
    files,
    combine='by_coords',
    preprocess=fix_time_coord
)

# Solution 2: Use concat with explicit dimension
datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim='time', data_vars='minimal', coords='minimal')

# Solution 3: Drop conflicting coordinates
def drop_conflicts(ds):
    """Drop problematic coordinates."""
    if 'time_bnds' in ds:
        ds = ds.drop_vars('time_bnds')
    return ds

ds = xr.open_mfdataset(files, preprocess=drop_conflicts)

# Write to Zarr
ds.to_zarr('consolidated.zarr', mode='w')
ds.close()
```

**Prevention:**
- Validate file consistency before migration: `ncdump -h file.nc`
- Use `combine='nested'` with explicit `concat_dim` for known structure
- Check coordinate metadata: units, calendar, reference time

---

## Issue 3: Memory Overflow During Migration

**Symptom:**
```
MemoryError: Unable to allocate array
killed (Out of Memory)
```

**Cause:**
Loading entire array into memory during migration. Common with NetCDF → Zarr or large HDF5 datasets.

**Solution:**

```python
import xarray as xr
import dask.array as da
import zarr

# Problem: This loads full array into memory
ds = xr.open_dataset('large.nc')  # No chunks specified
ds.to_zarr('output.zarr')  # MemoryError

# Solution 1: Use Dask chunks (lazy loading)
ds = xr.open_dataset('large.nc', chunks={'time': 50, 'lat': 90, 'lon': 180})
ds.to_zarr('output.zarr', mode='w')  # Processes in chunks

# Solution 2: Manual batch processing
import h5py

h5f = h5py.File('large.h5', 'r')
source = h5f['data']

dest = zarr.open_array(
    'output.zarr',
    mode='w',
    shape=source.shape,
    chunks=(100, 100, 100),
    dtype=source.dtype
)

# Process in batches
batch_size = 100
for i in range(0, source.shape[0], batch_size):
    end = min(i + batch_size, source.shape[0])
    dest[i:end] = source[i:end]  # Only batch loaded into memory

h5f.close()

# Solution 3: Use rechunker for existing Zarr
from rechunker import rechunk

source = zarr.open_array('poorly_chunked.zarr', mode='r')

rechunk_plan = rechunk(
    source,
    target_chunks=(50, 90, 180),
    max_mem='2GB',  # Limit memory usage
    target_store='rechunked.zarr',
    temp_store='temp.zarr'
)

rechunk_plan.execute()
```

**Prevention:**
- Always specify `chunks=` when opening large datasets
- Monitor memory usage: `htop` or `psutil`
- Use Dask with explicit memory limits
- Test on subset first

---

## Issue 4: Missing Coordinates After Migration

**Symptom:**
```python
# Coordinates missing after to_zarr
ds = xr.open_zarr('migrated.zarr')
print(ds.coords)  # Empty or incomplete
```

**Cause:**
Xarray's `to_zarr()` may not write non-dimension coordinates by default, or HDF5 coordinates not recognized.

**Solution:**

```python
import xarray as xr
import zarr
import h5py

# Problem: Coordinates lost during migration
ds = xr.open_dataset('data.nc')
print(f"Original coords: {list(ds.coords)}")  # ['time', 'lat', 'lon', 'time_bnds']

ds.to_zarr('output.zarr', mode='w')

ds_read = xr.open_zarr('output.zarr')
print(f"After migration: {list(ds_read.coords)}")  # ['time', 'lat', 'lon'] - missing time_bnds

# Solution 1: Explicitly include coordinates in encoding
encoding = {var: {'chunks': (10, 90, 180)} for var in ds.data_vars}

# Include non-dimension coordinates
for coord in ds.coords:
    if coord not in ds.dims:  # Non-dimension coordinate
        encoding[coord] = {}

ds.to_zarr('output.zarr', mode='w', encoding=encoding)

# Solution 2: For HDF5 sources, manually copy coordinates
h5f = h5py.File('data.h5', 'r')
dest_group = zarr.open_group('output.zarr', mode='w')

# Copy main data
zarr.copy(h5f['temperature'], dest_group, name='temperature')

# Manually copy coordinate arrays
for coord_name in ['time', 'lat', 'lon', 'height']:
    if coord_name in h5f:
        zarr.copy(h5f[coord_name], dest_group, name=coord_name)
        # Mark as coordinate in metadata
        dest_group[coord_name].attrs['_ARRAY_DIMENSIONS'] = [coord_name]

h5f.close()

# Solution 3: Use xr.set_coords() to restore
ds_read = xr.open_zarr('output.zarr')

# If variable should be coordinate
ds_read = ds_read.set_coords(['time_bnds'])

print(f"Restored coords: {list(ds_read.coords)}")
```

**Prevention:**
- Always verify coordinates after migration: `list(ds.coords)`
- Include all coordinates in encoding dictionary
- Use Xarray for NetCDF (preserves CF conventions)

---

## Issue 5: VirtualiZarr References Break After File Moves

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/old/path/data.nc'
```

**Cause:**
VirtualiZarr stores absolute paths to source files. Moving/renaming source files breaks references.

**Solution:**

```python
from virtualizarr import open_virtual_dataset
import xarray as xr
import json
from pathlib import Path

# Problem: Virtual Zarr references old paths
vds = open_virtual_dataset('data.nc', indexes={})
vds.virtualize.to_zarr('virtual.zarr')

# Move source file
# mv data.nc /new/location/data.nc

# Virtual references now broken
ds = xr.open_zarr('virtual.zarr')  # FileNotFoundError

# Solution 1: Use relative paths (if supported)
# Currently VirtualiZarr uses absolute paths, so manually edit references

# Read virtual Zarr metadata
with open('virtual.zarr/.zmetadata', 'r') as f:
    metadata = json.load(f)

# Update paths in references (example for Kerchunk format)
# This is format-specific and may require custom script

# Solution 2: Regenerate virtual dataset
vds = open_virtual_dataset('/new/location/data.nc', indexes={})
vds.virtualize.to_zarr('virtual.zarr', mode='w')  # Overwrite with new paths

# Solution 3: Materialize to native Zarr
# Best solution for production: convert virtual to physical
ds = xr.open_zarr('virtual.zarr')
ds.to_zarr('materialized.zarr', mode='w')  # Copies actual data

# Now independent of source files
```

**Prevention:**
- Use VirtualiZarr for exploration, not production
- Document source file locations
- Materialize to native Zarr for long-term storage
- Use symbolic links to keep source paths stable

---

## Issue 6: Metadata Loss with copy_store

**Symptom:**
```python
# After copy_store, attributes missing
source = zarr.open_group('source.zarr', mode='r')
print(source['temperature'].attrs)  # {'units': 'K', 'long_name': '...'}

# Copy with copy_store
zarr.copy_store(source.store, 'dest.zarr')

dest = zarr.open_group('dest.zarr', mode='r')
print(dest['temperature'].attrs)  # Empty or missing attributes
```

**Cause:**
`zarr.copy_store()` performs binary copy of chunk files but may not copy all metadata files correctly.

**Solution:**

```python
import zarr

# Problem: copy_store is fast but loses metadata
source = zarr.open_group('source.zarr', mode='r')

# BAD: Binary copy (fast but incomplete)
zarr.copy_store(source.store, 'dest_incomplete.zarr')

# Solution 1: Use copy_all (copies data + metadata)
dest = zarr.open_group('dest_complete.zarr', mode='w')
zarr.copy_all(source, dest)  # Copies arrays, groups, and all metadata

# Verify
print("Metadata preserved:")
for name, arr in dest.arrays():
    print(f"  {name}: {dict(arr.attrs)}")

# Solution 2: Manual metadata copy after copy_store
zarr.copy_store(source.store, 'dest_manual.zarr')

# Reopen and copy metadata
source = zarr.open_group('source.zarr', mode='r')
dest = zarr.open_group('dest_manual.zarr', mode='r+')

def copy_metadata_recursive(src_group, dst_group):
    """Recursively copy all metadata."""
    # Copy group attributes
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    # Copy array attributes
    for name, item in src_group.items():
        if isinstance(item, zarr.Array) and name in dst_group:
            for key, value in item.attrs.items():
                dst_group[name].attrs[key] = value
        elif isinstance(item, zarr.Group) and name in dst_group:
            copy_metadata_recursive(item, dst_group[name])

copy_metadata_recursive(source, dest)

# Solution 3: Use zarr.copy() per array (preserves metadata)
source = zarr.open_group('source.zarr', mode='r')
dest = zarr.open_group('dest_copy.zarr', mode='w')

for name, source_arr in source.arrays():
    zarr.copy(source_arr, dest, name=name)  # Includes metadata
```

**When to use each method:**

| Method | Speed | Metadata | Rechunking | Use Case |
|--------|-------|----------|------------|----------|
| `copy_store()` | Fastest | ❌ No | ❌ No | Cloud backup, exact binary copy |
| `copy_all()` | Fast | ✓ Yes | ❌ No | Standard migration with metadata |
| `zarr.copy()` | Medium | ✓ Yes | ✓ Yes | Migration with rechunking |

**Prevention:**
- Use `copy_all()` instead of `copy_store()` for migrations
- Always verify metadata after migration
- Document which copy method was used

---

## Issue 7: Slow Cloud Writes Without Consolidation

**Symptom:**
Very slow reads from cloud Zarr stores after migration (100+ seconds for metadata).

**Cause:**
Unconsolidated metadata requires separate network request for each `.zarray`, `.zattrs`, and `.zgroup` file.

**Solution:**

```python
import zarr
import xarray as xr
import time

# Problem: Write to S3 without consolidation
ds = xr.open_dataset('data.nc')
ds.to_zarr('s3://bucket/data.zarr', mode='w')

# Reading is very slow (100+ requests)
start = time.time()
ds_slow = xr.open_zarr('s3://bucket/data.zarr')  # consolidated=False by default
elapsed = time.time() - start
print(f"Unconsolidated read: {elapsed:.2f}s")  # ~120s

# Solution: Always consolidate metadata after cloud writes
zarr.consolidate_metadata('s3://bucket/data.zarr')

# Reading is now fast (1 request)
start = time.time()
ds_fast = xr.open_zarr('s3://bucket/data.zarr', consolidated=True)
elapsed = time.time() - start
print(f"Consolidated read: {elapsed:.2f}s")  # ~1.2s

# For Xarray writes, use consolidated parameter
ds.to_zarr('s3://bucket/data.zarr', mode='w', consolidated=True)

# Re-consolidate after any structural changes
ds_new = xr.Dataset({'new_var': (['time'], [1, 2, 3])})
ds_new.to_zarr('s3://bucket/data.zarr', mode='a')  # Add variable

# Must re-consolidate
zarr.consolidate_metadata('s3://bucket/data.zarr')
```

**Performance impact:**

| Metadata State | Requests | Open Time |
|----------------|----------|-----------|
| Unconsolidated | 100+ | 60-120s |
| Consolidated | 1 | 1-2s |

**Prevention:**
- Always run `zarr.consolidate_metadata()` after cloud writes
- Re-consolidate after any append or structural change
- Use `consolidated=True` when reading

---

## Issue 8: Time-Based Chunking After Spatial Migration

**Symptom:**
Time-series queries are very slow after migrating with spatial-optimized chunks.

**Cause:**
Chunk strategy optimized for wrong access pattern. Spatial chunks require reading many chunks for time-series.

**Solution:**

```python
import zarr
from rechunker import rechunk

# Original: spatial-optimized chunks (1, 180, 360)
# Time-series query: must read 365 chunks for 1-year series at single point

arr = zarr.open_array('spatial_optimized.zarr', mode='r')
print(f"Current chunks: {arr.chunks}")  # (1, 180, 360)

# Solution 1: Rechunk for time-series access
from rechunker import rechunk

rechunk_plan = rechunk(
    arr,
    target_chunks=(30, 90, 180),  # Balanced for both access patterns
    max_mem='4GB',
    target_store='balanced.zarr',
    temp_store='temp_rechunk.zarr'
)

print("Rechunking...")
rechunk_plan.execute()

# Solution 2: Maintain two versions with different chunking
# Version 1: Spatial analysis (small time chunks)
zarr.copy(arr, zarr.open_group('spatial.zarr', mode='w'), name='temperature',
          chunks=(1, 180, 360))

# Version 2: Time-series analysis (large time chunks)
zarr.copy(arr, zarr.open_group('timeseries.zarr', mode='w'), name='temperature',
          chunks=(100, 45, 90))

# Use appropriate version for workload
ds_spatial = xr.open_zarr('spatial.zarr')  # For maps
ds_timeseries = xr.open_zarr('timeseries.zarr')  # For time series

# Solution 3: Choose balanced chunks from start (migration-time decision)
# Rule of thumb: ~3-5 MB chunks, balanced across dimensions
# For (365, 720, 1440) float32 data:
# Balanced: (30, 90, 180) → 30*90*180*4 = 19.4 MB ✓
```

**Prevention:**
- Profile access patterns before migration
- Choose balanced chunks: `(time_chunk, lat_chunk, lon_chunk)` all non-trivial
- Target 1-10 MB per chunk (uncompressed)
- Test representative queries on subset before full migration
