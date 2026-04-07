# Data Migration Patterns

## Pattern 1: Direct HDF5 to Local Zarr via h5py

**When to use:** Simple HDF5 migration with optional rechunking, local destination.

```python
import h5py
import zarr
import numpy as np

# Open HDF5 source
h5f = h5py.File('climate_model.h5', 'r')
source = h5f['temperature']  # Shape: (365, 720, 1440)

print(f"Source: {source.shape}, chunks={source.chunks}")

# Create Zarr destination with rechunking
dest_group = zarr.open_group('climate_model.zarr', mode='w')

# Copy with new chunks optimized for time-series access
zarr.copy(
    source,
    dest_group,
    name='temperature',
    chunks=(30, 90, 180),  # Larger time chunks
    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
)

dest = dest_group['temperature']

# Copy metadata
for key, value in source.attrs.items():
    dest.attrs[key] = value

# Copy coordinates if present
for coord in ['time', 'lat', 'lon']:
    if coord in h5f:
        zarr.copy(h5f[coord], dest_group, name=coord)

h5f.close()

print(f"✓ Migrated to {dest.store.path}")
print(f"  Chunks: {dest.chunks}")
print(f"  Compression: {dest.compressor}")
```

**Key points:**
- `zarr.copy()` decompresses source and recompresses to Zarr
- Can change chunks during copy
- Must manually copy metadata and coordinates
- Fast for single arrays

---

## Pattern 2: Multi-File NetCDF to Cloud Zarr

**When to use:** Consolidating multiple NetCDF files into single cloud Zarr store.

```python
import xarray as xr
import zarr

# Open multiple NetCDF files as single dataset
ds = xr.open_mfdataset(
    'data_*.nc',  # Pattern: data_2020.nc, data_2021.nc, ...
    combine='by_coords',  # Concatenate along time dimension
    parallel=True,  # Parallel file reading
    engine='netcdf4'
)

print(f"Combined dataset: {ds.sizes}")

# Define encoding for all variables
encoding = {
    'temperature': {
        'chunks': (10, 90, 180),
        'compressor': zarr.Blosc(cname='zstd', clevel=3),
        'dtype': 'float32'
    },
    'precipitation': {
        'chunks': (10, 90, 180),
        'compressor': zarr.Blosc(cname='zstd', clevel=3),
        'dtype': 'float32'
    }
}

# Write to S3
ds.to_zarr(
    's3://my-bucket/consolidated.zarr',
    mode='w',
    encoding=encoding,
    storage_options={'anon': False}  # Use AWS credentials
)

# Consolidate metadata (CRITICAL for cloud performance)
zarr.consolidate_metadata('s3://my-bucket/consolidated.zarr')

ds.close()

print("✓ Multi-file migration complete")
```

**Key points:**
- `open_mfdataset()` handles concatenation automatically
- Preserves CF metadata from NetCDF
- Always consolidate metadata for cloud stores
- Single Zarr store replaces multiple files

---

## Pattern 3: VirtualiZarr Zero-Copy Virtual Dataset

**When to use:** Immediate Zarr access without copying data, source files too large to duplicate.

```python
from virtualizarr import open_virtual_dataset
import xarray as xr

# Create virtual Zarr from multiple HDF5/NetCDF files
virtual_datasets = []

for file in ['obs_2020.nc', 'obs_2021.nc', 'obs_2022.nc']:
    vds = open_virtual_dataset(
        file,
        indexes={}  # Disable index loading for performance
    )
    virtual_datasets.append(vds)

# Concatenate into unified virtual dataset
combined = xr.concat(virtual_datasets, dim='time')

# Write virtual references (metadata only, no data copied)
combined.virtualize.to_zarr('virtual_obs.zarr')

print("✓ Virtual dataset created (no data copied)")

# Read via Zarr API
ds = xr.open_zarr('virtual_obs.zarr')
print(f"Accessible shape: {ds['temperature'].shape}")

# Data fetched from original files on access
subset = ds['temperature'].isel(time=slice(0, 10)).compute()

# Later: materialize to native Zarr if needed
ds.to_zarr('materialized_obs.zarr', mode='w')
```

**Key points:**
- Zero data copying, instant "migration"
- Virtual dataset references byte ranges in source files
- Source files must remain accessible
- Read-only access
- Use for exploration before committing to full migration

---

## Pattern 4: Zarr v2 to v3 Format Migration

**When to use:** Upgrading to Zarr v3 for sharding, better cloud performance, or new features.

```python
import zarr

# Open v2 source
z_v2 = zarr.open_group('data_v2.zarr', mode='r')

# Create v3 destination
z_v3 = zarr.open_group('data_v3.zarr', mode='w', zarr_format=3)

def copy_recursive(src_group, dst_group):
    """Recursively copy group hierarchy from v2 to v3."""

    # Copy group attributes
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    # Copy arrays and subgroups
    for name, item in src_group.items():
        if isinstance(item, zarr.Array):
            print(f"  Copying array: {name}")

            # Create v3 array with sharding
            arr_v3 = dst_group.create_array(
                name,
                shape=item.shape,
                chunks=item.chunks,
                shards=tuple(c * 10 for c in item.chunks),  # Shard 10x10x10 chunks
                dtype=item.dtype,
                fill_value=item.fill_value,
                compressors='zstd'  # v3 uses 'compressors' not 'compressor'
            )

            # Copy data
            arr_v3[:] = item[:]

            # Copy attributes
            for key, value in item.attrs.items():
                arr_v3.attrs[key] = value

        elif isinstance(item, zarr.Group):
            print(f"  Entering group: {name}")
            subgroup = dst_group.create_group(name)
            copy_recursive(item, subgroup)

# Perform migration
print("Migrating v2 → v3...")
copy_recursive(z_v2, z_v3)

print("✓ Format migration complete")
print(f"  v2 format: {z_v2.zarr_format}")
print(f"  v3 format: {z_v3.zarr_format}")
```

**Key points:**
- v3 supports sharding (reduces object count)
- v3 uses `compressors` parameter instead of `compressor`
- Default compressor changes from Blosc to Zstd
- Recursive function handles nested groups
- Preserves full metadata hierarchy

---

## Pattern 5: Cloud Migration with Streaming Copy

**When to use:** Moving large local Zarr to S3/GCS without loading into memory.

```python
import zarr
import s3fs
import fsspec

# Open local source
source = zarr.open_group('local_data.zarr', mode='r')

# Setup S3 filesystem
s3 = s3fs.S3FileSystem(anon=False)

# Create S3 store
s3_store = s3fs.S3Map(root='my-bucket/cloud_data.zarr', s3=s3)

# Binary copy (fastest, preserves exact chunks and compression)
print("Copying to S3...")
zarr.copy_store(source.store, s3_store)

print("✓ Cloud migration complete")

# Consolidate metadata for cloud performance
zarr.consolidate_metadata(s3_store)

# Verify
cloud_group = zarr.open_group(s3_store, mode='r')
print(f"Arrays in cloud store: {list(cloud_group.array_keys())}")
```

**Alternative: Copy with rechunking**

```python
# If you need to change chunks/compression during cloud migration
dest_group = zarr.open_group(s3_store, mode='w')

for name, source_arr in source.arrays():
    print(f"Copying {name}...")

    # Copy with new cloud-optimized chunks
    zarr.copy(
        source_arr,
        dest_group,
        name=name,
        chunks=tuple(max(c, 100) for c in source_arr.chunks),  # Larger chunks for cloud
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    # Copy metadata
    for key, value in source_arr.attrs.items():
        dest_group[name].attrs[key] = value

zarr.consolidate_metadata(s3_store)
```

**Key points:**
- `copy_store()` for exact binary copy (fastest)
- `zarr.copy()` for rechunking during migration
- Always consolidate metadata after cloud migration
- Larger chunks (>1 MB) recommended for cloud access

---

## Pattern 6: Incremental Migration for Large Archives

**When to use:** Dataset too large for single migration, need monitoring and validation per batch.

```python
import h5py
import zarr
import numpy as np
from tqdm import tqdm

def migrate_in_batches(h5_path, zarr_path, variable, batch_size=100):
    """Migrate large HDF5 dataset in batches."""

    # Open source
    h5f = h5py.File(h5_path, 'r')
    source = h5f[variable]

    print(f"Source shape: {source.shape}")

    # Initialize Zarr destination
    z = zarr.open_array(
        zarr_path,
        mode='w',
        shape=source.shape,
        chunks=(batch_size, *source.shape[1:]),
        dtype=source.dtype,
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    # Copy metadata
    for key, value in source.attrs.items():
        z.attrs[key] = value

    # Migrate in batches along first dimension
    n_batches = int(np.ceil(source.shape[0] / batch_size))

    print(f"Migrating {n_batches} batches of size {batch_size}...")

    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, source.shape[0])

        # Read batch from HDF5
        batch_data = source[start:end]

        # Write to Zarr
        z[start:end] = batch_data

        # Validate batch (sample check)
        if i % 10 == 0:  # Validate every 10th batch
            idx = (start, *[0] * (len(source.shape) - 1))
            assert np.allclose(source[idx], z[idx]), f"Validation failed at batch {i}"

    h5f.close()

    print("✓ Incremental migration complete")
    return z

# Usage
z = migrate_in_batches('large_archive.h5', 'large_archive.zarr', 'temperature', batch_size=100)

print(f"Final shape: {z.shape}")
print(f"Chunks: {z.chunks}")
```

**Parallel batch migration with Dask:**

```python
import dask.array as da
import zarr

# Wrap HDF5 as Dask array
h5f = h5py.File('large_archive.h5', 'r')
source = h5f['temperature']

# Create Dask array with controlled chunk size
dask_array = da.from_array(source, chunks=(100, 90, 180))

# Create Zarr destination
z = zarr.open_array(
    'large_archive.zarr',
    mode='w',
    shape=source.shape,
    chunks=(100, 90, 180),
    dtype=source.dtype,
    compressor=zarr.Blosc(cname='zstd', clevel=3)
)

# Parallel migration via Dask
print("Parallel batch migration...")
da.to_zarr(dask_array, 'large_archive.zarr', overwrite=True)

h5f.close()
```

**Key points:**
- Batch size controls memory usage
- Progress tracking with tqdm
- Periodic validation catches errors early
- Dask enables parallel batch processing
- Suitable for datasets exceeding available RAM

---

## Pattern 7: Rechunking Existing Zarr with Rechunker

**When to use:** Existing Zarr has suboptimal chunks, need to rechunk without full reload.

```python
from rechunker import rechunk
import zarr
import dask

# Open source Zarr (poor chunking)
source = zarr.open_array('poorly_chunked.zarr', mode='r')
print(f"Current chunks: {source.chunks}")  # e.g., (1, 720, 1440)

# Define target chunks (optimized for time-series)
target_chunks = (30, 90, 180)

# Setup rechunking
rechunk_plan = rechunk(
    source,
    target_chunks=target_chunks,
    max_mem='2GB',  # Memory limit per worker
    target_store='rechunked.zarr',
    temp_store='temp_rechunk.zarr'  # Temporary storage
)

# Execute rechunking
print("Rechunking...")
rechunk_plan.execute()

# Verify
rechunked = zarr.open_array('rechunked.zarr', mode='r')
print(f"New chunks: {rechunked.chunks}")

# Cleanup temp store
import shutil
shutil.rmtree('temp_rechunk.zarr')
```

**Key points:**
- Rechunker avoids loading full array into memory
- Requires temporary storage (same size as source)
- Can work on cloud stores (S3/GCS)
- More efficient than full re-migration
- Use when chunk strategy was suboptimal after initial migration
