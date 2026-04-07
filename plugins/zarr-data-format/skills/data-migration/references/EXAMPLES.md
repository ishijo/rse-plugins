# Data Migration Examples

## Example 1: Climate HDF5 Archive to Cloud Zarr with Rechunking

**Scenario:** Migrate 10-year climate model output (HDF5) to S3 with optimized chunking for spatial analysis.

```python
import h5py
import zarr
import s3fs
import numpy as np

# Source: HDF5 with time-optimized chunks
h5f = h5py.File('climate_10year.h5', 'r')
temp = h5f['temperature']  # Shape: (3650, 721, 1440) - daily 0.25° global

print(f"Source HDF5:")
print(f"  Shape: {temp.shape} (10 years × 721 lat × 1440 lon)")
print(f"  Chunks: {temp.chunks} - optimized for time-series")
print(f"  Size: {temp.size * temp.dtype.itemsize / 1024**3:.2f} GB")

# Target: S3 with spatial-optimized chunks
s3 = s3fs.S3FileSystem(anon=False, client_kwargs={'region_name': 'us-west-2'})
s3_store = s3fs.S3Map(root='climate-data/spatial_optimized.zarr', s3=s3)

dest_group = zarr.open_group(s3_store, mode='w')

# New chunks: smaller time, larger spatial (for map access)
target_chunks = (10, 180, 360)  # 10 days × 45° × 90° regions

print(f"\nMigrating to S3 with spatial chunks...")
print(f"  Target chunks: {target_chunks}")

# Copy with rechunking
zarr.copy(
    temp,
    dest_group,
    name='temperature',
    chunks=target_chunks,
    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
)

dest = dest_group['temperature']

# Copy metadata
dest.attrs['units'] = temp.attrs.get('units', 'K')
dest.attrs['long_name'] = temp.attrs.get('long_name', 'Air Temperature')
dest.attrs['source'] = 'CESM2 climate model'
dest.attrs['migration_date'] = '2024-04-06'
dest.attrs['chunk_strategy'] = 'spatial_optimized'

# Copy coordinates
for coord_name in ['time', 'lat', 'lon']:
    if coord_name in h5f:
        zarr.copy(h5f[coord_name], dest_group, name=coord_name)

h5f.close()

# Consolidate metadata (CRITICAL for S3 performance)
print("Consolidating metadata...")
zarr.consolidate_metadata(s3_store)

print("\n✓ Migration complete")
print(f"  Location: s3://climate-data/spatial_optimized.zarr")
print(f"  Access: xr.open_zarr('s3://climate-data/spatial_optimized.zarr', consolidated=True)")

# Verify cloud access
cloud_arr = zarr.open_array(s3_store, path='temperature', mode='r')
print(f"\nVerification:")
print(f"  Cloud shape: {cloud_arr.shape}")
print(f"  Cloud chunks: {cloud_arr.chunks}")

# Test spatial access performance
import time
start = time.time()
spatial_slice = cloud_arr[0, :, :]  # First day, full map
elapsed = time.time() - start
print(f"  Spatial slice (first day): {elapsed:.3f}s")
```

---

## Example 2: Multi-File NetCDF Time Series to Single Zarr

**Scenario:** Consolidate 100 monthly NetCDF files into single Zarr store with CF metadata.

```python
import xarray as xr
import zarr
import glob

# List all monthly files
files = sorted(glob.glob('monthly_obs_*.nc'))  # monthly_obs_202001.nc, ...202012.nc, etc.
print(f"Found {len(files)} monthly files")

# Open as single dataset with Dask
ds = xr.open_mfdataset(
    files,
    combine='by_coords',  # Concatenate along time
    parallel=True,  # Parallel file reading
    chunks={'time': 30, 'lat': 90, 'lon': 180}  # Dask chunks
)

print(f"\nCombined dataset:")
print(ds)
print(f"Total time steps: {ds.sizes['time']}")

# Configure encoding for all variables
encoding = {}
for var in ds.data_vars:
    encoding[var] = {
        'chunks': (30, 90, 180),  # Balanced chunks
        'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
        'dtype': 'float32'  # Downcast from float64 if appropriate
    }

# Also encode coordinates
for coord in ['time', 'lat', 'lon']:
    if coord in ds.coords:
        encoding[coord] = {'dtype': ds[coord].dtype}

# Write to local Zarr
output_path = 'consolidated_monthly.zarr'

print(f"\nWriting to Zarr: {output_path}")
ds.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)

ds.close()

print("✓ Consolidation complete")

# Verify
ds_zarr = xr.open_zarr(output_path, consolidated=True)
print(f"\nVerification:")
print(f"  Variables: {list(ds_zarr.data_vars)}")
print(f"  Time range: {ds_zarr.time.values[0]} to {ds_zarr.time.values[-1]}")
print(f"  Chunks: {ds_zarr['temperature'].chunks}")

# Compare file counts
import os
nc_file_count = len(files)
zarr_file_count = sum(1 for _ in Path(output_path).rglob('*') if _.is_file())
print(f"  Files: {nc_file_count} NetCDF → {zarr_file_count} Zarr objects")
```

---

## Example 3: VirtualiZarr Multi-File Aggregation with Icechunk

**Scenario:** Create zero-copy virtual dataset from 1000+ NetCDF files, then persist to Icechunk.

```python
from virtualizarr import open_virtual_dataset
import xarray as xr
from icechunk import IcechunkStore, StorageConfig
import zarr
from pathlib import Path

# List all files (1000+ satellite granules)
data_dir = Path('satellite_l2_data')
files = sorted(data_dir.glob('*.nc'))
print(f"Found {len(files)} granules")

# Create virtual datasets (no data loaded)
print("Creating virtual datasets...")
virtual_datasets = []

for i, file in enumerate(files[:1000]):  # Process first 1000
    if i % 100 == 0:
        print(f"  Processed {i}/{len(files)}")

    vds = open_virtual_dataset(
        str(file),
        indexes={}  # Skip index loading
    )
    virtual_datasets.append(vds)

# Concatenate into unified virtual dataset
print("\nConcatenating virtual datasets...")
combined = xr.concat(virtual_datasets, dim='time')

print(f"Virtual dataset shape: {combined['reflectance'].shape}")

# Option 1: Write virtual references to standard Zarr
print("\nWriting virtual references to Zarr...")
combined.virtualize.to_zarr('virtual_satellite.zarr')

print("✓ Virtual dataset ready (no data copied)")

# Option 2: Persist to Icechunk for ACID + versioning
print("\nPersisting to Icechunk...")

# Create Icechunk store (local or S3)
ic_store = IcechunkStore.create(
    storage=StorageConfig.filesystem('icechunk_satellite'),
    config={'inline_chunk_threshold_bytes': 512}
)

# Materialize virtual dataset to Icechunk
# This copies data but provides versioning
combined.to_zarr(ic_store, mode='w')

# Commit first version
ic_store.commit('Initial satellite dataset from 1000 granules')

print("✓ Icechunk store created")

# Add more data later (append)
print("\nAppending new granules...")
new_files = files[1000:1100]
new_vds_list = [open_virtual_dataset(str(f), indexes={}) for f in new_files]
new_combined = xr.concat(new_vds_list, dim='time')

# Append to Icechunk
new_combined.to_zarr(ic_store, mode='a', append_dim='time')
ic_store.commit('Added 100 more granules')

# View history
print("\nCommit history:")
for commit in ic_store.log():
    print(f"  {commit.id[:8]}: {commit.message}")
```

---

## Example 4: Large Migration with Comprehensive Validation

**Scenario:** Migrate 500 GB HDF5 archive with full validation suite beyond random sampling.

```python
import h5py
import zarr
import numpy as np
import hashlib
from tqdm import tqdm

def compute_checksum(array, chunk_size=1000):
    """Compute checksum for large array in chunks."""
    hasher = hashlib.sha256()

    for i in range(0, array.shape[0], chunk_size):
        end = min(i + chunk_size, array.shape[0])
        chunk = array[i:end]
        hasher.update(chunk.tobytes())

    return hasher.hexdigest()


def validate_comprehensive(h5_path, zarr_path, variable):
    """Comprehensive validation beyond random sampling."""

    print("Opening source and destination...")
    h5f = h5py.File(h5_path, 'r')
    z = zarr.open(zarr_path, mode='r')

    h5_arr = h5f[variable]
    z_arr = z[variable]

    print(f"\n1. Structural validation")
    assert h5_arr.shape == z_arr.shape, "Shape mismatch"
    assert h5_arr.dtype == z_arr.dtype, "Dtype mismatch"
    print(f"   ✓ Shape: {z_arr.shape}")
    print(f"   ✓ Dtype: {z_arr.dtype}")

    print(f"\n2. Checksum validation (chunked)")
    h5_checksum = compute_checksum(h5_arr, chunk_size=100)
    z_checksum = compute_checksum(z_arr, chunk_size=100)
    assert h5_checksum == z_checksum, "Checksums do not match"
    print(f"   ✓ Checksum: {z_checksum[:16]}...")

    print(f"\n3. Statistical validation")
    # Compare statistics without loading full arrays
    h5_sum = h5_arr[...].sum()  # Ellipsis for all dimensions
    z_sum = z_arr[...].sum()
    np.testing.assert_allclose(h5_sum, z_sum, rtol=1e-6)
    print(f"   ✓ Sum: {z_sum:.6e}")

    h5_mean = h5_arr[...].mean()
    z_mean = z_arr[...].mean()
    np.testing.assert_allclose(h5_mean, z_mean, rtol=1e-6)
    print(f"   ✓ Mean: {z_mean:.6f}")

    print(f"\n4. Metadata validation")
    for key in h5_arr.attrs.keys():
        assert key in z_arr.attrs, f"Missing attribute: {key}"
        h5_val = h5_arr.attrs[key]
        z_val = z_arr.attrs[key]
        assert h5_val == z_val, f"Attribute mismatch: {key}"

    print(f"   ✓ All {len(h5_arr.attrs)} attributes preserved")

    print(f"\n5. Slice validation (10 random slices)")
    np.random.seed(42)
    for i in range(10):
        idx = tuple(slice(np.random.randint(0, s-10), np.random.randint(10, s))
                    for s in h5_arr.shape)
        h5_slice = h5_arr[idx]
        z_slice = z_arr[idx]
        np.testing.assert_array_equal(h5_slice, z_slice)

    print(f"   ✓ All slices match")

    h5f.close()
    print("\n✓ Comprehensive validation passed")


# Run migration
print("Starting large migration...")
h5f = h5py.File('large_archive_500gb.h5', 'r')
source = h5f['observation_data']

dest = zarr.open_array(
    'large_archive_500gb.zarr',
    mode='w',
    shape=source.shape,
    chunks=(100, 90, 180),
    dtype=source.dtype,
    compressor=zarr.Blosc(cname='zstd', clevel=3)
)

# Migrate in batches with progress bar
batch_size = 100
for i in tqdm(range(0, source.shape[0], batch_size)):
    end = min(i + batch_size, source.shape[0])
    dest[i:end] = source[i:end]

h5f.close()

# Run comprehensive validation
validate_comprehensive('large_archive_500gb.h5', 'large_archive_500gb.zarr', 'observation_data')
```

---

## Example 5: Parallel Migration with Dask Distributed

**Scenario:** Use Dask cluster to parallelize migration of large multi-variable dataset.

```python
from dask.distributed import Client, LocalCluster
import xarray as xr
import zarr
import h5py

# Setup Dask cluster
cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit='4GB')
client = Client(cluster)

print(f"Dask dashboard: {client.dashboard_link}")

# Open HDF5 with Xarray (using Dask)
ds = xr.open_dataset(
    'multi_variable.h5',
    engine='h5netcdf',
    chunks={'time': 50, 'lat': 90, 'lon': 180}
)

print(f"Dataset variables: {list(ds.data_vars)}")

# Configure encoding for each variable
encoding = {}
for var in ds.data_vars:
    encoding[var] = {
        'chunks': (50, 90, 180),
        'compressor': zarr.Blosc(cname='zstd', clevel=3),
        'dtype': 'float32'
    }

# Write to Zarr (Dask handles parallelization)
print("Migrating with Dask workers...")

future = ds.to_zarr(
    'multi_variable.zarr',
    mode='w',
    encoding=encoding,
    compute=False  # Return delayed task
)

# Execute with progress
result = future.compute()

print("✓ Parallel migration complete")

# Consolidate metadata
zarr.consolidate_metadata('multi_variable.zarr')

# Verify
ds_zarr = xr.open_zarr('multi_variable.zarr', consolidated=True)
print(f"\nMigrated variables: {list(ds_zarr.data_vars)}")

client.close()
cluster.close()
```
