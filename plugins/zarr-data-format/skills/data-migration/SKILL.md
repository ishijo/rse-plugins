---
name: data-migration
description: This skill should be used when the user asks to "migrate to Zarr", "convert HDF5 to Zarr", "convert NetCDF to Zarr", "upgrade Zarr v2 to v3", "rechunk dataset", "migrate data to cloud", "consolidate multiple files", "create virtual dataset", "use VirtualiZarr", "set up Icechunk", or needs guidance on data format migration, rechunking strategies, zero-copy virtual datasets, ACID-compliant Zarr stores, or production migration workflows.
---

# Data Migration to Zarr

Master data migration workflows for converting legacy scientific data formats (HDF5, NetCDF, Zarr v2) to modern cloud-optimized Zarr stores using direct copy, rechunking, parallel processing, and ACID-compliant stores. Covers VirtualiZarr for zero-copy virtual datasets, Icechunk for versioned stores, and production migration patterns with validation.

## Quick Reference Card

### Installation & Setup

See the plugin README for installation instructions.

### Migration Decision Tree

```
What format are you migrating FROM?
├─ HDF5 (.h5, .hdf5)
│  ├─ Need immediate access? → VirtualiZarr (zero-copy)
│  ├─ Need rechunking? → zarr.copy() or Xarray
│  └─ Simple copy? → h5py + zarr.copy()
│
├─ NetCDF (.nc, .nc4)
│  ├─ Climate/oceanographic data? → Xarray (CF metadata preserved)
│  ├─ Multiple files to consolidate? → Xarray + concat
│  └─ Virtual dataset? → VirtualiZarr
│
├─ Zarr v2 → Zarr v3
│  └─ zarr.copy() with target v3 store
│
└─ Multiple formats/files
   └─ VirtualiZarr + multi-file aggregation

Do you need to rechunk during migration?
├─ YES → Use zarr.copy() or Xarray with encoding
└─ NO → Direct copy or VirtualiZarr

Is the dataset larger than available memory?
├─ YES → Dask-based parallel migration
└─ NO → Direct in-memory migration

Do you need version control and ACID properties?
├─ YES → Migrate to Icechunk store
└─ NO → Standard Zarr store

Migrating to cloud storage (S3/GCS/Azure)?
├─ YES → Use storage_options with Xarray or fsspec
└─ NO → Local filesystem migration
```

### Migration Options Table

| Method | Use Case | Pros | Cons | Speed |
|--------|----------|------|------|-------|
| **zarr.copy()** | Direct array copy, optional rechunking | Simple, low-level control | No metadata handling | Fast |
| **Xarray** | NetCDF/HDF5 with metadata | Preserves CF metadata, high-level | Requires Xarray-compatible format | Medium |
| **VirtualiZarr** | Zero-copy virtual datasets | No data duplication, instant | Read-only, requires source files | Instant |
| **Icechunk** | ACID compliance, versioning | Git-like versioning, transactional | Newer technology, limited ecosystem | Medium |
| **Dask** | Large datasets (> memory) | Parallel, out-of-core | More complex setup | Fast (parallel) |

## When to Use This Skill

Use this skill when:

- **Migrating legacy formats** - Converting HDF5, NetCDF, GRIB to Zarr
- **Upgrading Zarr versions** - Migrating Zarr v2 → Zarr v3
- **Rechunking for cloud** - Optimizing chunk sizes for cloud access patterns
- **Consolidating datasets** - Merging multiple files into single Zarr store
- **Creating virtual datasets** - Zero-copy aggregation with VirtualiZarr
- **Cloud migration** - Moving local data to S3/GCS/Azure
- **Version control** - Setting up ACID-compliant Icechunk stores
- **Validation** - Verifying migration correctness and performance

## Core Concepts

### 1. Migration Strategies Overview

**Direct Copy:**
- Preserve existing chunks and compression
- Fastest migration path
- Use when chunk structure already optimal

**Rechunking Migration:**
- Change chunk sizes during migration
- Optimize for new access patterns (time-series vs spatial)
- Balances migration cost with future performance

**Virtual Migration:**
- Zero-copy virtual Zarr using VirtualiZarr
- No data duplication, instant "migration"
- Read-only, requires original files accessible

**Incremental Migration:**
- Migrate data in batches
- Useful for very large datasets
- Allows monitoring and validation per batch

### 2. zarr.copy() - Low-Level Array Copying

**Basic usage:**

```python
import zarr

# Open source (HDF5 via h5py)
import h5py
h5f = h5py.File('data.h5', 'r')
source = h5f['temperature']  # HDF5 dataset

# Open destination (Zarr)
dest_group = zarr.open_group('data.zarr', mode='w')

# Direct copy (preserves chunks)
zarr.copy(source, dest_group, name='temperature')
```

**With rechunking:**

```python
import zarr

# Copy with new chunk sizes
zarr.copy(
    source,
    dest_group,
    name='temperature',
    chunks=(30, 90, 180),  # New chunks
    compressor=zarr.Blosc(cname='zstd', clevel=3)
)
```

**Parallel copy with Dask:**

```python
import zarr
import dask.array as da

# Wrap source as Dask array
source_da = da.from_array(source, chunks=(30, 90, 180))

# Create destination
dest = zarr.open_array(
    'data.zarr/temperature',
    mode='w',
    shape=source.shape,
    chunks=(30, 90, 180),
    dtype=source.dtype,
    compressor=zarr.Blosc(cname='zstd', clevel=3)
)

# Parallel copy
da.to_zarr(source_da, dest, overwrite=True)
```

### 3. Xarray Migration (High-Level)

**NetCDF → Zarr:**

```python
import xarray as xr
import zarr

# Read NetCDF
ds = xr.open_dataset('climate_data.nc')

# Define encoding (chunking + compression)
encoding = {
    'temperature': {
        'chunks': (30, 90, 180),
        'compressor': zarr.Blosc(cname='zstd', clevel=3),
        'dtype': 'float32'
    }
}

# Write to Zarr (preserves metadata)
ds.to_zarr('climate_data.zarr', mode='w', encoding=encoding)

# Consolidate metadata for cloud
zarr.consolidate_metadata('climate_data.zarr')
```

**Multi-file NetCDF → Single Zarr:**

```python
import xarray as xr

# Open multiple files (time series)
ds = xr.open_mfdataset(
    'data_*.nc',
    combine='by_coords',
    parallel=True
)

# Merge and write to Zarr
ds.to_zarr('consolidated.zarr', mode='w')
```

### 4. VirtualiZarr - Zero-Copy Virtual Datasets

**What is VirtualiZarr?**

VirtualiZarr creates a **virtual Zarr dataset** that references data in existing HDF5/NetCDF files without copying. The virtual Zarr store contains only metadata pointing to byte ranges in source files.

**When to use:**
- Immediate access to HDF5/NetCDF data via Zarr API
- Source files too large to duplicate
- Read-only access sufficient
- Data exploration before full migration

**Basic workflow:**

```python
from virtualizarr import open_virtual_dataset
import xarray as xr

# Create virtual Zarr from HDF5/NetCDF (no data copied)
vds = open_virtual_dataset(
    'large_dataset.nc',
    indexes={}  # Disable index loading for performance
)

# Write virtual references to Zarr
vds.virtualize.to_zarr('virtual.zarr')

# Read via Zarr API (data fetched from original file)
ds = xr.open_zarr('virtual.zarr')

# Use like normal Zarr
subset = ds['temperature'].isel(time=0)
```

**Multi-file virtual aggregation:**

```python
from virtualizarr import open_virtual_dataset
import xarray as xr

# Create virtual datasets for each file
virtual_datasets = []
for file in ['data_2020.nc', 'data_2021.nc', 'data_2022.nc']:
    vds = open_virtual_dataset(file, indexes={})
    virtual_datasets.append(vds)

# Concatenate into single virtual dataset
combined = xr.concat(virtual_datasets, dim='time')

# Write virtual references
combined.virtualize.to_zarr('combined_virtual.zarr')

# Access as unified dataset
ds = xr.open_zarr('combined_virtual.zarr')
print(f"Total time steps: {ds.sizes['time']}")
```

**Limitations:**
- **Read-only**: Cannot modify data
- **Source dependencies**: Original files must remain accessible
- **Performance**: Reads slower than native Zarr (file seeks)

**When to materialize:**

After exploration, convert to native Zarr for production:

```python
import xarray as xr

# Read virtual
vds = xr.open_zarr('virtual.zarr')

# Materialize to native Zarr
vds.to_zarr('materialized.zarr', mode='w')
```

### 5. Icechunk - ACID-Compliant Versioned Zarr

**What is Icechunk?**

Icechunk provides **Git-like version control** for Zarr stores with ACID (Atomic, Consistent, Isolated, Durable) guarantees. Each write creates a new snapshot; rollback and branching supported.

**When to use:**
- Need transactional writes (all-or-nothing)
- Want version history and rollback
- Collaborative workflows (multiple writers)
- Data lineage tracking

**Installation:**

```bash
pixi add icechunk
```

**Basic workflow:**

```python
from icechunk import IcechunkStore
import zarr

# Create Icechunk store (local storage)
store = IcechunkStore.create(storage='local', path='icechunk_data')

# Open root group
root = zarr.open_group(store, mode='w')

# Write data in transaction
root.create_array(
    'temperature',
    shape=(365, 180, 360),
    chunks=(30, 90, 180),
    dtype='f4'
)

# Commit transaction (creates snapshot)
store.commit('Initial temperature data')

# View commit history
for commit in store.log():
    print(f"{commit.id[:8]}: {commit.message}")
```

**Versioning and rollback:**

```python
from icechunk import IcechunkStore
import zarr

store = IcechunkStore.open(storage='local', path='icechunk_data')
root = zarr.open_group(store)

# Make changes
root['temperature'][:10, :, :] = new_data

# Commit
store.commit('Updated first 10 days')

# Rollback to previous version
previous_commit = store.log()[1].id  # Second-to-last commit
store.checkout(previous_commit)

# Data reverted
print(root['temperature'][:10, :, :].mean())  # Original data
```

**Migration to Icechunk:**

```python
import xarray as xr
from icechunk import IcechunkStore
import zarr

# Read existing Zarr
ds = xr.open_zarr('standard.zarr')

# Create Icechunk store
ic_store = IcechunkStore.create(storage='s3', bucket='my-bucket', path='icechunk_data')

# Write to Icechunk
ds.to_zarr(ic_store, mode='w')

# Commit
ic_store.commit('Migrated from standard Zarr')
```

### 6. Migration Validation

**Validation checklist:**

1. **Shape and dtype**: Arrays match source
2. **Data integrity**: Values match (checksums, sampling)
3. **Metadata**: Attributes preserved
4. **Performance**: Chunk access faster than source
5. **Compression**: Storage size reasonable
6. **Completeness**: All variables/groups migrated

**Validation script:**

```python
import h5py
import zarr
import numpy as np

def validate_migration(h5_path, zarr_path, variable):
    """Validate HDF5 → Zarr migration for a variable."""

    # Open both
    h5f = h5py.File(h5_path, 'r')
    z = zarr.open(zarr_path, mode='r')

    h5_arr = h5f[variable]
    z_arr = z[variable]

    # Check 1: Shape
    assert h5_arr.shape == z_arr.shape, \
        f"Shape mismatch: {h5_arr.shape} vs {z_arr.shape}"

    # Check 2: Dtype
    assert h5_arr.dtype == z_arr.dtype, \
        f"Dtype mismatch: {h5_arr.dtype} vs {z_arr.dtype}"

    # Check 3: Sample data (random slices)
    np.random.seed(42)
    for _ in range(10):
        idx = tuple(np.random.randint(0, s) for s in h5_arr.shape)
        h5_val = h5_arr[idx]
        z_val = z_arr[idx]

        np.testing.assert_allclose(
            h5_val, z_val,
            rtol=1e-5,
            err_msg=f"Value mismatch at {idx}"
        )

    # Check 4: Metadata
    for key in h5_arr.attrs:
        assert key in z_arr.attrs, f"Missing attribute: {key}"

    print(f"✓ Validation passed for '{variable}'")
    print(f"  Shape: {z_arr.shape}")
    print(f"  Dtype: {z_arr.dtype}")
    print(f"  Chunks: {z_arr.chunks}")

    h5f.close()

# Usage
validate_migration('data.h5', 'data.zarr', 'temperature')
```

## Patterns

See [references/PATTERNS.md](references/PATTERNS.md) for detailed migration patterns including:
- Direct copy (HDF5 → Zarr)
- Rechunking during migration
- Parallel migration with Dask
- Virtual datasets with VirtualiZarr
- Incremental migration for large datasets
- Post-migration validation

## Real-World Examples

See [references/EXAMPLES.md](references/EXAMPLES.md) for complete examples including:
- HDF5 climate model → Zarr
- NetCDF time series → Zarr with rechunking
- Multi-file NetCDF consolidation
- Local → S3 cloud migration
- VirtualiZarr multi-file aggregation
- Icechunk versioned store setup

## Common Issues and Solutions

See [references/COMMON_ISSUES.md](references/COMMON_ISSUES.md) for solutions to:
- Memory errors during migration
- Metadata loss or corruption
- Encoding/compression problems
- Performance degradation
- Validation failures
- Cloud upload errors

## Production Migration Template

See [assets/migration-template.py](assets/migration-template.py) for a production-ready migration script with:
- Configurable source/destination formats
- Rechunking and compression options
- Progress monitoring
- Validation and rollback
- Error handling and logging
- Cloud storage support

## Best Practices Checklist

### Planning
- Profile source data (size, chunks, access patterns)
- Estimate migration time and storage requirements
- Choose rechunking strategy for target workload
- Test migration on subset before full migration
- Plan validation strategy

### Execution
- Use appropriate tool (zarr.copy vs Xarray vs VirtualiZarr)
- Monitor memory usage during migration
- Enable compression (Blosc/Zstd recommended)
- Consolidate metadata after migration
- Validate data integrity

### Post-Migration
- Run validation suite (shape, dtype, values, metadata)
- Benchmark performance vs source format
- Document migration details (date, tool, settings)
- Keep source data until validation complete
- Update access code to use Zarr paths

### Cloud Migration
- Test with small dataset first
- Use consolidated metadata
- Verify cloud credentials and permissions
- Monitor transfer costs and time
- Validate data after upload

## Resources and References

### Official Documentation
- **Zarr Python**: https://zarr.readthedocs.io/
- **Xarray**: https://docs.xarray.dev/
- **VirtualiZarr**: https://virtualizarr.readthedocs.io/
- **Icechunk**: https://icechunk.io/
- **Kerchunk**: https://fsspec.github.io/kerchunk/

### Migration Guides
- **Zarr Migration Guide**: https://zarr.readthedocs.io/en/stable/tutorial.html#migrating-data
- **Pangeo Guide**: https://pangeo.io/

### Related Tools
- **h5py**: https://docs.h5py.org/ (HDF5 for Python)
- **netCDF4**: https://unidata.github.io/netcdf4-python/
- **fsspec**: https://filesystem-spec.readthedocs.io/ (cloud storage)

### Research and Background
- Nguyen et al. 2023: "Zarr: Cloud-optimized scientific data storage"
- Icechunk paper: ACID guarantees for Zarr
