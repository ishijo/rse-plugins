# Xarray-Zarr Integration Patterns

Reusable code patterns for using Zarr as an Xarray storage backend.

## Pattern 1: Basic Read-Write Cycle

**When to use:** Simple local Zarr dataset with Xarray.

```python
import xarray as xr
import numpy as np

# Create Xarray Dataset
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360)),
    'precipitation': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360))
}, coords={
    'time': range(365),
    'lat': np.linspace(-90, 90, 180),
    'lon': np.linspace(-180, 180, 360)
})

# Add CF metadata
ds['temperature'].attrs['units'] = 'K'
ds['temperature'].attrs['long_name'] = 'Temperature'

# Write to Zarr
ds.to_zarr('dataset.zarr', mode='w')

# Read back
ds_read = xr.open_zarr('dataset.zarr')

print(ds_read['temperature'].mean().values)
```

**Key points:**
- `to_zarr()` for writing
- `open_zarr()` for reading
- Metadata preserved automatically
- No need for explicit chunking (Xarray uses sensible defaults)

---

## Pattern 2: Cloud Read with Consolidated Metadata

**When to use:** Reading Zarr datasets from S3, GCS, or Azure.

```python
import xarray as xr

# S3 with consolidated metadata and credentials
ds = xr.open_zarr(
    's3://bucket/climate-dataset.zarr',
    consolidated=True,  # CRITICAL: Single metadata read
    chunks={},          # Preserve Zarr chunk structure
    storage_options={
        'anon': False,  # Use credentials
        'client_kwargs': {'region_name': 'us-west-2'}
    }
)

# Access data (lazy loading)
temp = ds['temperature']

# Compute subset (only downloads required chunks)
subset = temp.isel(time=slice(0, 10)).compute()

print(f"First 10 days, mean temp: {subset.mean().values:.2f} K")
```

**Key points:**
- `consolidated=True` → 1 network request vs 100+
- `chunks={}` → Preserves optimized Zarr chunks
- `storage_options` → Authentication and configuration
- `.compute()` triggers actual data download

---

## Pattern 3: Time-Series Append Workflow

**When to use:** Incremental data ingestion (daily updates, real-time feeds).

```python
import xarray as xr
import numpy as np
import zarr

# Initial write (days 0-99)
ds_initial = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(100, 180, 360))
}, coords={'time': range(100)})

ds_initial.to_zarr('timeseries.zarr', mode='w')

# Consolidate metadata (first time)
zarr.consolidate_metadata('timeseries.zarr')

# Later: Append new data (days 100-199)
ds_new = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(100, 180, 360))
}, coords={'time': range(100, 200)})

ds_new.to_zarr('timeseries.zarr', mode='a', append_dim='time')

# CRITICAL: Re-consolidate after append
zarr.consolidate_metadata('timeseries.zarr')

# Verify
ds_full = xr.open_zarr('timeseries.zarr', consolidated=True)
print(f"Total time steps: {ds_full.sizes['time']}")  # 200
```

**Key points:**
- `mode='w'` for initial write
- `mode='a'` + `append_dim='time'` for incremental updates
- **Must re-consolidate** after each append
- Coordinates must not overlap (time=0-99, then 100-199)

**Automation:**
```python
def append_daily_data(zarr_path, new_data, time_coord):
    """Append daily data to Zarr store."""
    ds_new = xr.Dataset({'temperature': (['time', 'lat', 'lon'], new_data)},
                        coords={'time': time_coord})
    ds_new.to_zarr(zarr_path, mode='a', append_dim='time')
    zarr.consolidate_metadata(zarr_path)
```

---

## Pattern 4: Distributed Parallel Writes with Dask

**When to use:** Large datasets, distributed computation, parallel writes.

```python
import xarray as xr
import dask.array as da
import zarr

# Step 1: Create Dask-backed dataset
data = da.random.random((1000, 180, 360), chunks=(10, 90, 180))
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], data)
}, coords={'time': range(1000)})

# Step 2: Define encoding
encoding = {
    'temperature': {
        'compressor': zarr.codecs.Blosc(cname='zstd', clevel=3),
        'chunks': (10, 90, 180)
    }
}

# Step 3: Initialize Zarr store (without computing)
ds.to_zarr('large-dataset.zarr', mode='w', encoding=encoding, compute=False)

# Step 4: Parallel write with region='auto'
write_task = ds.to_zarr('large-dataset.zarr', region='auto', compute=False)
write_task.compute()  # Dask distributes chunk writes

# Step 5: Consolidate metadata
zarr.consolidate_metadata('large-dataset.zarr')

print("✓ Distributed write complete")
```

**Key points:**
- `compute=False` → Initialize structure without writing data
- `region='auto'` → Dask auto-detects regions for parallel writes
- Encoding controls compression and chunking
- Consolidate metadata **after** all writes complete

**Dask cluster version:**
```python
from dask.distributed import Client

client = Client()  # Local cluster or remote

# ... same code as above ...
# Dask scheduler coordinates parallel writes across workers

client.close()
```

---

## Pattern 5: Region-Based Updates (Spatial/Temporal)

**When to use:** Updating specific regions without rewriting entire dataset.

### Temporal Region Update

```python
import xarray as xr
import numpy as np

# Update specific time range (days 50-59)
ds_update = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(10, 180, 360))
}, coords={'time': range(50, 60)})

# Write to specific time slice
ds_update.to_zarr(
    'dataset.zarr',
    mode='a',
    region={'time': slice(50, 60)}
)

print("✓ Updated days 50-59")
```

### Spatial Region Update

```python
import xarray as xr
import numpy as np

# Update specific lat/lon box
ds_spatial = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(365, 20, 40))
}, coords={
    'time': range(365),
    'lat': range(80, 100),    # Indices 80-99
    'lon': range(100, 140)    # Indices 100-139
})

# Write to specific spatial region
ds_spatial.to_zarr(
    'dataset.zarr',
    mode='a',
    region={'lat': slice(80, 100), 'lon': slice(100, 140)}
)

print("✓ Updated spatial region (lat 80-99, lon 100-139)")
```

**Key points:**
- `region` parameter specifies which part to update
- Can update temporal, spatial, or any combination of dimensions
- More efficient than rewriting entire dataset
- Good for partial corrections or updates

---

## Pattern 6: Optimized Encoding for Different Data Types

**When to use:** Mixed data types requiring different compression strategies.

```python
import xarray as xr
import zarr
import numpy as np

# Create dataset with different data types
ds = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360).astype('f4')),
    'land_mask': (['lat', 'lon'], np.random.randint(0, 2, (180, 360)).astype('i1')),
    'station_id': (['station'], np.arange(1000).astype('i4'))
})

# Optimized encoding per variable
encoding = {
    'temperature': {
        'chunks': (10, 90, 180),
        'compressor': zarr.codecs.Blosc(
            cname='zstd',
            clevel=3,
            shuffle=zarr.codecs.BITSHUFFLE  # Best for float
        ),
        'dtype': 'float32'
    },
    'land_mask': {
        'chunks': (180, 360),  # Single chunk (static)
        'compressor': zarr.codecs.Blosc(
            cname='lz4',
            clevel=5,
            shuffle=zarr.codecs.SHUFFLE  # Best for integers
        ),
        'dtype': 'int8'
    },
    'station_id': {
        'chunks': (1000,),  # Single chunk (1D array)
        'compressor': zarr.codecs.Zstd(level=5),
        'dtype': 'int32'
    }
}

ds.to_zarr('mixed-dataset.zarr', mode='w', encoding=encoding)

print("✓ Written with optimized encoding per variable")
```

**Encoding strategies by data type:**
| Data Type | Compressor | Shuffle Mode | Reasoning |
|-----------|-----------|--------------|-----------|
| Float32/64 | Blosc+Zstd | BITSHUFFLE | Best for floating-point |
| Int8/16/32 | Blosc+LZ4 | SHUFFLE | Fast for integers |
| Boolean | Blosc+LZ4 | NOSHUFFLE | High compression ratio |
| Categorical | Delta + Blosc | SHUFFLE | Exploits repeated values |

---

## Pattern 7: Lazy Loading with Dask for Large Datasets

**When to use:** Working with datasets larger than memory.

```python
import xarray as xr

# Open with Dask chunks (lazy loading)
ds = xr.open_zarr(
    's3://bucket/large-dataset.zarr',
    consolidated=True,
    chunks='auto'  # Dask auto-chunks for computation
)

# All operations are lazy (no data loaded yet)
temp = ds['temperature']
mean_temp = temp.mean(dim='time')  # Lazy computation graph

# Compute only when needed
result = mean_temp.compute()  # NOW data is loaded and computed

print(f"Time-mean temperature: {result.values}")
```

**Partial computation (efficient):**
```python
import xarray as xr

ds = xr.open_zarr('large-dataset.zarr', chunks='auto')

# Select subset BEFORE computing (avoids loading full dataset)
january = ds.sel(time=slice('2024-01-01', '2024-01-31'))
january_mean = january.mean(dim='time').compute()  # Only loads January

print("✓ Computed mean for January (only loaded 1 month)")
```

**Key points:**
- `chunks='auto'` → Dask lazy loading
- `.compute()` triggers actual computation
- Select/slice BEFORE compute to minimize data transfer
- Dask builds computation graph before executing

---

## Pattern 8: Multi-Variable Append with Encoding

**When to use:** Adding new variables to existing store with custom compression.

```python
import xarray as xr
import zarr
import numpy as np

# Existing store has 'temperature'
# Add 'humidity' with different encoding

ds_new = xr.Dataset({
    'humidity': (['time', 'lat', 'lon'], np.random.randn(365, 180, 360))
}, coords={'time': range(365)})

# Encoding for new variable
encoding = {
    'humidity': {
        'chunks': (10, 90, 180),
        'compressor': zarr.codecs.Blosc(cname='lz4', clevel=5),  # Faster for humidity
        'dtype': 'float32'
    }
}

# Append new variable
ds_new.to_zarr('dataset.zarr', mode='a', encoding=encoding)

# Re-consolidate metadata
zarr.consolidate_metadata('dataset.zarr')

# Verify
ds_full = xr.open_zarr('dataset.zarr', consolidated=True)
print(f"Variables: {list(ds_full.data_vars)}")  # ['temperature', 'humidity']
```

**Key points:**
- `mode='a'` appends new variables (doesn't overwrite)
- Encoding can differ from existing variables
- **Must re-consolidate** after adding variables
- Coordinates must match existing store

---

## Cross-References

- **xarray-for-multidimensional-data** (scientific-domain-applications) — Xarray fundamentals
- **zarr-fundamentals** — chunking strategies, Zarr basics
- **cloud-storage-backends** — storage_options configuration
- **compression-codecs** — choosing compressors for encoding
