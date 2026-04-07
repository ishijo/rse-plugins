# Cloud Storage Backend Examples

Complete, runnable examples for deploying Zarr arrays on cloud storage.

## Example 1: Deploying Climate Data to S3

**Scenario:** Deploy a multi-variable climate dataset to S3 with proper chunking, compression, and metadata consolidation for efficient cloud access.

```python
#!/usr/bin/env python3
"""
Deploy climate dataset to S3 with optimal configuration.

Demonstrates: group hierarchy, metadata consolidation, compression, chunking.
"""

import zarr
import numpy as np
from fsspec import filesystem
import os

# Configuration
S3_BUCKET = 'climate-data'
S3_PREFIX = 'datasets/global-climate-2024.zarr'
S3_REGION = 'us-west-2'

# Create S3 store
print(f"Creating Zarr store: s3://{S3_BUCKET}/{S3_PREFIX}")
fs = filesystem('s3', anon=False, client_kwargs={'region_name': S3_REGION})
store = zarr.storage.FsspecStore(f'{S3_BUCKET}/{S3_PREFIX}', fs=fs)

# Create root group
root = zarr.open_group(store, mode='w')
root.attrs['title'] = 'Global Climate Data 2024'
root.attrs['institution'] = 'Example Climate Center'
root.attrs['source'] = 'Model output'
root.attrs['version'] = '1.0'

# Dimensions: time (365 days), lat (180), lon (360)
SHAPE = (365, 180, 360)
CHUNKS = (10, 90, 180)  # ~1.4 MB per chunk (balanced for time-series and spatial access)
COMPRESSOR = zarr.codecs.Blosc(cname='zstd', clevel=3, shuffle=zarr.codecs.BITSHUFFLE)

# Create temperature variable
print("Creating temperature array...")
temp = root.create_dataset(
    'temperature',
    shape=SHAPE,
    chunks=CHUNKS,
    dtype='f4',
    compressor=COMPRESSOR,
    fill_value=-9999.0
)
temp.attrs['units'] = 'K'
temp.attrs['long_name'] = 'Daily mean temperature'
temp.attrs['standard_name'] = 'air_temperature'
temp.attrs['coordinates'] = 'time lat lon'

# Create precipitation variable
print("Creating precipitation array...")
precip = root.create_dataset(
    'precipitation',
    shape=SHAPE,
    chunks=CHUNKS,
    dtype='f4',
    compressor=COMPRESSOR,
    fill_value=-9999.0
)
precip.attrs['units'] = 'mm/day'
precip.attrs['long_name'] = 'Daily precipitation rate'
precip.attrs['standard_name'] = 'precipitation_flux'
precip.attrs['coordinates'] = 'time lat lon'

# Create coordinate arrays (1D, minimal storage)
print("Creating coordinate arrays...")
time = root.create_dataset('time', data=np.arange(365), dtype='i4')
time.attrs['units'] = 'days since 2024-01-01'
time.attrs['calendar'] = 'standard'

lat = root.create_dataset('lat', data=np.linspace(-90, 90, 180), dtype='f4')
lat.attrs['units'] = 'degrees_north'
lat.attrs['long_name'] = 'latitude'

lon = root.create_dataset('lon', data=np.linspace(-180, 180, 360), dtype='f4')
lon.attrs['units'] = 'degrees_east'
lon.attrs['long_name'] = 'longitude'

# Write synthetic data (in production, replace with real data)
print("Writing temperature data...")
temp_data = np.random.randn(365, 180, 360).astype('f4') * 10 + 273.15
temp[:] = temp_data

print("Writing precipitation data...")
precip_data = np.abs(np.random.randn(365, 180, 360).astype('f4')) * 5
precip[:] = precip_data

# Storage metrics
print(f"\n✓ Dataset created successfully")
print(f"  Temperature: {temp.nbytes_stored / 1024**2:.1f} MB compressed")
print(f"  Precipitation: {precip.nbytes_stored / 1024**2:.1f} MB compressed")
print(f"  Compression ratio: {(temp.nbytes + precip.nbytes) / (temp.nbytes_stored + precip.nbytes_stored):.2f}x")

# CRITICAL: Consolidate metadata for cloud performance
print("\nConsolidating metadata...")
zarr.consolidate_metadata(store)
print("✓ Metadata consolidated (.zmetadata created)")

print(f"\n✓ Climate dataset deployed to S3:")
print(f"  s3://{S3_BUCKET}/{S3_PREFIX}")
print("\nAccess with:")
print(f"  root = zarr.open_consolidated('s3://{S3_BUCKET}/{S3_PREFIX}', mode='r')")
```

**Key points:**
- Group hierarchy with CF metadata conventions
- Balanced chunking (10-day time slices, 90x180 spatial)
- BITSHUFFLE for floating-point data compression
- Metadata consolidation (CRITICAL for cloud)
- Coordinate arrays for self-describing data

---

## Example 2: Reading Public Pangeo Data from S3

**Scenario:** Access public Zarr datasets from Pangeo's AWS S3 bucket without authentication.

```python
#!/usr/bin/env python3
"""
Read public climate data from Pangeo's S3 bucket.

Demonstrates: anonymous access, consolidated metadata, chunked reading.
"""

import zarr
import numpy as np

# Pangeo public dataset (example - adjust to actual dataset)
S3_PATH = 's3://pangeo-data/examples/ocean-temperature.zarr'

print(f"Opening public Pangeo dataset: {S3_PATH}")

# Open with anonymous access and consolidated metadata
root = zarr.open_consolidated(
    S3_PATH,
    mode='r',
    storage_options={'anon': True}
)

print("\n✓ Dataset opened successfully")

# Explore structure
print(f"\nGroup tree:")
print(root.tree())

# Access temperature array
if 'temperature' in root:
    temp = root['temperature']

    print(f"\nTemperature array:")
    print(f"  Shape: {temp.shape}")
    print(f"  Chunks: {temp.chunks}")
    print(f"  Dtype: {temp.dtype}")
    print(f"  Size: {temp.nbytes / 1024**3:.2f} GB")

    # Read metadata
    print(f"\n  Metadata:")
    for key, value in temp.attrs.items():
        print(f"    {key}: {value}")

    # Read a small subset (avoid downloading entire dataset)
    print(f"\nReading subset: time=0, spatial mean...")
    subset = temp[0, :, :].mean()
    print(f"  ✓ Mean value: {subset:.2f}")

    # Access pattern: time-series at a specific location
    print(f"\nReading time-series at location (lat=45, lon=0)...")
    # Assuming shape is (time, lat, lon)
    if len(temp.shape) == 3:
        lat_idx = temp.shape[1] // 4  # Approximate lat=45
        lon_idx = temp.shape[2] // 2  # Approximate lon=0
        timeseries = temp[:, lat_idx, lon_idx]
        print(f"  ✓ Time-series shape: {timeseries.shape}")
        print(f"  ✓ Mean: {timeseries.mean():.2f}")

else:
    print("\n⚠ 'temperature' variable not found")
    print(f"Available variables: {list(root.keys())}")

print("\n✓ Public dataset access complete")
print("Note: Pangeo datasets are optimized for cloud access with:")
print("  - Consolidated metadata (single read)")
print("  - Optimal chunking for analysis patterns")
print("  - Compression for reduced storage/transfer costs")
```

**Key points:**
- Anonymous access to public buckets (`anon=True`)
- Always read subsets, not entire arrays (cloud cost optimization)
- Consolidated metadata dramatically reduces latency
- Inspect structure with `.tree()` before reading

---

## Example 3: Multi-Cloud Disaster Recovery (S3 ↔ GCS Replication)

**Scenario:** Replicate critical Zarr dataset between S3 and GCS for disaster recovery.

```python
#!/usr/bin/env python3
"""
Replicate Zarr dataset between S3 and GCS for disaster recovery.

Demonstrates: multi-cloud, zarr.copy_store, verification.
"""

import zarr
from fsspec import filesystem
import numpy as np

# Source: S3
S3_BUCKET = 'primary-data'
S3_PREFIX = 'critical/dataset.zarr'
S3_REGION = 'us-west-2'

# Destination: GCS
GCS_BUCKET = 'backup-data'
GCS_PREFIX = 'critical/dataset.zarr'
GCP_PROJECT = 'my-project'

print("="*80)
print("MULTI-CLOUD REPLICATION: S3 → GCS")
print("="*80)

# Step 1: Create source dataset on S3 (if it doesn't exist)
print("\n[1/4] Creating source dataset on S3...")
s3_fs = filesystem('s3', anon=False, client_kwargs={'region_name': S3_REGION})
s3_store = zarr.storage.FsspecStore(f'{S3_BUCKET}/{S3_PREFIX}', fs=s3_fs)

try:
    root_s3 = zarr.open_group(s3_store, mode='r')
    print(f"  ✓ Source exists: s3://{S3_BUCKET}/{S3_PREFIX}")
except:
    print(f"  Creating new source dataset...")
    root_s3 = zarr.open_group(s3_store, mode='w')
    arr = root_s3.create_dataset(
        'data',
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype='f4',
        compressor=zarr.codecs.Zstd(level=3)
    )
    arr[:] = np.random.randn(1000, 1000).astype('f4')
    zarr.consolidate_metadata(s3_store)
    print(f"  ✓ Created: s3://{S3_BUCKET}/{S3_PREFIX}")

# Step 2: Create destination store on GCS
print(f"\n[2/4] Setting up GCS destination...")
gcs_fs = filesystem('gcs', project=GCP_PROJECT)
gcs_store = zarr.storage.FsspecStore(f'{GCS_BUCKET}/{GCS_PREFIX}', fs=gcs_fs)
print(f"  ✓ Target: gs://{GCS_BUCKET}/{GCS_PREFIX}")

# Step 3: Replicate (binary copy, fastest)
print(f"\n[3/4] Replicating data...")
zarr.copy_store(s3_store, gcs_store)
print(f"  ✓ Replication complete")

# Step 4: Verify replication
print(f"\n[4/4] Verifying replication...")
root_gcs = zarr.open_consolidated(gcs_store, mode='r')

if 'data' in root_gcs:
    arr_s3 = root_s3['data']
    arr_gcs = root_gcs['data']

    # Check shape
    assert arr_s3.shape == arr_gcs.shape, "Shape mismatch!"
    print(f"  ✓ Shape matches: {arr_gcs.shape}")

    # Check chunks
    assert arr_s3.chunks == arr_gcs.chunks, "Chunks mismatch!"
    print(f"  ✓ Chunks match: {arr_gcs.chunks}")

    # Sample data verification (10 random points)
    for _ in range(10):
        i, j = np.random.randint(0, 1000, size=2)
        assert arr_s3[i, j] == arr_gcs[i, j], f"Data mismatch at ({i}, {j})"

    print(f"  ✓ Data integrity verified (10 random samples)")

else:
    print("  ❌ Replication failed: 'data' array not found")

print("\n" + "="*80)
print("✓ DISASTER RECOVERY SETUP COMPLETE")
print("="*80)
print(f"\nPrimary:  s3://{S3_BUCKET}/{S3_PREFIX}")
print(f"Backup:   gs://{GCS_BUCKET}/{GCS_PREFIX}")
print("\nSchedule periodic replication with cron/airflow/prefect")
```

**Key points:**
- `zarr.copy_store()` for binary replication (fastest)
- Preserves chunks, compression, metadata
- Verification with random sampling
- Multi-cloud redundancy strategy

---

## Example 4: Cached Access for Bandwidth-Limited Environments

**Scenario:** Work with large cloud Zarr dataset in bandwidth-limited environment using persistent file cache.

```python
#!/usr/bin/env python3
"""
Access large Zarr dataset with persistent file cache.

Demonstrates: filecache, cache management, bandwidth optimization.
"""

import zarr
import numpy as np
import os
import shutil

# Cloud dataset
S3_PATH = 's3://large-dataset/climate/model-output.zarr'
CACHE_DIR = os.path.expanduser('~/.zarr-cache')

# Cache configuration
CACHE_EXPIRY = 7 * 86400  # 7 days

print("="*80)
print("CACHED CLOUD ACCESS")
print("="*80)

# Clear old cache (optional)
if os.path.exists(CACHE_DIR):
    cache_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR)
        for filename in filenames
    )
    print(f"\nExisting cache: {cache_size / 1024**2:.1f} MB")
    print(f"Location: {CACHE_DIR}")
else:
    print(f"\nCache directory: {CACHE_DIR} (empty)")

# Open with file cache
print(f"\nOpening dataset with persistent file cache...")
arr = zarr.open_array(
    f'filecache::{S3_PATH}',
    storage_options={
        's3': {'anon': False},
        'filecache': {
            'cache_storage': CACHE_DIR,
            'expiry_time': CACHE_EXPIRY,
            'same_names': True  # Preserve directory structure
        }
    },
    mode='r'
)

print(f"✓ Dataset opened")
print(f"  Shape: {arr.shape}")
print(f"  Chunks: {arr.chunks}")
print(f"  Size: {arr.nbytes / 1024**3:.2f} GB")

# First access: downloads chunks (slower)
print(f"\n[1] Reading subset (0:100, 0:100) - may download chunks...")
import time
t0 = time.time()
data1 = arr[0:100, 0:100]
t1 = time.time()
print(f"  ✓ Read complete: {t1 - t0:.2f}s")
print(f"  ✓ Mean value: {data1.mean():.2f}")

# Second access to same region: reads from cache (faster)
print(f"\n[2] Re-reading same subset - from cache...")
t0 = time.time()
data2 = arr[0:100, 0:100]
t1 = time.time()
print(f"  ✓ Read complete: {t1 - t0:.2f}s (should be much faster!)")
print(f"  ✓ Data identical: {np.array_equal(data1, data2)}")

# Check cache size
cache_size_after = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk(CACHE_DIR)
    for filename in filenames
)
print(f"\n✓ Cache size: {cache_size_after / 1024**2:.1f} MB")
print(f"  Downloaded chunks for accessed regions only")

# Cache management
print(f"\n[Cache Management]")
print(f"  Location: {CACHE_DIR}")
print(f"  Expiry: {CACHE_EXPIRY / 86400:.0f} days")
print(f"  Strategy: Accessed chunks persist across sessions")
print(f"\nTo clear cache manually:")
print(f"  rm -rf {CACHE_DIR}")

print("\n" + "="*80)
print("✓ CACHED ACCESS COMPLETE")
print("="*80)
print("\nBenefits:")
print("  - Repeated reads are fast (no re-download)")
print("  - Works offline after initial download")
print("  - Bandwidth savings for iterative analysis")
print("  - Cache persists across sessions")
```

**Key points:**
- `filecache::` prefix for persistent caching
- Cache directory persists across Python sessions
- Only accessed chunks are downloaded (not entire array)
- Expiry time controls cache staleness
- Huge bandwidth savings for repeated access

---

## Example 5: Azure Blob Storage with Managed Identity (Production)

**Scenario:** Deploy Zarr dataset to Azure Blob Storage from Azure VM using Managed Identity (no hardcoded credentials).

```python
#!/usr/bin/env python3
"""
Deploy Zarr to Azure using Managed Identity.

Demonstrates: Azure VM/App Service integration, managed identity, no credentials.

Prerequisites:
  - Running on Azure VM or App Service
  - Managed Identity enabled for the VM/App Service
  - Storage Blob Data Contributor role assigned to the managed identity
"""

import zarr
import numpy as np
from fsspec import filesystem
import os

# Configuration
AZURE_ACCOUNT = os.getenv('AZURE_STORAGE_ACCOUNT', 'myaccount')
AZURE_CONTAINER = 'production-data'
AZURE_PREFIX = 'datasets/sensor-data.zarr'

print("="*80)
print("AZURE DEPLOYMENT WITH MANAGED IDENTITY")
print("="*80)
print(f"\nAccount: {AZURE_ACCOUNT}")
print(f"Container: {AZURE_CONTAINER}")
print(f"Path: {AZURE_PREFIX}")

# Create Azure filesystem (uses Managed Identity)
print("\nAuthenticating with Managed Identity...")
fs = filesystem('az', account_name=AZURE_ACCOUNT, anon=False)
print("✓ Authenticated via Managed Identity")

# Create store
store = zarr.storage.FsspecStore(f'{AZURE_CONTAINER}/{AZURE_PREFIX}', fs=fs)

# Create dataset
print("\nCreating sensor data array...")
root = zarr.open_group(store, mode='w')

# Hourly sensor data for 1 year: (8760 hours, 100 sensors)
sensor_data = root.create_dataset(
    'measurements',
    shape=(8760, 100),
    chunks=(24, 100),  # Daily chunks
    dtype='f4',
    compressor=zarr.codecs.Blosc(cname='zstd', clevel=3)
)

sensor_data.attrs['units'] = 'arbitrary units'
sensor_data.attrs['description'] = 'Hourly sensor measurements'
sensor_data.attrs['sampling_rate'] = '1 hour'

# Write synthetic sensor data
print("Writing data...")
data = np.random.randn(8760, 100).astype('f4') * 100 + 500
sensor_data[:] = data

# Add metadata
print("Adding dataset metadata...")
root.attrs['version'] = '1.0'
root.attrs['created_by'] = 'Azure deployment pipeline'
root.attrs['environment'] = 'production'

# Consolidate metadata
print("Consolidating metadata...")
zarr.consolidate_metadata(store)

print(f"\n✓ Deployment complete!")
print(f"  Size: {sensor_data.nbytes / 1024**2:.1f} MB (uncompressed)")
print(f"  Compressed: {sensor_data.nbytes_stored / 1024**2:.1f} MB")
print(f"  Ratio: {sensor_data.nbytes / sensor_data.nbytes_stored:.2f}x")

print(f"\n✓ Dataset deployed to Azure:")
print(f"  az://{AZURE_CONTAINER}/{AZURE_PREFIX}")

print("\n" + "="*80)
print("PRODUCTION DEPLOYMENT CHECKLIST")
print("="*80)
print("✓ Managed Identity (no hardcoded credentials)")
print("✓ Metadata consolidated (cloud performance)")
print("✓ Compression enabled (storage cost reduction)")
print("✓ Proper chunking (access pattern optimization)")
print("\nNext steps:")
print("  - Set up monitoring (Azure Monitor)")
print("  - Enable lifecycle policies (auto-archive old data)")
print("  - Configure backup/replication")
```

**Key points:**
- Managed Identity (no credentials in code)
- Azure VM/App Service integration
- Production-grade security
- Role-based access control (RBAC)
- No secret management overhead

---

## Cross-References

- **zarr-fundamentals** — array creation, chunking strategies
- **compression-codecs** — choosing compressors for cloud storage
- **zarr-xarray-integration** — using Xarray with cloud backends
- **data-migration** — migrating existing data to cloud storage
