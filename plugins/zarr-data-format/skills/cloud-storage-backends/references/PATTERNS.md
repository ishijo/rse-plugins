

# Cloud Storage Backend Patterns

Reusable code patterns for Zarr cloud storage backends across S3, GCS, and Azure.

## Pattern 1: Simple Cloud URL Access (Quick Start)

**When to use:** Quick prototyping, minimal configuration needed, using default credentials.

**S3:**
```python
import zarr

# Automatic credential discovery (env vars, IAM role, or AWS config)
arr = zarr.open_array('s3://bucket/path/array.zarr', mode='r')
data = arr[:]
```

**GCS:**
```python
import zarr

# Uses Application Default Credentials (gcloud auth application-default login)
arr = zarr.open_array('gs://bucket/path/array.zarr', mode='r')
data = arr[:]
```

**Azure:**
```python
import zarr

# Uses AZURE_STORAGE_CONNECTION_STRING environment variable
arr = zarr.open_array('az://container/path/array.zarr', mode='r')
data = arr[:]
```

**Key points:**
- Simplest pattern — single line of code
- Relies on environment variables or default credential chains
- Read-only is safest for cloud data
- No caching or custom configuration

---

## Pattern 2: Explicit FsspecStore (Full Control)

**When to use:** Custom authentication, non-default regions, advanced configuration.

**S3 with Custom Configuration:**
```python
import zarr
from fsspec import filesystem

# Create S3 filesystem with custom config
fs = filesystem(
    's3',
    key='ACCESS_KEY',
    secret='SECRET_KEY',
    client_kwargs={
        'region_name': 'us-west-2',
        'endpoint_url': 'https://s3.us-west-2.amazonaws.com'
    }
)

# Create store
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)

# Open array
arr = zarr.open_array(store, mode='r')
```

**GCS with Service Account:**
```python
import zarr
from fsspec import filesystem

# Create GCS filesystem with service account
fs = filesystem(
    'gcs',
    project='my-project',
    token='/path/to/service-account.json'
)

store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Azure with SAS Token:**
```python
import zarr
from fsspec import filesystem

# Create Azure filesystem with SAS token
fs = filesystem(
    'az',
    account_name='myaccount',
    sas_token='?sv=2021-06-08&ss=...'
)

store = zarr.storage.FsspecStore('container/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Benefits:**
- Full control over authentication
- Custom endpoints (e.g., MinIO, LocalStack)
- Advanced networking configuration
- Reusable filesystem objects

---

## Pattern 3: Metadata Consolidation (Cloud Performance)

**When to use:** ALWAYS for cloud storage with multiple arrays. Reduces latency by 10-100x.

**Write with Consolidation:**
```python
import zarr

# Create cloud store
store = zarr.storage.FsspecStore('s3://bucket/dataset.zarr')

# Create group with multiple arrays
root = zarr.open_group(store, mode='w')
root.create_dataset('temperature', shape=(365, 100, 100), chunks=(10, 100, 100))
root.create_dataset('pressure', shape=(365, 100, 100), chunks=(10, 100, 100))
root.create_dataset('humidity', shape=(365, 100, 100), chunks=(10, 100, 100))

# Add group metadata
root.attrs['description'] = 'Daily climate data'
root.attrs['year'] = 2024

# CRITICAL: Consolidate metadata before sharing
zarr.consolidate_metadata(store)
```

**Read with Consolidation:**
```python
import zarr

# Open with consolidated metadata (single network request)
root = zarr.open_consolidated('s3://bucket/dataset.zarr', mode='r')

# Access arrays (no additional metadata fetches)
temp = root['temperature']
press = root['pressure']
humid = root['humidity']

# Metadata already loaded
print(root.attrs['description'])
```

**Re-consolidate After Structural Changes:**
```python
import zarr

store = zarr.storage.FsspecStore('s3://bucket/dataset.zarr')

# Add new array
root = zarr.open_group(store, mode='a')
root.create_dataset('wind_speed', shape=(365, 100, 100), chunks=(10, 100, 100))

# MUST re-consolidate (structural change)
zarr.consolidate_metadata(store)
```

**When NOT to re-consolidate:**
- Data updates (arr[:] = new_data) — no structural change
- Reading data — read-only operation

---

## Pattern 4: Persistent Caching (Repeated Access)

**When to use:** Repeated reads, limited bandwidth, same dataset accessed multiple times.

**File Cache (Persistent Across Sessions):**
```python
import zarr

arr = zarr.open_array(
    'filecache::s3://bucket/path/array.zarr',
    storage_options={
        's3': {'anon': False},
        'filecache': {
            'cache_storage': '/home/user/.zarr-cache',
            'expiry_time': 7 * 86400,  # 7 days in seconds
            'same_names': True          # Preserve directory structure
        }
    },
    mode='r'
)

# First access: downloads chunks to /home/user/.zarr-cache
data1 = arr[0:100, :]

# Subsequent accesses: reads from local cache (fast!)
data2 = arr[100:200, :]
```

**Simple Cache (In-Memory/Temp):**
```python
import zarr

arr = zarr.open_array(
    'simplecache::gs://bucket/path/array.zarr',
    storage_options={
        'gcs': {'project': 'my-project'},
        'simplecache': {
            'cache_storage': '/tmp/zarr-cache'
        }
    },
    mode='r'
)

# Cache lifetime: single session
data = arr[:]
```

**Block Cache (Partial Chunk Reads):**
```python
import zarr

arr = zarr.open_array(
    'blockcache::az://container/path/array.zarr',
    storage_options={
        'az': {'account_name': 'myaccount'},
        'blockcache': {
            'cache_storage': '/tmp/blockcache',
            'block_size': 5 * 1024**2  # 5 MB blocks
        }
    },
    mode='r'
)

# Only caches accessed 5 MB segments (not full chunks)
data = arr[0:10, 0:10]  # Small read, minimal cache usage
```

**Clear Cache:**
```bash
# Remove cached files manually
rm -rf /home/user/.zarr-cache
```

---

## Pattern 5: High-Performance with Obstore (Zarr v3)

**When to use:** Zarr v3 only, high-throughput reads, Rust-based performance.

**S3 with Obstore:**
```python
import zarr

# Create ObjectStore (requires zarr>=3.0)
store = zarr.store.ObjectStore.from_url(
    's3://bucket/path/array.zarr',
    storage_options={
        'aws_region': 'us-west-2',
        'aws_access_key_id': 'key',
        'aws_secret_access_key': 'secret'
    }
)

# Open Zarr v3 array
arr = zarr.open_array(store, mode='r', zarr_format=3)
data = arr[:]
```

**GCS with Obstore:**
```python
import zarr

store = zarr.store.ObjectStore.from_url(
    'gs://bucket/path/array.zarr',
    storage_options={
        'google_service_account_path': '/path/to/credentials.json'
    }
)

arr = zarr.open_array(store, mode='r', zarr_format=3)
```

**Azure with Obstore:**
```python
import zarr

store = zarr.store.ObjectStore.from_url(
    'az://container/path/array.zarr',
    storage_options={
        'azure_storage_account_name': 'myaccount',
        'azure_storage_account_key': 'key'
    }
)

arr = zarr.open_array(store, mode='r', zarr_format=3)
```

**Benefits:**
- Rust-based implementation (faster than Python fsspec)
- Excellent for read-heavy workloads
- Apache Arrow integration
- Native async I/O

**Limitations:**
- Zarr v3 only (no v2 support)
- Requires `obstore` package
- Less mature than fsspec

---

## Pattern 6: Versioned Stores with Icechunk (ACID Compliance)

**When to use:** Collaborative workflows, version history, time-travel, ACID requirements.

**Create Icechunk Store:**
```python
from icechunk import IcechunkStore, StorageConfig

# Configure S3 backend
storage_config = StorageConfig.s3_from_env(
    bucket='my-bucket',
    prefix='data/versioned-array.zarr'
)

# Open or create versioned store
store = IcechunkStore.open_or_create(storage_config)

# Create array
arr = zarr.create_array(
    store,
    shape=(1000, 1000),
    chunks=(100, 100),
    dtype='f4',
    zarr_format=3
)

# Write data
arr[:] = data

# Commit (creates snapshot)
store.commit('Initial data import')
```

**Update and Version:**
```python
# Update data
arr[0:100, :] = new_data

# Commit new version
commit_id = store.commit('Updated first 100 rows')

# Later: checkout specific version
old_store = IcechunkStore.open_or_create(storage_config)
old_store.checkout(commit_id='<previous-commit-id>')
old_arr = zarr.open_array(old_store, mode='r')
```

**Time-Travel:**
```python
# List all commits
commits = store.log()

for commit in commits:
    print(f"{commit.id}: {commit.message} ({commit.timestamp})")

# Revert to specific commit
store.checkout(commit_id=commits[2].id)
arr = zarr.open_array(store, mode='r')  # Data from that point in time
```

**Benefits:**
- ACID compliance (atomic commits)
- Version history with time-travel
- Collaboration-friendly (branching, merging)
- Preferred for VirtualiZarr workflows

**Limitations:**
- Zarr v3 only
- Additional overhead vs standard stores
- Newer library (1.0 released Jul 2025)

---

## Pattern 7: Multi-Cloud Strategy (Provider Abstraction)

**When to use:** Multi-cloud deployments, provider portability, disaster recovery.

**Abstract Storage Configuration:**
```python
import zarr
from fsspec import filesystem
import os

# Provider-agnostic configuration
PROVIDER = os.getenv('CLOUD_PROVIDER', 's3')  # s3, gcs, az
BUCKET = os.getenv('CLOUD_BUCKET', 'my-bucket')
PREFIX = os.getenv('CLOUD_PREFIX', 'data/array.zarr')

def get_cloud_store(provider, bucket, prefix):
    """Get Zarr store for any cloud provider."""

    if provider == 's3':
        fs = filesystem('s3', anon=False)
        uri = f's3://{bucket}/{prefix}'

    elif provider == 'gcs':
        fs = filesystem('gcs', token=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
        uri = f'gs://{bucket}/{prefix}'

    elif provider == 'az':
        fs = filesystem('az', account_name=os.getenv('AZURE_STORAGE_ACCOUNT'))
        uri = f'az://{bucket}/{prefix}'

    else:
        raise ValueError(f"Unknown provider: {provider}")

    store = zarr.storage.FsspecStore(f'{bucket}/{prefix}', fs=fs)
    return store, uri

# Use with any provider
store, uri = get_cloud_store(PROVIDER, BUCKET, PREFIX)
arr = zarr.open_array(store, mode='r')

print(f"Accessing: {uri}")
```

**Replication Across Providers:**
```python
import zarr

# Read from S3
source_store = zarr.storage.FsspecStore('s3://source-bucket/array.zarr')
arr = zarr.open_array(source_store, mode='r')

# Replicate to GCS
dest_store = zarr.storage.FsspecStore('gcs://dest-bucket/array.zarr')
zarr.copy_store(source_store, dest_store)

print("✓ Replicated S3 → GCS")
```

**Benefits:**
- Provider portability
- Easier migration between clouds
- Multi-cloud redundancy
- Vendor lock-in mitigation

---

## Pattern 8: Anonymous Public Access

**When to use:** Reading public datasets, no authentication required.

**S3 Public Bucket:**
```python
import zarr
from fsspec import filesystem

# Anonymous S3 access
fs = filesystem('s3', anon=True)
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')

# Example: Pangeo public datasets
arr = zarr.open_array('s3://pangeo-data/dataset.zarr',
                       storage_options={'anon': True},
                       mode='r')
```

**GCS Public Bucket:**
```python
import zarr
from fsspec import filesystem

# Anonymous GCS access
fs = filesystem('gcs', token='anon')
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Azure Public Container:**
```python
import zarr
from fsspec import filesystem

# Anonymous Azure access
fs = filesystem('az', account_name='storageaccount', anon=True)
store = zarr.storage.FsspecStore('container/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Best practices:**
- Always use `mode='r'` (read-only)
- Check dataset license/terms of use
- Be mindful of egress costs (even for public data)

---

## Cross-References

- **zarr-fundamentals** — chunking strategies, array creation, data types
- **compression-codecs** — compression impacts cloud storage costs and transfer time
- **zarr-xarray-integration** — Xarray's storage_options for cloud backends
- **data-migration** — migrating data to cloud storage
