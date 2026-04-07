---
name: cloud-storage-backends
description: |
  Use this skill when the user asks to "store zarr on S3", "read zarr from GCS", "configure azure blob for zarr", "set up fsspec for zarr", "use obstore", "configure icechunk", "access public zarr data", "consolidate zarr metadata", or needs guidance on Zarr storage backends, cloud provider configuration, authentication, caching, metadata consolidation, or cloud data catalogs.
---

# Cloud Storage Backends for Zarr

Comprehensive guide to storing and accessing Zarr arrays on cloud object storage (Amazon S3, Google Cloud Storage, Azure Blob Storage) and local backends. Covers fsspec, obstore, Icechunk, authentication patterns, metadata consolidation, and caching strategies.

## Quick Reference: Cloud Access Patterns

### Pattern 1: fsspec URL (Simplest)
```python
import zarr

# S3 (automatic credential discovery)
arr = zarr.open_array('s3://bucket/path/array.zarr', mode='r')

# GCS
arr = zarr.open_array('gs://bucket/path/array.zarr', mode='r')

# Azure
arr = zarr.open_array('az://container/path/array.zarr', mode='r')
```

### Pattern 2: Explicit FsspecStore (Configurable)
```python
import zarr
from fsspec import filesystem

# S3 with custom config
fs = filesystem('s3', anon=False, client_kwargs={'region_name': 'us-west-2'})
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

### Pattern 3: Obstore (High Performance)
```python
import zarr

# S3 with obstore (Rust-based, Apache Arrow integration)
store = zarr.store.ObjectStore.from_url(
    's3://bucket/path/array.zarr',
    storage_options={'aws_region': 'us-west-2'}
)
arr = zarr.open_array(store, mode='r')
```

### Pattern 4: Icechunk (ACID + Versioning)
```python
from icechunk import IcechunkStore, StorageConfig

# Versioned, ACID-compliant store
storage_config = StorageConfig.s3_from_env(
    bucket='bucket',
    prefix='path/array.zarr'
)
store = IcechunkStore.open_or_create(storage_config)
arr = zarr.open_array(store, mode='r')
```

### Pattern 5: Cached Access
```python
import zarr

# fsspec caching (local cache for repeated reads)
arr = zarr.open_array(
    'simplecache::s3://bucket/path/array.zarr',
    storage_options={
        's3': {'anon': False},
        'simplecache': {'cache_storage': '/tmp/zarr-cache'}
    },
    mode='r'
)
```

## Available Storage Backends

### Complete Backend Table

| Backend | Zarr v2 | Zarr v3 | Use Case | Dependencies |
|---------|---------|---------|----------|--------------|
| **LocalStore** | ✅ | ✅ | Local filesystem | None (built-in) |
| **MemoryStore** | ✅ | ✅ | Testing, temporary data | None (built-in) |
| **ZipStore** | ✅ | ✅ | Archived datasets | None (built-in) |
| **FsspecStore** | ✅ | ✅ | Cloud storage, remote FS | `fsspec` + provider |
| **ObjectStore** | ❌ | ✅ | High-perf cloud (Rust) | `obstore` (optional) |
| **IcechunkStore** | ❌ | ✅ | Versioned, ACID | `icechunk` |
| **RedisStore** | ✅ | ❌ | In-memory cache | `redis-py` |
| **MongoDBStore** | ✅ | ❌ | Document storage | `pymongo` |
| **SQLiteStore** | ✅ | ❌ | Single-file DB | None (stdlib) |
| **LMDBStore** | ✅ | ❌ | High-perf key-value | `lmdb` |
| **DirectoryStore** | ✅ | ❌ | v2 default filesystem | None (built-in) |
| **NestedDirectoryStore** | ✅ | ❌ | v2 nested hierarchy | None (built-in) |
| **DBMStore** | ✅ | ❌ | Simple key-value | None (stdlib) |
| **N5Store** | ✅ | ❌ | N5 format compatibility | None (built-in) |
| **ConsolidatedMetadataStore** | ✅ | ❌ | Wrapped cached metadata | None (built-in) |

**Notes:**
- **v2 only stores** (Redis, MongoDB, SQLite, LMDB, DBM, N5) are deprecated in Zarr v3
- **v3 stores** (ObjectStore, Icechunk) use new storage interface and async I/O
- **FsspecStore** works with both v2 and v3, most flexible for cloud access
- **LocalStore** is the default for v3, replaces DirectoryStore

### Cloud Provider Support via Fsspec

| Provider | Fsspec Protocol | Implementation Library | Status |
|----------|----------------|----------------------|--------|
| **Amazon S3** | `s3://` | `s3fs` | Mature |
| **Google Cloud Storage** | `gs://` or `gcs://` | `gcsfs` | Mature |
| **Azure Blob Storage** | `az://` or `abfs://` | `adlfs` | Mature |
| **HTTP/HTTPS** | `http://`, `https://` | `aiohttp` | Read-only |
| **FTP** | `ftp://` | built-in | Legacy |
| **SFTP** | `sftp://` | `paramiko` | Secure transfer |
| **Google Drive** | `gdrive://` | `gdrivefs` | Experimental |
| **Dropbox** | `dropbox://` | `dropboxdrivefs` | Experimental |
| **Local** | `file://` | built-in | Default |

## Backend Comparison: fsspec vs obstore vs Icechunk

| Feature | fsspec (FsspecStore) | obstore (ObjectStore) | Icechunk |
|---------|---------------------|---------------------|----------|
| **Maturity** | Mature (4+ years) | Emerging (2024+) | New (1.0 Jul 2025) |
| **Performance** | Good | Excellent (Rust) | Good |
| **Zarr v2** | ✅ Yes | ❌ No | ❌ No |
| **Zarr v3** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Multi-cloud** | ✅ Yes (S3/GCS/Azure) | ✅ Yes | ✅ Yes |
| **Async I/O** | ✅ Yes | ✅ Yes (native) | ✅ Yes |
| **Versioning** | ❌ No | ❌ No | ✅ Yes (time-travel) |
| **ACID compliance** | ❌ No | ❌ No | ✅ Yes |
| **VirtualiZarr** | ✅ Yes | ✅ Yes | ✅ Yes (preferred) |
| **Caching** | ✅ Built-in | ⚠️ Manual | ✅ Built-in |
| **Dependencies** | fsspec + provider | `obstore` (Rust) | `icechunk` |
| **Best for** | General cloud access | High-throughput reads | Collaborative workflows |

**Recommendations:**
- **Default choice:** fsspec (FsspecStore) — mature, widely used, flexible
- **High performance:** obstore (ObjectStore) — Rust-based, excellent for read-heavy workloads
- **Versioned data:** Icechunk — ACID compliance, time-travel, collaboration
- **Legacy v2 support:** fsspec only (obstore and Icechunk are v3-only)

## Authentication Patterns by Provider

### Amazon S3

**Option 1: Environment Variables (Recommended)**
```python
import os
import zarr

# Set AWS credentials (or use IAM role on EC2/Lambda)
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key'
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

arr = zarr.open_array('s3://bucket/path/array.zarr', mode='r')
```

**Option 2: AWS Profile**
```python
from fsspec import filesystem

fs = filesystem('s3', profile='my-aws-profile')
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Option 3: IAM Role (On EC2/ECS/Lambda)**
```python
# No credentials needed — IAM role automatically used
arr = zarr.open_array('s3://bucket/path/array.zarr', mode='r')
```

**Option 4: Anonymous Access (Public Buckets)**
```python
from fsspec import filesystem

fs = filesystem('s3', anon=True)
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Custom Configuration**
```python
from fsspec import filesystem

fs = filesystem(
    's3',
    key='your-access-key',       # Access key
    secret='your-secret-key',    # Secret key
    token='session-token',        # Optional session token
    endpoint_url='https://...',   # Custom S3-compatible endpoint (MinIO, Wasabi)
    use_ssl=True,
    client_kwargs={
        'region_name': 'us-west-2',
        'config': {
            'signature_version': 's3v4',
            'max_pool_connections': 50
        }
    }
)
```

### Google Cloud Storage (GCS)

**Option 1: Application Default Credentials (Recommended)**
```python
import zarr

# Uses gcloud auth application-default login
arr = zarr.open_array('gs://bucket/path/array.zarr', mode='r')
```

**Option 2: Service Account Key**
```python
from fsspec import filesystem

fs = filesystem(
    'gcs',
    project='my-gcp-project',
    token='/path/to/service-account-key.json'
)
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Option 3: Anonymous Access (Public Buckets)**
```python
from fsspec import filesystem

fs = filesystem('gcs', token='anon')
store = zarr.storage.FsspecStore('bucket/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Custom Configuration**
```python
from fsspec import filesystem

fs = filesystem(
    'gcs',
    project='my-project',
    token='/path/to/credentials.json',  # Service account JSON
    access='read_only',                  # read_only, read_write, full_control
    consistency='md5',                   # md5, size, or None
    cache_timeout=300                    # Metadata cache duration (seconds)
)
```

### Azure Blob Storage

**Option 1: Connection String**
```python
import os
import zarr

os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;...'
arr = zarr.open_array('az://container/path/array.zarr', mode='r')
```

**Option 2: Account Key**
```python
from fsspec import filesystem

fs = filesystem(
    'az',
    account_name='mystorageaccount',
    account_key='your-account-key'
)
store = zarr.storage.FsspecStore('container/path/array.zarr', fs=fs)
arr = zarr.open_array(store, mode='r')
```

**Option 3: SAS Token**
```python
from fsspec import filesystem

fs = filesystem(
    'az',
    account_name='mystorageaccount',
    sas_token='?sv=2021-06-08&ss=...'  # Shared Access Signature
)
```

**Option 4: Managed Identity (On Azure VMs)**
```python
from fsspec import filesystem

fs = filesystem('az', account_name='mystorageaccount', anon=False)
# Uses Azure Managed Identity for authentication
```

**Anonymous Access**
```python
from fsspec import filesystem

fs = filesystem('az', account_name='mystorageaccount', anon=True)
```

## Metadata Consolidation (Critical for Cloud Performance)

Zarr arrays split metadata across many small JSON files (`.zarray`, `.zattrs`, `.zgroup`). On cloud storage, each file requires a separate network request, causing significant latency for large hierarchies.

**Metadata consolidation** combines all metadata into a single `.zmetadata` file, reducing hundreds of requests to one.

### Creating Consolidated Metadata (Zarr v2)

```python
import zarr

# Write data to cloud
store = zarr.storage.FsspecStore('s3://bucket/path/dataset.zarr')
root = zarr.open_group(store, mode='w')
root.create_dataset('temperature', shape=(365, 100, 100), chunks=(10, 100, 100))
root.create_dataset('pressure', shape=(365, 100, 100), chunks=(10, 100, 100))

# Consolidate metadata (CRITICAL for cloud performance)
zarr.consolidate_metadata(store)
```

**What happens:**
- Creates `.zmetadata` file with all metadata
- Single read instead of N reads (where N = number of arrays + groups)
- **Massive performance improvement** for cloud access

### Reading with Consolidated Metadata

```python
import zarr

# Read with consolidated metadata (default)
store = zarr.storage.FsspecStore('s3://bucket/path/dataset.zarr')
root = zarr.open_consolidated(store, mode='r')  # Uses .zmetadata

# Access arrays (no additional metadata requests)
temp = root['temperature']
press = root['pressure']
```

**Performance impact:**
- Without consolidation: 10-100+ network requests on open
- With consolidation: 1 network request on open
- **10-100x faster** for datasets with many arrays

### When to Re-Consolidate

You **MUST** re-consolidate metadata after structural changes:

```python
import zarr

store = zarr.storage.FsspecStore('s3://bucket/path/dataset.zarr')

# Scenario 1: Adding new arrays
root = zarr.open_group(store, mode='a')
root.create_dataset('humidity', shape=(365, 100, 100), chunks=(10, 100, 100))
zarr.consolidate_metadata(store)  # Re-consolidate!

# Scenario 2: Modifying attributes
root = zarr.open_group(store, mode='a')
root.attrs['version'] = '2.0'
zarr.consolidate_metadata(store)  # Re-consolidate!

# Scenario 3: Deleting arrays
del root['old_variable']
zarr.consolidate_metadata(store)  # Re-consolidate!
```

**Note:** Data writes (not structural changes) do NOT require re-consolidation.

### Zarr v3 Metadata Handling

Zarr v3 uses a different metadata structure:
- All metadata in `zarr.json` files (no separate `.zarray`, `.zattrs`)
- Consolidation works differently (still recommended for cloud)
- Use same `zarr.consolidate_metadata()` API

## Caching Strategies

Caching reduces cloud API calls by storing frequently accessed data locally.

### Cache Types

| Cache Type | Use Case | Behavior |
|------------|----------|----------|
| **simplecache** | Read-once workflows | In-memory, non-persistent |
| **filecache** | Repeated reads | Disk cache, persistent |
| **blockcache** | Partial chunk reads | Caches chunk segments |

### Simple Cache (In-Memory)

```python
import zarr

arr = zarr.open_array(
    'simplecache::s3://bucket/path/array.zarr',
    storage_options={
        's3': {'anon': False},
        'simplecache': {'cache_storage': '/tmp/zarr-cache'}
    },
    mode='r'
)
```

**Behavior:**
- Downloads entire files on first access
- Stores in memory (or temp directory)
- Lifetime: single session

### File Cache (Persistent)

```python
import zarr

arr = zarr.open_array(
    'filecache::s3://bucket/path/array.zarr',
    storage_options={
        's3': {'anon': False},
        'filecache': {
            'cache_storage': '/home/user/.zarr-cache',
            'expiry_time': 86400  # 24 hours in seconds
        }
    },
    mode='r'
)
```

**Behavior:**
- Downloads files to local disk
- Persists across sessions
- Respects `expiry_time` (in seconds)
- Automatic cache eviction

### Block Cache (Chunk Segments)

```python
import zarr

arr = zarr.open_array(
    'blockcache::s3://bucket/path/array.zarr',
    storage_options={
        's3': {'anon': False},
        'blockcache': {
            'cache_storage': '/tmp/blockcache',
            'block_size': 5*1024**2  # 5 MB blocks
        }
    },
    mode='r'
)
```

**Behavior:**
- Caches fixed-size blocks (not full chunks)
- Efficient for sparse access patterns
- Good for very large chunks

### Nested Caching

```python
# Combine caching layers
arr = zarr.open_array(
    'simplecache::blockcache::s3://bucket/path/array.zarr',
    storage_options={
        's3': {'anon': False},
        'blockcache': {'block_size': 1024**2},
        'simplecache': {'cache_storage': '/tmp/cache'}
    },
    mode='r'
)
```

## Concurrency Tuning

Zarr v3 uses async I/O for parallel chunk reads/writes. Configure concurrency per provider.

### Global Concurrency Configuration

```python
import zarr

# Set global async concurrency (applies to all operations)
zarr.config.set({'async.concurrency': 64})
```

**Recommended values by provider:**
- **S3**: 64-128 (high concurrency, excellent parallelism)
- **GCS**: 32-64 (moderate concurrency)
- **Azure**: 32-64 (moderate concurrency)
- **Local filesystem**: 4-8 (limited by I/O, not network)

### Per-Operation Concurrency

```python
import zarr

# Specify concurrency for a single write
arr = zarr.create_array(
    shape=(1000, 1000),
    chunks=(100, 100),
    dtype='f8',
    store='s3://bucket/path/array.zarr'
)

# Write with custom concurrency
with zarr.config.set({'async.concurrency': 128}):
    arr[:] = data  # Uses 128 concurrent writes
```

### Optimization Guidelines

**Read-heavy workloads:**
- Higher concurrency (64-128 for S3)
- Enable caching (filecache or simplecache)
- Use consolidated metadata

**Write-heavy workloads:**
- Moderate concurrency (32-64)
- Larger chunks to reduce number of writes
- Batch writes when possible

**Mixed workloads:**
- Start with defaults (provider-specific)
- Profile and adjust based on latency/throughput metrics

## Storage Options: Best Practices

### Checklist for Cloud Deployment

- [ ] **Authentication configured** (environment variables, IAM roles, or service accounts)
- [ ] **Metadata consolidated** (for v2) or verified (for v3)
- [ ] **Caching enabled** for read-heavy workflows (filecache recommended)
- [ ] **Concurrency tuned** based on provider (S3: 64-128, GCS/Azure: 32-64)
- [ ] **Chunking optimized** for access patterns (see zarr-fundamentals skill)
- [ ] **Region matching** between compute and storage (reduces latency and cost)
- [ ] **Cost monitoring** enabled (cloud storage + egress charges)
- [ ] **Backup strategy** in place (versioning, snapshots, or replication)

### Security Considerations

- **Never hardcode credentials** in code (use environment variables or IAM roles)
- **Use least-privilege access** (read-only when possible)
- **Enable encryption** at rest (S3/GCS/Azure support server-side encryption)
- **Use HTTPS** for all cloud access (default for fsspec)
- **Audit access logs** (CloudTrail, Cloud Audit Logs, Azure Monitor)
- **Implement lifecycle policies** (auto-delete or archive old data)

## Advanced: Obstore and Icechunk

### Obstore (High-Performance Cloud Access)

```python
import zarr

# S3 with obstore
store = zarr.store.ObjectStore.from_url(
    's3://bucket/path/array.zarr',
    storage_options={
        'aws_region': 'us-west-2',
        'aws_access_key_id': 'key',      # Or use env vars
        'aws_secret_access_key': 'secret'
    }
)

arr = zarr.open_array(store, mode='r')
```

**Advantages:**
- Rust-based implementation (very fast)
- Apache Arrow integration
- Native async I/O
- Lower memory overhead

**Limitations:**
- Zarr v3 only
- Newer, less battle-tested than fsspec

### Icechunk (Versioned, ACID-Compliant Stores)

```python
from icechunk import IcechunkStore, StorageConfig

# Create versioned store on S3
storage_config = StorageConfig.s3_from_env(
    bucket='my-bucket',
    prefix='data/array.zarr'
)

store = IcechunkStore.open_or_create(storage_config)

# Write data (transactional)
arr = zarr.create_array(store, shape=(1000, 1000), chunks=(100, 100), dtype='f8')
arr[:] = data

store.commit('Initial commit')  # Snapshot 1

# Update data
arr[0:100, :] = new_data
store.commit('Updated first 100 rows')  # Snapshot 2

# Time-travel: revert to previous snapshot
old_snapshot = store.checkout(snapshot_id='<previous-id>')
```

**Advantages:**
- ACID compliance (atomicity, consistency, isolation, durability)
- Version history and time-travel
- Collaboration-friendly (branching, merging)
- Preferred for VirtualiZarr workflows

**Limitations:**
- Zarr v3 only
- Additional complexity vs standard stores
- Requires icechunk library (new as of Jul 2025)

## Related Skills and References

**Cross-references:**
- **zarr-fundamentals** — chunking strategies, array creation, Zarr v2 vs v3
- **compression-codecs** — compression impacts cloud costs and transfer time
- **zarr-xarray-integration** — Xarray's `to_zarr()` storage_options parameter

**Reference files in this skill:**
- `references/PATTERNS.md` — 6+ cloud access patterns with complete code
- `references/EXAMPLES.md` — 4+ end-to-end examples (S3 deployment, multi-cloud, etc.)
- `references/COMMON_ISSUES.md` — 5+ troubleshooting guides (auth errors, permissions, caching)

**Runnable templates:**
- `assets/s3-config-template.py` — Production S3 configuration with error handling
- `assets/gcs-config-template.py` — Production GCS configuration
- `assets/azure-config-template.py` — Production Azure configuration

---

**This skill covers cloud storage backends for Zarr. For cloud-specific deployment and optimization, see the zarr-cloud-architect agent.**
