# Common Issues: Cloud Storage Backends

Troubleshooting guide for Zarr cloud storage backends (S3, GCS, Azure).

## Issue 1: Authentication Failures

### Symptoms
```python
zarr.open_array('s3://bucket/array.zarr', mode='r')
# PermissionError: Access Denied
# botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

### Causes
1. **Missing credentials** — No AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY set
2. **Wrong region** — Bucket in different region than configured
3. **IAM permissions** — Insufficient S3 permissions (s3:GetObject, s3:PutObject)
4. **Expired credentials** — Temporary credentials or session tokens expired

### Solutions

**S3 — Environment Variables:**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"
```

**S3 — Explicit Credentials:**
```python
from fsspec import filesystem

fs = filesystem(
    's3',
    key='your-access-key',
    secret='your-secret-key',
    client_kwargs={'region_name': 'us-west-2'}
)
store = zarr.storage.FsspecStore('bucket/path', fs=fs)
```

**GCS — Application Default Credentials:**
```bash
# One-time setup
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

**Azure — Connection String:**
```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
```

**Debugging:**
```python
# Test credentials before opening Zarr
from fsspec import filesystem

fs = filesystem('s3')  # Will raise error if no credentials
print(fs.ls('bucket/'))  # Test list operation
```

### Prevention
- Use IAM roles on cloud VMs (no hardcoded credentials)
- Store credentials in environment variables (not in code)
- Use credential management tools (AWS Secrets Manager, HashiCorp Vault)
- Check CloudTrail/Cloud Audit Logs for permission denials

---

## Issue 2: Slow Cloud Access (Metadata Overhead)

### Symptoms
```python
root = zarr.open_group('s3://bucket/dataset.zarr', mode='r')
# Takes 10-60 seconds to open
# Hundreds of S3 GET requests in CloudWatch
```

### Causes
1. **No metadata consolidation** — Each array/group requires separate network request
2. **Large hierarchy** — Hundreds of arrays = hundreds of `.zarray`/`.zattrs` files
3. **High latency** — Network round-trips dominate open time

### Solutions

**Consolidate Metadata (v2):**
```python
import zarr

# Writer consolidates metadata
store = zarr.storage.FsspecStore('s3://bucket/dataset.zarr')
root = zarr.open_group(store, mode='w')
# ... create arrays ...
zarr.consolidate_metadata(store)  # CRITICAL!

# Reader uses consolidated metadata
root = zarr.open_consolidated('s3://bucket/dataset.zarr', mode='r')  # Fast!
```

**Verify Consolidation:**
```python
# Check if .zmetadata exists
from fsspec import filesystem

fs = filesystem('s3')
files = fs.ls('bucket/dataset.zarr/')
if any('.zmetadata' in f for f in files):
    print("✓ Metadata is consolidated")
else:
    print("❌ No .zmetadata found — consolidate now!")
```

**Re-consolidate After Changes:**
```python
# After adding new arrays or modifying attributes
store = zarr.storage.FsspecStore('s3://bucket/dataset.zarr')
root = zarr.open_group(store, mode='a')
root.create_dataset('new_variable', shape=(100, 100), chunks=(10, 10))
zarr.consolidate_metadata(store)  # Must re-consolidate!
```

**Performance Impact:**
- Without consolidation: 100+ network requests on open
- With consolidation: 1 network request on open
- **10-100x faster** for large datasets

### Prevention
- ALWAYS consolidate metadata for cloud storage
- Add consolidation to deployment pipelines
- Monitor S3 GET request counts

---

## Issue 3: Region Mismatch (High Latency/Egress Costs)

### Symptoms
```python
# Compute in us-east-1, data in us-west-2
arr = zarr.open_array('s3://bucket/array.zarr', mode='r')
data = arr[:]  # Slow, expensive cross-region transfer
```

### Causes
1. **Compute and storage in different regions** — Cross-region latency (20-100ms)
2. **Data egress charges** — AWS charges for cross-region data transfer
3. **Bandwidth limits** — Inter-region bandwidth caps

### Solutions

**Match Regions:**
```python
import zarr
from fsspec import filesystem

# Ensure bucket and compute are in same region
S3_REGION = 'us-west-2'  # Match EC2/Lambda region

fs = filesystem('s3', client_kwargs={'region_name': S3_REGION})
store = zarr.storage.FsspecStore('bucket/array.zarr', fs=fs)
```

**Check Bucket Region:**
```bash
# AWS CLI
aws s3api get-bucket-location --bucket my-bucket

# Output: {"LocationConstraint": "us-west-2"}
```

**Replicate to Local Region:**
```python
import zarr

# Copy data to local region bucket
source = zarr.storage.FsspecStore('s3://us-west-2-bucket/array.zarr')
dest = zarr.storage.FsspecStore('s3://us-east-1-bucket/array.zarr')
zarr.copy_store(source, dest)
```

**Cost Impact:**
- Same region: $0.00/GB transfer
- Cross-region: $0.02/GB transfer (varies by region pair)
- **Huge savings** for large datasets

### Prevention
- Deploy compute near data (not vice versa)
- Use S3 Requester Pays for public datasets (user pays egress)
- Monitor egress costs in CloudWatch

---

## Issue 4: Cache Not Working (Repeated Downloads)

### Symptoms
```python
# Multiple reads, but always re-downloading chunks
arr = zarr.open_array('filecache::s3://bucket/array.zarr', ...)
data1 = arr[0:100, :]  # Downloads chunks
data2 = arr[0:100, :]  # Downloads chunks AGAIN (should be cached!)
```

### Causes
1. **Cache directory permissions** — Cannot write to cache directory
2. **Insufficient disk space** — Cache writes fail silently
3. **Wrong cache configuration** — Cache not enabled or misconfigured
4. **simplecache vs filecache** — simplecache is in-memory, doesn't persist

### Solutions

**Verify Cache Configuration:**
```python
import zarr
import os

CACHE_DIR = '/tmp/zarr-cache'

# Ensure cache directory exists and is writable
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache dir: {CACHE_DIR}")
print(f"Writable: {os.access(CACHE_DIR, os.W_OK)}")

# Use filecache (persistent)
arr = zarr.open_array(
    'filecache::s3://bucket/array.zarr',
    storage_options={
        's3': {'anon': False},
        'filecache': {
            'cache_storage': CACHE_DIR,
            'same_names': True  # Preserve structure
        }
    },
    mode='r'
)
```

**Check Cache Contents:**
```bash
# List cached files
ls -lah /tmp/zarr-cache/

# Check disk space
df -h /tmp
```

**Clear and Rebuild Cache:**
```bash
# Remove old cache
rm -rf /tmp/zarr-cache

# Recreate
mkdir -p /tmp/zarr-cache
chmod 755 /tmp/zarr-cache
```

**Test Caching:**
```python
import zarr
import time

arr = zarr.open_array('filecache::s3://bucket/array.zarr', ...)

# First read (downloads)
t0 = time.time()
data1 = arr[0:100, :]
t1 = time.time()
print(f"First read: {t1 - t0:.2f}s")

# Second read (cached, should be much faster)
t0 = time.time()
data2 = arr[0:100, :]
t1 = time.time()
print(f"Second read: {t1 - t0:.2f}s (should be <0.1s)")
```

### Prevention
- Use `filecache::` for persistent cache (not `simplecache::`)
- Verify cache directory permissions before deployment
- Monitor disk space on cache volume
- Set cache expiry to avoid stale data

---

## Issue 5: Write Failures (Incomplete Uploads)

### Symptoms
```python
arr = zarr.create_array('s3://bucket/array.zarr', shape=(1000, 1000), ...)
arr[:] = data
# Script crashes, but some chunks uploaded to S3
# Later reads fail with inconsistent data
```

### Causes
1. **Network interruption** — Upload interrupted mid-write
2. **Insufficient permissions** — Can read but not write (s3:PutObject missing)
3. **Quota exceeded** — Storage quota limit reached
4. **Non-atomic writes** — Zarr writes are not transactional by default

### Solutions

**Use Write Mode Carefully:**
```python
import zarr

# mode='w' — Overwrites existing store (DANGEROUS)
# mode='a' — Append (safer, but can create inconsistent state if interrupted)
# mode='r+' — Read-write (modifies in place)

# Safest: Write to temporary location, then atomic rename
temp_store = zarr.storage.FsspecStore('s3://bucket/temp-array.zarr')
arr = zarr.create_array(temp_store, shape=(1000, 1000), chunks=(100, 100))
arr[:] = data

# After successful write, move to final location
import fsspec
fs = fsspec.filesystem('s3')
fs.rename('bucket/temp-array.zarr', 'bucket/final-array.zarr')
```

**Verify Permissions:**
```python
from fsspec import filesystem

fs = filesystem('s3')

# Test write permission
test_path = 'bucket/test-write.txt'
try:
    with fs.open(test_path, 'w') as f:
        f.write('test')
    fs.rm(test_path)
    print("✓ Write permissions OK")
except Exception as e:
    print(f"❌ Write failed: {e}")
```

**Use Icechunk for ACID Writes (v3 only):**
```python
from icechunk import IcechunkStore, StorageConfig

# ACID-compliant store (atomic commits)
storage_config = StorageConfig.s3_from_env(
    bucket='bucket',
    prefix='array.zarr'
)
store = IcechunkStore.open_or_create(storage_config)

# Write data
arr = zarr.create_array(store, shape=(1000, 1000), chunks=(100, 100))
arr[:] = data

# Atomic commit (all-or-nothing)
store.commit('Initial data upload')
```

**Recovery from Incomplete Write:**
```bash
# List all chunks
aws s3 ls s3://bucket/array.zarr/ --recursive

# Remove incomplete write
aws s3 rm s3://bucket/array.zarr/ --recursive
```

### Prevention
- Use temporary staging location + atomic rename
- Enable S3 versioning (allows rollback)
- Use Icechunk for ACID compliance (Zarr v3)
- Implement retry logic with exponential backoff
- Monitor upload progress and checksum integrity

---

## Issue 6: "Too Many Open Files" Error

### Symptoms
```python
# Opening many Zarr stores or arrays
for i in range(1000):
    arr = zarr.open_array(f's3://bucket/array-{i}.zarr', mode='r')
    data = arr[0:10, :]

# OSError: [Errno 24] Too many open files
```

### Causes
1. **File descriptor leak** — fsspec keeps S3 connections open
2. **OS limits** — Operating system limits concurrent open files
3. **Not closing stores** — Zarr stores not explicitly closed

### Solutions

**Close Stores Explicitly:**
```python
import zarr

# Open, use, close pattern
store = zarr.storage.FsspecStore('s3://bucket/array.zarr')
arr = zarr.open_array(store, mode='r')
data = arr[0:10, :]
store.close()  # Explicitly close
```

**Use Context Manager:**
```python
# fsspec filesystem context manager
from fsspec import filesystem

with filesystem('s3').open('bucket/path/file.txt', 'r') as f:
    content = f.read()
# Automatically closed
```

**Increase OS Limits (Linux/Mac):**
```bash
# Check current limit
ulimit -n

# Increase to 4096 (temporary)
ulimit -n 4096

# Permanent (add to ~/.bashrc or /etc/security/limits.conf)
echo "* soft nofile 4096" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 8192" | sudo tee -a /etc/security/limits.conf
```

**Batch Processing:**
```python
import zarr

# Process in smaller batches
BATCH_SIZE = 100

for batch_start in range(0, 1000, BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE

    for i in range(batch_start, batch_end):
        store = zarr.storage.FsspecStore(f's3://bucket/array-{i}.zarr')
        arr = zarr.open_array(store, mode='r')
        data = arr[0:10, :]
        store.close()

    # Batch complete, connections released
```

### Prevention
- Always close stores when done
- Process large file lists in batches
- Use connection pooling configuration
- Monitor open file descriptors: `lsof -p <pid> | wc -l`

---

## Issue 7: Zarr v2 vs v3 Incompatibility

### Symptoms
```python
# Created with Zarr v3
arr = zarr.create_array('s3://bucket/array.zarr', shape=(100, 100), zarr_format=3)

# Try to read with Zarr v2 code
import zarr  # v2.x
arr = zarr.open_array('s3://bucket/array.zarr', mode='r')
# Raises: ValueError or KeyError (cannot parse metadata)
```

### Causes
1. **Metadata format incompatibility** — v2 uses `.zarray`, v3 uses `zarr.json`
2. **Different default stores** — v2: DirectoryStore, v3: LocalStore
3. **Codec differences** — v3 uses codec pipeline, v2 uses compressor parameter

### Solutions

**Check Zarr Version:**
```python
import zarr
print(f"Zarr version: {zarr.__version__}")
# 2.x or 3.x
```

**Specify Format When Creating:**
```python
import zarr

# Force Zarr v2 format (for compatibility)
arr = zarr.create_array(
    's3://bucket/array.zarr',
    shape=(100, 100),
    chunks=(10, 10),
    zarr_format=2  # Explicit v2
)
```

**Detect Format:**
```python
from fsspec import filesystem

fs = filesystem('s3')
files = fs.ls('bucket/array.zarr/')

if any('zarr.json' in f for f in files):
    print("Zarr v3 format detected")
    # Use zarr >= 3.0
elif any('.zarray' in f for f in files):
    print("Zarr v2 format detected")
    # Use zarr < 3.0 or zarr >= 3.0 with zarr_format=2
else:
    print("Unknown format or empty store")
```

**Migrate v2 → v3:**
```python
import zarr

# Read v2
source = zarr.open_array('s3://bucket/v2-array.zarr', zarr_format=2, mode='r')

# Write as v3
dest = zarr.create_array(
    's3://bucket/v3-array.zarr',
    shape=source.shape,
    chunks=source.chunks,
    dtype=source.dtype,
    zarr_format=3
)
dest[:] = source[:]
```

### Prevention
- Document Zarr version for datasets
- Pin Zarr version in requirements.txt
- Use Zarr v2 for maximum compatibility (as of 2026)
- Add format metadata to root group: `root.attrs['zarr_format'] = 3`

---

## Issue 8: High Cloud Storage Costs

### Symptoms
```
Monthly S3 bill: $1,000+ for Zarr datasets
Breakdown: 80% GET requests, 20% storage
```

### Causes
1. **No metadata consolidation** — Thousands of GET requests per dataset open
2. **Small chunks** — More chunks = more objects = more requests
3. **Frequent access** — Repeated reads without caching
4. **No compression** — Wasting storage capacity
5. **Cross-region egress** — Data transfer charges

### Solutions

**Metadata Consolidation (Critical):**
```python
# Reduces GET requests by 100x
zarr.consolidate_metadata(store)
```

**Optimize Chunking (Reduce Object Count):**
```python
# Bad: 1 MB chunks → 1 million objects for 1 TB dataset
arr = zarr.create_array(shape=(10000, 10000, 10000), chunks=(10, 10, 10), dtype='f4')

# Good: 10 MB chunks → 100k objects for 1 TB dataset
arr = zarr.create_array(shape=(10000, 10000, 10000), chunks=(50, 50, 50), dtype='f4')
```

**Enable Compression (Reduce Storage):**
```python
# Save 50-80% storage costs
arr = zarr.create_array(
    shape=(1000, 1000),
    chunks=(100, 100),
    compressor=zarr.codecs.Blosc(cname='zstd', clevel=3)
)
```

**Use Caching (Reduce Repeated Access):**
```python
# Avoid re-downloading same chunks
arr = zarr.open_array('filecache::s3://bucket/array.zarr', ...)
```

**S3 Intelligent-Tiering:**
```bash
# Auto-move infrequently accessed data to cheaper storage
aws s3api put-bucket-intelligent-tiering-configuration \
    --bucket my-bucket \
    --id zarr-tiering \
    --intelligent-tiering-configuration file://tiering.json
```

**Monitor Costs:**
```bash
# AWS Cost Explorer
# Track S3 GET/PUT requests and storage costs by tag
```

**Cost Breakdown (Example 1 TB dataset):**
```
Without optimization:
  - Storage: 1 TB × $0.023/GB = $23/month
  - Requests: 1M GET × $0.0004/1k = $400/month
  - Total: $423/month

With optimization:
  - Storage: 300 GB (70% compression) × $0.023/GB = $7/month
  - Requests: 1k GET (metadata consolidation) × $0.0004/1k = $0.40/month
  - Total: $7.40/month

Savings: $415/month (98% reduction!)
```

### Prevention
- ALWAYS consolidate metadata
- Use 1-10 MB chunks (balance between access and object count)
- Enable compression (Blosc+Zstd recommended)
- Implement caching for repeated access
- Set lifecycle policies (delete old temp data)
- Use S3 Intelligent-Tiering or Glacier for archival data

---

## Cross-References

- **zarr-fundamentals** — chunking strategies, Zarr v2 vs v3 differences
- **compression-codecs** — choosing compressors for cloud cost reduction
- **zarr-xarray-integration** — Xarray cloud access patterns
- **data-migration** — migrating data to cloud storage safely
