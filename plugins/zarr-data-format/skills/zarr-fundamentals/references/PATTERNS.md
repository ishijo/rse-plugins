# Zarr Fundamentals Patterns

## Pattern 1: Array Creation and Initialization

**Create arrays with optimal chunking:**
```python
import zarr
import numpy as np

# Explicit array creation with all parameters
arr = zarr.create_array(
    store="scientific_data.zarr",
    shape=(10000, 1000, 1000),  # (time, lat, lon)
    chunks=(100, 100, 100),  # ~4 MB chunks for float32
    dtype="float32",
    fill_value=np.nan,
    compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
    overwrite=False,
    zarr_format=2  # Explicit format version
)

# Convenience functions (NumPy-like)
z_zeros = zarr.zeros((5000, 5000), chunks=(500, 500), dtype="i4", store="zeros.zarr")
z_ones = zarr.ones((1000, 1000), chunks=(100, 100), dtype="f8", store="ones.zarr")
z_full = zarr.full((2000, 3000), fill_value=42.0, chunks=(200, 300), dtype="f4")
z_empty = zarr.empty((10000, 10000), chunks=(1000, 1000), dtype="f4")

# Open existing or create new
z = zarr.open_array(
    "data.zarr",
    mode="a",  # Read/write, create if doesn't exist
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4"
)
```

**Initialize with data:**
```python
# From NumPy array
data = np.random.randn(1000, 1000).astype("f4")
z = zarr.array(
    data,
    chunks=(100, 100),
    store="from_numpy.zarr"
)

# Copy array structure
z_template = zarr.zeros((5000, 5000), chunks=(500, 500), dtype="f4")
z_copy = zarr.empty_like(z_template, store="copy.zarr")

# From existing array with new chunks
z_rechunked = zarr.zeros_like(z_template, chunks=(100, 100), store="rechunked.zarr")
```

## Pattern 2: Reading and Writing Data

**Efficient I/O patterns:**
```python
# Full array write
arr[:] = np.random.randn(*arr.shape)

# Slice write
arr[0:1000, :] = data_slice

# Partial update (read-modify-write)
block = arr[1000:2000, 500:1500]
block_modified = block * 2.0 + 10
arr[1000:2000, 500:1500] = block_modified

# Append-like pattern (time dimension)
for i, time_slice in enumerate(data_generator):
    arr[i, :, :] = time_slice

# Chunk-aligned writes (optimal performance)
chunk_size = arr.chunks[0]
for i in range(0, arr.shape[0], chunk_size):
    end = min(i + chunk_size, arr.shape[0])
    arr[i:end, :] = generate_data(end - i, arr.shape[1])
```

**Reading patterns:**
```python
# Full read
data = arr[:]

# Slice read
time_series = arr[:, 50, 100]  # Single location over time
spatial_slice = arr[0, :, :]  # Spatial map at time 0

# Subregion
region = arr[100:200, 50:150, 200:300]

# Strided read
decimated = arr[::10, ::5, ::5]  # Downsample

# Out-of-core computation (avoid loading full array)
chunk_sums = []
for i in range(0, arr.shape[0], arr.chunks[0]):
    chunk = arr[i:i+arr.chunks[0], :, :]
    chunk_sums.append(chunk.sum())
total = np.sum(chunk_sums)
```

## Pattern 3: Opening and Reading Remote Zarr Data via URL

**Access Zarr stores from HTTP/S3/GCS URLs:**
```python
import zarr
import fsspec

# HTTP/HTTPS access (public data)
# Direct URL access
arr_http = zarr.open_array(
    "https://example.com/data.zarr",
    mode="r"
)

# Read data
data = arr_http[0:100, :]

# S3 access (AWS)
# Method 1: Direct S3 URL (uses default credentials)
arr_s3 = zarr.open_array(
    "s3://bucket-name/path/to/data.zarr",
    mode="r"
)

# Method 2: Explicit S3 filesystem with credentials
import s3fs

fs_s3 = s3fs.S3FileSystem(
    key="AWS_ACCESS_KEY_ID",
    secret="AWS_SECRET_ACCESS_KEY",
    endpoint_url="https://s3.us-west-2.amazonaws.com"
)

store_s3 = s3fs.S3Map(root="bucket-name/data.zarr", s3=fs_s3)
arr_s3_explicit = zarr.open_array(store_s3, mode="r")

# Method 3: Anonymous access (public buckets)
fs_anon = s3fs.S3FileSystem(anon=True)
store_anon = s3fs.S3Map(root="public-bucket/data.zarr", s3=fs_anon)
arr_public = zarr.open_array(store_anon, mode="r")

# GCS access (Google Cloud Storage)
import gcsfs

fs_gcs = gcsfs.GCSFileSystem(
    project="my-project",
    token="path/to/credentials.json"  # Or use Application Default Credentials
)

store_gcs = gcsfs.GCSMap(root="bucket-name/data.zarr", gcs=fs_gcs)
arr_gcs = zarr.open_array(store_gcs, mode="r")

# Azure Blob Storage
import adlfs

fs_azure = adlfs.AzureBlobFileSystem(
    account_name="myaccount",
    account_key="KEY"
)

store_azure = adlfs.AzureBlobFile(
    root="container/data.zarr",
    fs=fs_azure
)
arr_azure = zarr.open_array(store_azure, mode="r")
```

**Using fsspec for URL-based access:**
```python
# fsspec supports many protocols: http, s3, gcs, az, ftp, etc.
import fsspec

# Reference mapper for virtual datasets
mapper = fsspec.get_mapper("s3://bucket/data.zarr")
arr = zarr.open_array(mapper, mode="r")

# With storage options
mapper_with_opts = fsspec.get_mapper(
    "s3://bucket/data.zarr",
    s3={"anon": True}  # Anonymous access
)

# HTTP with caching
mapper_cached = fsspec.get_mapper(
    "simplecache::https://example.com/data.zarr",
    simplecache={"cache_storage": "/tmp/zarr_cache"}
)
arr_cached = zarr.open_array(mapper_cached, mode="r")
```

**Consolidated metadata for remote access:**
```python
# For Zarr v2 on cloud storage, consolidate metadata first
# This reduces the number of requests significantly

# After creating cloud store (write mode)
zarr.consolidate_metadata("s3://bucket/data.zarr")

# When reading (read mode)
arr = zarr.open_consolidated("s3://bucket/data.zarr")

# This loads all metadata in a single request instead of N+1 requests
# Critical for performance on cloud storage
```

**Streaming access patterns:**
```python
# Read specific chunks without downloading entire array
arr = zarr.open_array("s3://bucket/large-data.zarr", mode="r")

# Only fetches needed chunks
subset = arr[1000:2000, 500:1500]  # Only downloads relevant chunks

# Iterate over chunks efficiently
for i in range(0, arr.shape[0], arr.chunks[0]):
    chunk = arr[i:i+arr.chunks[0], :]
    # Process chunk
    result = chunk.mean()
```

## Pattern 4: Group Hierarchies and Organization

**Organizing data with groups:**
```python
import zarr

# Create root group
root = zarr.open_group("experiment.zarr", mode="w")

# Nested group structure
observations = root.create_group("observations")
models = root.create_group("models")
analysis = root.create_group("analysis")

# Add arrays to groups
obs_temp = observations.create_array(
    "temperature",
    shape=(365, 180, 360),
    chunks=(30, 18, 36),
    dtype="f4"
)

obs_precip = observations.create_array(
    "precipitation",
    shape=(365, 180, 360),
    chunks=(30, 18, 36),
    dtype="f4"
)

# Deep nesting
stations = observations.create_group("stations")
station_a = stations.create_group("station_a")
station_a.create_array("data", shape=(10000,), chunks=(1000,), dtype="f4")
station_a.create_array("quality_flags", shape=(10000,), chunks=(1000,), dtype="u1")

# Access via path
temp = root["observations/temperature"]
station_data = root["observations/stations/station_a/data"]

# Iterate over group contents
for name, item in observations.items():
    if isinstance(item, zarr.Array):
        print(f"Array: {name}, shape={item.shape}")
    elif isinstance(item, zarr.Group):
        print(f"Group: {name}")

# Visualize structure
print(root.tree())
```

**Group-level operations:**
```python
# Copy entire group
source = zarr.open_group("source.zarr", mode="r")
dest = zarr.open_group("dest.zarr", mode="w")
zarr.copy_all(source, dest)

# Group metadata
observations.attrs["description"] = "Observational data 2020-2024"
observations.attrs["institution"] = "Research Center"

# Find all arrays in group hierarchy
def find_arrays(group, path=""):
    for name, item in group.items():
        full_path = f"{path}/{name}" if path else name
        if isinstance(item, zarr.Array):
            yield full_path, item
        elif isinstance(item, zarr.Group):
            yield from find_arrays(item, full_path)

for path, arr in find_arrays(root):
    print(f"{path}: {arr.shape}")
```

## Pattern 5: Metadata Management

**CF-compliant metadata:**
```python
import zarr

arr = zarr.open_array("temperature.zarr", mode="w", shape=(365, 180, 360), chunks=(30, 18, 36), dtype="f4")

# Essential CF metadata
arr.attrs["long_name"] = "Air Temperature"
arr.attrs["units"] = "Celsius"
arr.attrs["standard_name"] = "air_temperature"
arr.attrs["coordinates"] = "time lat lon"

# Data quality metadata
arr.attrs["_FillValue"] = -999.0
arr.attrs["valid_range"] = [-100.0, 60.0]
arr.attrs["missing_value"] = -999.0

# Provenance metadata
arr.attrs["source"] = "Climate Model CESM 2.0"
arr.attrs["history"] = "2024-04-06: Created from model output; 2024-04-07: Applied bias correction"
arr.attrs["references"] = "Smith et al. (2023), DOI: 10.xxxx/xxxxx"
arr.attrs["comment"] = "Bias-corrected using quantile mapping against observations"

# Processing metadata
arr.attrs["processing_level"] = "Level 3"
arr.attrs["processing"] = {
    "method": "bias_correction",
    "algorithm": "quantile_mapping",
    "reference_period": "1980-2010",
    "timestamp": "2024-04-07T12:00:00Z"
}
```

**Dimensional metadata:**
```python
# Create dimension coordinate arrays
time_arr = root.create_array("time", shape=(365,), chunks=(365,), dtype="f8")
time_arr[:] = np.arange(365)
time_arr.attrs["long_name"] = "Time"
time_arr.attrs["units"] = "days since 2024-01-01"
time_arr.attrs["calendar"] = "gregorian"
time_arr.attrs["axis"] = "T"

lat_arr = root.create_array("lat", shape=(180,), chunks=(180,), dtype="f4")
lat_arr[:] = np.linspace(-90, 90, 180)
lat_arr.attrs["long_name"] = "Latitude"
lat_arr.attrs["units"] = "degrees_north"
lat_arr.attrs["axis"] = "Y"
lat_arr.attrs["standard_name"] = "latitude"

lon_arr = root.create_array("lon", shape=(360,), chunks=(360,), dtype="f4")
lon_arr[:] = np.linspace(-180, 180, 360, endpoint=False)
lon_arr.attrs["long_name"] = "Longitude"
lon_arr.attrs["units"] = "degrees_east"
lon_arr.attrs["axis"] = "X"
lon_arr.attrs["standard_name"] = "longitude"

# Link coordinates to data array
arr.attrs["coordinates"] = "time lat lon"
```

**Custom metadata patterns:**
```python
# Versioning
arr.attrs["version"] = "1.2.0"
arr.attrs["created_at"] = "2024-04-06T10:30:00Z"
arr.attrs["updated_at"] = "2024-04-06T15:45:00Z"

# Quality assurance
arr.attrs["qa_status"] = "validated"
arr.attrs["qa_checks"] = ["range_check", "spatial_consistency", "temporal_continuity"]
arr.attrs["qa_timestamp"] = "2024-04-06T16:00:00Z"

# Data access/license
arr.attrs["license"] = "CC-BY-4.0"
arr.attrs["doi"] = "10.xxxx/xxxxx"
arr.attrs["access_constraints"] = "none"
arr.attrs["use_constraints"] = "Cite as: Smith et al. (2024)"
```

## Pattern 6: Concurrent Access Patterns

**Thread-safe writes:**
```python
import zarr
from zarr import ThreadSynchronizer
import threading
import numpy as np

# Create array with thread synchronizer
arr = zarr.open_array(
    "concurrent_data.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    synchronizer=ThreadSynchronizer()
)

# Thread worker function
def write_chunk(arr, start_row, end_row):
    """Write data to specific row range."""
    data = np.random.randn(end_row - start_row, arr.shape[1]).astype("f4")
    arr[start_row:end_row, :] = data
    print(f"Thread {threading.current_thread().name}: wrote rows {start_row}-{end_row}")

# Launch threads
n_threads = 10
chunk_size = arr.shape[0] // n_threads
threads = []

for i in range(n_threads):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < n_threads - 1 else arr.shape[0]
    t = threading.Thread(target=write_chunk, args=(arr, start, end), name=f"Worker-{i}")
    threads.append(t)
    t.start()

# Wait for completion
for t in threads:
    t.join()

print("All threads complete")
```

**Process-safe writes:**
```python
import zarr
from zarr import ProcessSynchronizer
from multiprocessing import Process
import numpy as np

def write_chunk_process(store_path, sync_path, start_row, end_row):
    """Process worker - must reopen array with synchronizer."""
    arr = zarr.open_array(
        store_path,
        mode="r+",
        synchronizer=ProcessSynchronizer(sync_path)
    )
    data = np.random.randn(end_row - start_row, arr.shape[1]).astype("f4")
    arr[start_row:end_row, :] = data
    print(f"Process {start_row}-{end_row}: complete")

# Create array in main process
store_path = "multiprocess_data.zarr"
sync_path = "multiprocess_data.sync"

arr = zarr.open_array(
    store_path,
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4",
    synchronizer=ProcessSynchronizer(sync_path)
)

# Launch processes
n_processes = 10
chunk_size = arr.shape[0] // n_processes
processes = []

for i in range(n_processes):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < n_processes - 1 else arr.shape[0]
    p = Process(target=write_chunk_process, args=(store_path, sync_path, start, end))
    processes.append(p)
    p.start()

# Wait for completion
for p in processes:
    p.join()

print("All processes complete")
```

**Dask parallel writes:**
```python
import zarr
import dask.array as da

# Create Zarr array
z = zarr.open_array(
    "dask_output.zarr",
    mode="w",
    shape=(10000, 10000),
    chunks=(1000, 1000),
    dtype="f4"
)

# Create Dask array with matching chunks
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# Compute and store (parallel writes)
da.to_zarr(x, "dask_output.zarr", overwrite=True)

# Or use store method
x_computed = x.compute()
z[:] = x_computed
```

## Pattern 7: Format Conversion (v2 ↔ v3)

**Convert Zarr v2 to v3:**
```python
import zarr

# Open v2 array
z_v2 = zarr.open_array("data_v2.zarr", mode="r")

# Create v3 array with same structure
z_v3 = zarr.open_array(
    "data_v3.zarr",
    mode="w",
    shape=z_v2.shape,
    chunks=z_v2.chunks,
    dtype=z_v2.dtype,
    zarr_format=3,
    compressors="zstd"  # v3 uses 'compressors'
)

# Copy data
z_v3[:] = z_v2[:]

# Copy metadata
for key, value in z_v2.attrs.items():
    z_v3.attrs[key] = value

print(f"Converted v2 array to v3: {z_v2.shape}")
```

**Convert entire group:**
```python
def convert_v2_to_v3(source_path, dest_path):
    """Convert entire Zarr v2 group to v3."""
    source = zarr.open_group(source_path, mode="r")
    dest = zarr.open_group(dest_path, mode="w", zarr_format=3)

    def copy_recursive(src_group, dst_group):
        # Copy group attributes
        for key, value in src_group.attrs.items():
            dst_group.attrs[key] = value

        # Copy arrays and subgroups
        for name, item in src_group.items():
            if isinstance(item, zarr.Array):
                # Create v3 array
                arr_v3 = dst_group.create_array(
                    name,
                    shape=item.shape,
                    chunks=item.chunks,
                    dtype=item.dtype,
                    fill_value=item.fill_value
                )
                # Copy data
                arr_v3[:] = item[:]
                # Copy attributes
                for key, value in item.attrs.items():
                    arr_v3.attrs[key] = value
            elif isinstance(item, zarr.Group):
                # Recurse into subgroup
                subgroup = dst_group.create_group(name)
                copy_recursive(item, subgroup)

    copy_recursive(source, dest)

# Usage
convert_v2_to_v3("data_v2.zarr", "data_v3.zarr")
```

**Use zarr.copy for format conversion:**
```python
# Simple copy with format change
source = zarr.open_array("data_v2.zarr", mode="r")
dest = zarr.open_array("data_v3.zarr", mode="w", shape=source.shape, chunks=source.chunks, dtype=source.dtype, zarr_format=3)

# Copy with zarr.copy (handles decompression/recompression)
zarr.copy(source, dest)
```

## Pattern 8: Sharding Configuration (v3)

**When and how to use sharding:**
```python
import zarr

# Array with many small chunks → benefits from sharding
arr = zarr.open_array(
    "sharded_data.zarr",
    mode="w",
    shape=(100000, 100000, 100),
    chunks=(100, 100, 100),  # Small chunks
    shards=(1000, 1000, 100),  # Shards contain 10×10×1 = 100 chunks
    dtype="f4",
    zarr_format=3  # Required
)

# Explanation:
# - Without sharding: 100×100×1 = 10,000 chunk files
# - With sharding: 100×100 / (10×10) = 100 shard files
# - 100x reduction in number of files/objects

# Sharding best practices:
# 1. Shard dimensions must be multiples of chunk dimensions
# 2. Each shard should be ~10-100 MB
# 3. Useful when chunk count > 1000-10000

# Calculate shard size
import numpy as np

chunk_size = np.prod(arr.chunks) * np.dtype(arr.dtype).itemsize
shard_size = np.prod(arr.shards) * np.dtype(arr.dtype).itemsize
chunks_per_shard = np.prod(np.array(arr.shards) // np.array(arr.chunks))

print(f"Chunk size: {chunk_size / 1024**2:.2f} MB")
print(f"Shard size: {shard_size / 1024**2:.2f} MB")
print(f"Chunks per shard: {int(chunks_per_shard)}")
```

## Pattern 9: Variable-Length and Complex Data Types

**Variable-length strings:**
```python
import zarr
from numcodecs import VLenUTF8

# Create array for variable-length strings
arr = zarr.open_array(
    "strings.zarr",
    mode="w",
    shape=(1000,),
    chunks=(100,),
    dtype=object,
    object_codec=VLenUTF8()
)

# Store strings of varying lengths
arr[0] = "short"
arr[1] = "a much longer string with more content"
arr[2] = "medium length"

# Read back
strings = arr[:]
```

**Structured/record arrays:**
```python
# Define structured dtype
dtype = np.dtype([
    ("timestamp", "datetime64[s]"),
    ("station_id", "U10"),
    ("temperature", "f4"),
    ("pressure", "f4"),
    ("quality_flag", "u1")
])

arr = zarr.open_array(
    "observations.zarr",
    mode="w",
    shape=(100000,),
    chunks=(1000,),
    dtype=dtype
)

# Write data
arr["temperature"][:1000] = np.random.randn(1000) * 5 + 20
arr["pressure"][:1000] = np.random.randn(1000) * 10 + 1013
arr["quality_flag"][:1000] = 1

# Read by field
all_temps = arr["temperature"][:]
valid_temps = arr["temperature"][arr["quality_flag"][:] == 1]
```

**Complex objects with JSON codec:**
```python
from numcodecs import JSON

# Store complex Python objects
arr = zarr.open_array(
    "metadata.zarr",
    mode="w",
    shape=(100,),
    chunks=(10,),
    dtype=object,
    object_codec=JSON()
)

# Store nested dictionaries, lists, etc.
arr[0] = {
    "experiment": "A1",
    "parameters": {"temp": 25.0, "pressure": 101.3},
    "results": [1.2, 3.4, 5.6],
    "metadata": {"quality": "high", "validated": True}
}

# Read back as Python objects
data = arr[0]
assert data["parameters"]["temp"] == 25.0
```
