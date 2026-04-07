# Real-World Examples

Complete, production-ready examples demonstrating Zarr best practices for scientific workflows including cloud-native climate datasets, parallel processing, incremental updates, and data migration patterns.

## Example 1: Creating Cloud-Native Scientific Dataset

**Build a climate dataset with proper metadata and chunking for cloud access:**

```python
import zarr
import numpy as np
import pandas as pd
from pathlib import Path

def create_climate_dataset():
    """Create a cloud-optimized climate dataset with Zarr."""

    # Create root group
    root = zarr.open_group("climate_data.zarr", mode="w")

    # Dataset metadata
    root.attrs["title"] = "Global Climate Model Output"
    root.attrs["institution"] = "Climate Research Center"
    root.attrs["source"] = "CESM 2.0"
    root.attrs["experiment"] = "SSP2-4.5"
    root.attrs["frequency"] = "daily"
    root.attrs["created"] = pd.Timestamp.now().isoformat()
    root.attrs["conventions"] = "CF-1.8"

    # Define dimensions
    n_time = 365 * 10  # 10 years daily
    n_lat = 180
    n_lon = 360

    # Create dimension coordinates
    time_arr = root.create_array(
        "time",
        shape=(n_time,),
        chunks=(365,),  # One year per chunk
        dtype="f8"
    )
    time_arr[:] = np.arange(n_time)
    time_arr.attrs["long_name"] = "Time"
    time_arr.attrs["units"] = "days since 2020-01-01"
    time_arr.attrs["calendar"] = "gregorian"
    time_arr.attrs["axis"] = "T"

    lat_arr = root.create_array(
        "lat",
        shape=(n_lat,),
        chunks=(n_lat,),
        dtype="f4"
    )
    lat_arr[:] = np.linspace(-90, 90, n_lat, dtype="f4")
    lat_arr.attrs["long_name"] = "Latitude"
    lat_arr.attrs["units"] = "degrees_north"
    lat_arr.attrs["axis"] = "Y"
    lat_arr.attrs["standard_name"] = "latitude"

    lon_arr = root.create_array(
        "lon",
        shape=(n_lon,),
        chunks=(n_lon,),
        dtype="f4"
    )
    lon_arr[:] = np.linspace(-180, 180, n_lon, endpoint=False, dtype="f4")
    lon_arr.attrs["long_name"] = "Longitude"
    lon_arr.attrs["units"] = "degrees_east"
    lon_arr.attrs["axis"] = "X"
    lon_arr.attrs["standard_name"] = "longitude"

    # Create temperature variable
    # Chunking: 30 days × 18° × 36° = ~2.3 MB per chunk (good for cloud)
    temp = root.create_array(
        "temperature",
        shape=(n_time, n_lat, n_lon),
        chunks=(30, 18, 36),  # Balanced for time-series and spatial access
        dtype="f4",
        fill_value=-999.0,
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )

    # CF-compliant metadata
    temp.attrs["long_name"] = "Near-Surface Air Temperature"
    temp.attrs["standard_name"] = "air_temperature"
    temp.attrs["units"] = "Celsius"
    temp.attrs["coordinates"] = "time lat lon"
    temp.attrs["_FillValue"] = -999.0
    temp.attrs["valid_range"] = [-100.0, 60.0]
    temp.attrs["cell_methods"] = "time: mean"

    # Generate synthetic temperature data
    print("Generating temperature data...")
    for t in range(0, n_time, 30):  # Write in chunk-aligned blocks
        end_t = min(t + 30, n_time)
        # Simulate seasonal cycle + spatial pattern
        day_of_year = t % 365
        seasonal = 15 * np.cos(2 * np.pi * day_of_year / 365)

        lat_grid, lon_grid = np.meshgrid(lat_arr[:], lon_arr[:], indexing="ij")
        spatial_pattern = 20 * np.cos(np.deg2rad(lat_grid))

        time_slice = seasonal + spatial_pattern + np.random.randn(n_lat, n_lon) * 3
        temp[t:end_t, :, :] = time_slice[np.newaxis, :, :].astype("f4")

        if t % 365 == 0:
            print(f"  Year {t // 365 + 1}/10 complete")

    # Create precipitation variable
    precip = root.create_array(
        "precipitation",
        shape=(n_time, n_lat, n_lon),
        chunks=(30, 18, 36),
        dtype="f4",
        fill_value=0.0,
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )

    precip.attrs["long_name"] = "Precipitation Rate"
    precip.attrs["standard_name"] = "precipitation_flux"
    precip.attrs["units"] = "mm/day"
    precip.attrs["coordinates"] = "time lat lon"
    precip.attrs["_FillValue"] = 0.0
    precip.attrs["valid_range"] = [0.0, 500.0]

    # Generate synthetic precipitation (positive values only)
    print("Generating precipitation data...")
    for t in range(0, n_time, 30):
        end_t = min(t + 30, n_time)
        precip_data = np.abs(np.random.randn(n_lat, n_lon) * 5 + 2).astype("f4")
        precip[t:end_t, :, :] = precip_data[np.newaxis, :, :]

    # Print summary
    print("\nDataset created successfully!")
    print(f"Location: climate_data.zarr")
    print(f"Size: {temp.nbytes / 1024**3:.2f} GB (uncompressed)")
    print(f"Stored: {temp.nbytes_stored / 1024**3:.2f} GB (compressed)")
    print(f"Compression ratio: {temp.nbytes / temp.nbytes_stored:.2f}x")
    print("\nStructure:")
    print(root.tree())

    return root

if __name__ == "__main__":
    root = create_climate_dataset()
```

**Usage:**
```python
# Read the dataset
import zarr

root = zarr.open_group("climate_data.zarr", mode="r")

# Time-series access at specific location
temp = root["temperature"]
time_series = temp[:, 90, 180]  # Equator, prime meridian

# Spatial map at specific time
spatial_map = temp[0, :, :]

# Subsample for quick visualization
decimated = temp[::30, ::10, ::10]  # Monthly, 10° resolution
```

## Example 2: Migrating from HDF5 to Zarr

**Convert HDF5 file to Zarr with optimized chunking:**

```python
import h5py
import zarr
import numpy as np

def migrate_hdf5_to_zarr(hdf5_path, zarr_path, chunk_size_mb=5):
    """
    Migrate HDF5 file to Zarr with optimal chunking.

    Parameters:
    -----------
    hdf5_path : str
        Path to source HDF5 file
    zarr_path : str
        Path to destination Zarr store
    chunk_size_mb : float
        Target chunk size in megabytes
    """

    with h5py.File(hdf5_path, "r") as h5f:
        root_zarr = zarr.open_group(zarr_path, mode="w")

        def copy_attrs(h5_obj, zarr_obj):
            """Copy HDF5 attributes to Zarr."""
            for key, value in h5_obj.attrs.items():
                try:
                    # Handle numpy types
                    if isinstance(value, np.ndarray):
                        zarr_obj.attrs[key] = value.tolist()
                    else:
                        zarr_obj.attrs[key] = value
                except TypeError:
                    print(f"Warning: Could not copy attribute {key}")

        def calculate_chunks(shape, dtype, target_mb):
            """Calculate optimal chunk sizes."""
            itemsize = np.dtype(dtype).itemsize
            target_bytes = target_mb * 1024**2

            # Start with equal chunks in each dimension
            ndim = len(shape)
            chunk_items = int((target_bytes / itemsize) ** (1/ndim))

            chunks = tuple(min(chunk_items, s) for s in shape)
            return chunks

        def copy_dataset(h5_dataset, zarr_group, name):
            """Copy HDF5 dataset to Zarr array."""
            shape = h5_dataset.shape
            dtype = h5_dataset.dtype
            chunks = calculate_chunks(shape, dtype, chunk_size_mb)

            print(f"Copying {name}: shape={shape}, chunks={chunks}")

            # Create Zarr array
            zarr_array = zarr_group.create_array(
                name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=zarr.Blosc(cname="zstd", clevel=3)
            )

            # Copy data in chunks (memory-efficient)
            chunk_size = chunks[0] if len(chunks) > 0 else shape[0]
            for i in range(0, shape[0], chunk_size):
                end = min(i + chunk_size, shape[0])
                zarr_array[i:end] = h5_dataset[i:end]

            # Copy attributes
            copy_attrs(h5_dataset, zarr_array)

            return zarr_array

        def copy_group(h5_group, zarr_group, path=""):
            """Recursively copy HDF5 group to Zarr."""
            # Copy group attributes
            copy_attrs(h5_group, zarr_group)

            # Copy datasets and subgroups
            for name, item in h5_group.items():
                full_path = f"{path}/{name}" if path else name

                if isinstance(item, h5py.Dataset):
                    copy_dataset(item, zarr_group, name)
                elif isinstance(item, h5py.Group):
                    subgroup = zarr_group.create_group(name)
                    copy_group(item, subgroup, full_path)

        # Start recursive copy
        copy_group(h5f, root_zarr)

        print(f"\nMigration complete!")
        print(f"HDF5: {hdf5_path}")
        print(f"Zarr: {zarr_path}")
        print("\nZarr structure:")
        print(root_zarr.tree())

# Example usage
if __name__ == "__main__":
    # Create example HDF5 file first
    with h5py.File("example.h5", "w") as f:
        f.attrs["title"] = "Example Dataset"
        f.attrs["version"] = "1.0"

        data_group = f.create_group("data")
        data_group.create_dataset("temperature", data=np.random.randn(10000, 1000), dtype="f4")
        data_group.create_dataset("pressure", data=np.random.randn(10000, 1000), dtype="f4")

    # Migrate to Zarr
    migrate_hdf5_to_zarr("example.h5", "example.zarr", chunk_size_mb=5)
```

## Example 3: Concurrent Writes with Dask

**Parallel computation and writing to Zarr:**

```python
import zarr
import dask
import dask.array as da
import numpy as np

def parallel_zarr_computation():
    """Demonstrate parallel computation with Zarr output."""

    # Create large Zarr array
    output = zarr.open_array(
        "parallel_output.zarr",
        mode="w",
        shape=(20000, 10000),
        chunks=(1000, 1000),
        dtype="f4",
        compressor=zarr.Blosc(cname="zstd", clevel=1)  # Light compression for speed
    )

    # Set metadata
    output.attrs["description"] = "Parallel computation result"
    output.attrs["created"] = "2024-04-06"

    # Create Dask array with matching chunks
    print("Creating Dask computation graph...")
    x = da.random.random((20000, 10000), chunks=(1000, 1000))

    # Complex computation (lazy)
    result = (x ** 2 + 2 * x + 1) / (x + 0.1)
    result = result - result.mean()  # Centered

    # Compute and store in parallel
    print("Computing and writing to Zarr (parallel)...")
    with dask.config.set(scheduler="threads", num_workers=4):
        da.to_zarr(result, "parallel_output.zarr", overwrite=True)

    print(f"Complete! Stored {output.nbytes / 1024**3:.2f} GB")

    # Verify
    z = zarr.open_array("parallel_output.zarr", mode="r")
    print(f"Mean: {z[:].mean():.6f} (should be ~0)")
    print(f"Std: {z[:].std():.6f}")

    return z

def incremental_parallel_writes():
    """Demonstrate incremental parallel writes to existing Zarr array."""

    # Create empty array
    arr = zarr.open_array(
        "incremental.zarr",
        mode="w",
        shape=(10000, 10000),
        chunks=(1000, 1000),
        dtype="f4"
    )

    # Process in batches with Dask
    print("Incremental parallel writes...")
    batch_size = 2000

    for i in range(0, arr.shape[0], batch_size):
        end = min(i + batch_size, arr.shape[0])
        print(f"  Processing rows {i}-{end}...")

        # Create Dask array for this batch
        batch = da.random.random((end - i, arr.shape[1]), chunks=(1000, 1000))
        batch_result = batch * 10 + 5  # Some computation

        # Compute and write
        with dask.config.set(scheduler="threads", num_workers=4):
            arr[i:end, :] = batch_result.compute()

    print("Incremental writes complete!")

if __name__ == "__main__":
    parallel_zarr_computation()
    incremental_parallel_writes()
```

## Example 4: Time-Series Data with Optimal Chunking

**Create time-series dataset optimized for different access patterns:**

```python
import zarr
import numpy as np
import pandas as pd

def create_timeseries_dataset():
    """
    Create time-series dataset with multiple chunking strategies
    for different access patterns.
    """

    # Dimensions: 10 years hourly data, 100 stations
    n_time = 365 * 24 * 10  # 87,600 hours
    n_stations = 100

    # Create root group
    root = zarr.open_group("timeseries.zarr", mode="w")

    # Dataset metadata
    root.attrs["title"] = "Multi-Station Weather Observations"
    root.attrs["temporal_resolution"] = "hourly"
    root.attrs["spatial_coverage"] = "Regional network"
    root.attrs["start_date"] = "2015-01-01T00:00:00Z"
    root.attrs["end_date"] = "2024-12-31T23:00:00Z"

    # Create time coordinate
    time_arr = root.create_array(
        "time",
        shape=(n_time,),
        chunks=(24 * 365,),  # One year
        dtype="datetime64[h]"
    )
    time_arr[:] = pd.date_range(
        "2015-01-01",
        periods=n_time,
        freq="h"
    ).values
    time_arr.attrs["long_name"] = "Time"
    time_arr.attrs["standard_name"] = "time"

    # Create station coordinate
    station_arr = root.create_array(
        "station",
        shape=(n_stations,),
        chunks=(n_stations,),
        dtype="i4"
    )
    station_arr[:] = np.arange(n_stations)
    station_arr.attrs["long_name"] = "Station ID"

    # Version 1: Optimized for time-series access (single station over time)
    # Large time chunks, smaller station chunks
    temp_timeseries = root.create_array(
        "temperature_for_timeseries",
        shape=(n_time, n_stations),
        chunks=(24 * 30, 10),  # 30 days × 10 stations (~11 KB)
        dtype="f4",
        fill_value=-999.0,
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )
    temp_timeseries.attrs["long_name"] = "Air Temperature (time-series optimized)"
    temp_timeseries.attrs["units"] = "Celsius"
    temp_timeseries.attrs["chunking_strategy"] = "time-series access"

    # Version 2: Optimized for cross-station analysis (all stations at one time)
    # Small time chunks, large station chunks
    temp_spatial = root.create_array(
        "temperature_for_spatial",
        shape=(n_time, n_stations),
        chunks=(24, 100),  # 1 day × all stations (~10 KB)
        dtype="f4",
        fill_value=-999.0,
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )
    temp_spatial.attrs["long_name"] = "Air Temperature (spatial optimized)"
    temp_spatial.attrs["units"] = "Celsius"
    temp_spatial.attrs["chunking_strategy"] = "spatial access"

    # Generate synthetic data
    print("Generating synthetic data...")
    print(f"  Total size: {n_time * n_stations * 4 / 1024**3:.2f} GB")

    # Simulate seasonal cycle + diurnal cycle + noise
    for t in range(0, n_time, 24 * 30):  # Monthly batches
        end_t = min(t + 24 * 30, n_time)
        batch_size = end_t - t

        # Day of year for seasonal cycle
        day_of_year = (t // 24) % 365

        # Seasonal component
        seasonal = 15 * np.cos(2 * np.pi * day_of_year / 365)

        # Diurnal component (hour of day)
        hours = np.arange(batch_size) % 24
        diurnal = 5 * np.cos(2 * np.pi * hours / 24 - np.pi/2)

        # Station differences
        station_offset = np.random.randn(n_stations) * 3

        # Combine
        temps = seasonal + diurnal[:, np.newaxis] + station_offset[np.newaxis, :] + np.random.randn(batch_size, n_stations) * 2

        # Write to both arrays
        temp_timeseries[t:end_t, :] = temps.astype("f4")
        temp_spatial[t:end_t, :] = temps.astype("f4")

        if (t // (24 * 30)) % 12 == 0:
            year = 2015 + t // (24 * 365)
            print(f"  Year {year} complete")

    # Create quality flags (structured array)
    flag_dtype = np.dtype([
        ("valid", "bool"),
        ("suspect", "bool"),
        ("missing", "bool")
    ])
    flags = root.create_array(
        "quality_flags",
        shape=(n_time, n_stations),
        chunks=(24 * 30, 10),
        dtype=flag_dtype
    )
    flags.attrs["description"] = "Data quality flags"

    # Set all to valid initially
    flags["valid"][:] = True
    flags["suspect"][:] = False
    flags["missing"][:] = False

    # Randomly mark some as suspect
    suspect_mask = np.random.rand(n_time, n_stations) < 0.01
    flags["suspect"][:] = suspect_mask

    print("\nDataset created!")
    print("\nStructure:")
    print(root.tree())

    # Compare access patterns
    print("\n" + "="*60)
    print("ACCESS PATTERN COMPARISON")
    print("="*60)

    # Time-series access (one station, all times)
    print("\n1. Time-series access (Station 0, all times):")
    print(f"   Time-series optimized: {temp_timeseries.chunks} chunks")
    print(f"   Spatial optimized: {temp_spatial.chunks} chunks")

    # Chunks accessed for this pattern:
    ts_chunks_accessed = n_time // temp_timeseries.chunks[0] + 1
    sp_chunks_accessed = n_time // temp_spatial.chunks[0] * (1 if n_stations == temp_spatial.chunks[1] else n_stations // temp_spatial.chunks[1])

    print(f"   Time-series version: ~{ts_chunks_accessed} chunks accessed")
    print(f"   Spatial version: ~{sp_chunks_accessed} chunks accessed")
    print(f"   → Time-series version is {sp_chunks_accessed / ts_chunks_accessed:.1f}x more efficient")

    # Spatial access (all stations, one time)
    print("\n2. Spatial access (All stations, time=0):")
    ts_spatial_chunks = (1 if n_stations == temp_timeseries.chunks[1] else n_stations // temp_timeseries.chunks[1])
    sp_spatial_chunks = 1

    print(f"   Time-series version: ~{ts_spatial_chunks} chunks accessed")
    print(f"   Spatial version: ~{sp_spatial_chunks} chunks accessed")
    print(f"   → Spatial version is {ts_spatial_chunks / sp_spatial_chunks:.1f}x more efficient")

    return root

if __name__ == "__main__":
    root = create_timeseries_dataset()

    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)

    temp_ts = root["temperature_for_timeseries"]

    # Time-series for station 0
    station_0_ts = temp_ts[:, 0]
    print(f"\nStation 0 time-series: {len(station_0_ts)} hours")
    print(f"Mean: {station_0_ts.mean():.2f}°C")
    print(f"Min: {station_0_ts.min():.2f}°C, Max: {station_0_ts.max():.2f}°C")

    # Cross-station at one time
    temp_sp = root["temperature_for_spatial"]
    all_stations_t0 = temp_sp[0, :]
    print(f"\nAll stations at t=0: {len(all_stations_t0)} stations")
    print(f"Mean: {all_stations_t0.mean():.2f}°C")
```

## Example 5: Structured Data with Multiple Data Types

**Complex dataset combining numeric, string, and structured data:**

```python
import zarr
import numpy as np
from numcodecs import VLenUTF8, JSON

def create_mixed_dtype_dataset():
    """Create dataset with multiple data types."""

    root = zarr.open_group("mixed_types.zarr", mode="w")

    # 1. Structured array for observations
    obs_dtype = np.dtype([
        ("timestamp", "datetime64[s]"),
        ("station_id", "i4"),
        ("temperature", "f4"),
        ("humidity", "f4"),
        ("pressure", "f4"),
        ("quality", "u1")
    ])

    observations = root.create_array(
        "observations",
        shape=(100000,),
        chunks=(10000,),
        dtype=obs_dtype
    )

    # Fill with sample data
    observations["timestamp"][:] = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-01") + np.timedelta64(100000, "h"),
        dtype="datetime64[s]"
    )
    observations["station_id"][:] = np.random.randint(0, 50, 100000)
    observations["temperature"][:] = np.random.randn(100000) * 5 + 15
    observations["humidity"][:] = np.random.rand(100000) * 100
    observations["pressure"][:] = np.random.randn(100000) * 10 + 1013
    observations["quality"][:] = np.random.choice([0, 1, 2], 100000, p=[0.8, 0.15, 0.05])

    observations.attrs["description"] = "Hourly weather observations"

    # 2. Variable-length strings for station names
    station_names = root.create_array(
        "station_names",
        shape=(50,),
        chunks=(50,),
        dtype=object,
        object_codec=VLenUTF8()
    )

    station_names[:] = [f"Station_{chr(65+i//26)}{i%26}" for i in range(50)]

    # 3. JSON metadata for each station
    station_metadata = root.create_array(
        "station_metadata",
        shape=(50,),
        chunks=(50,),
        dtype=object,
        object_codec=JSON()
    )

    for i in range(50):
        station_metadata[i] = {
            "station_id": i,
            "name": station_names[i],
            "latitude": np.random.uniform(-90, 90),
            "longitude": np.random.uniform(-180, 180),
            "elevation": np.random.uniform(0, 3000),
            "sensors": ["temp", "humidity", "pressure"],
            "active": True,
            "last_calibration": "2024-01-01"
        }

    print("Mixed-type dataset created!")
    print("\nStructure:")
    print(root.tree())

    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)

    # Query 1: High quality observations
    quality_1 = observations["quality"][:] == 1
    high_quality_temps = observations["temperature"][quality_1]
    print(f"\nHigh quality observations: {len(high_quality_temps)}")
    print(f"Mean temperature: {high_quality_temps.mean():.2f}°C")

    # Query 2: Observations from specific station
    station_5 = observations["station_id"][:] == 5
    station_5_data = observations[station_5]
    print(f"\nStation 5 observations: {len(station_5_data)}")
    print(f"Mean temperature: {station_5_data['temperature'].mean():.2f}°C")

    # Query 3: Station metadata
    print(f"\nStation 0 metadata:")
    print(f"  Name: {station_metadata[0]['name']}")
    print(f"  Location: ({station_metadata[0]['latitude']:.2f}, {station_metadata[0]['longitude']:.2f})")
    print(f"  Elevation: {station_metadata[0]['elevation']:.1f} m")

    return root

if __name__ == "__main__":
    create_mixed_dtype_dataset()
```

## Example 6: Reading and Querying Remote Zarr Store from Pangeo/AWS

**Access and analyze public cloud-native datasets from Pangeo Gallery:**

```python
import zarr
import numpy as np
import s3fs

def explore_pangeo_dataset():
    """
    Access and query public Zarr datasets from Pangeo/AWS.

    This example demonstrates reading cloud-optimized climate data
    from public S3 buckets without downloading the entire dataset.
    """

    # Pangeo hosts many public datasets on AWS S3
    # Example: NOAA Optimum Interpolation Sea Surface Temperature (OISST)
    # Public bucket - no credentials needed

    # Set up anonymous S3 access
    fs = s3fs.S3FileSystem(anon=True)

    # Example: ERA5 reanalysis data (public Pangeo dataset)
    # Note: Use actual public Pangeo dataset URLs
    bucket = "your-public-pangeo-bucket"
    dataset_path = "era5-pds/zarr/2020/01/data.zarr"

    # Alternative: Use a simulated example for demonstration
    print("=" * 70)
    print("READING REMOTE ZARR DATA FROM AWS S3")
    print("=" * 70)

    # For demonstration, we'll show the pattern with a public bucket
    # In practice, replace with actual Pangeo dataset URL

    try:
        # Create S3 mapper for the dataset
        store = s3fs.S3Map(
            root=f"{bucket}/{dataset_path}",
            s3=fs,
            check=False
        )

        # Open the Zarr store (read-only)
        root = zarr.open_group(store, mode="r")

        print("\n1. Dataset Structure:")
        print(root.tree())

        # Inspect available variables
        print("\n2. Available Variables:")
        for name in root.array_keys():
            arr = root[name]
            print(f"   {name}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}")

        # Example: Read temperature data
        if "temperature" in root:
            temp = root["temperature"]

            # Check metadata
            print("\n3. Temperature Metadata:")
            for key, value in temp.attrs.items():
                print(f"   {key}: {value}")

            # Efficient spatial subset (only downloads needed chunks)
            print("\n4. Spatial Subset Query:")
            # Read specific region: [time_slice, lat_range, lon_range]
            # Only fetches chunks that intersect with this region
            subset = temp[0, 40:50, 100:110]  # Example: 10x10 degree region
            print(f"   Subset shape: {subset.shape}")
            print(f"   Mean temperature: {subset.mean():.2f}")
            print(f"   Min: {subset.min():.2f}, Max: {subset.max():.2f}")

            # Time series at specific location
            print("\n5. Time Series at Point:")
            # Only downloads 1D slice across time
            point_ts = temp[:100, 45, 105]  # 100 time steps at one location
            print(f"   Time series length: {len(point_ts)}")
            print(f"   Mean: {point_ts.mean():.2f}")

            # Statistical analysis without loading full array
            print("\n6. Chunk-wise Statistics:")
            chunk_means = []

            # Iterate over time chunks
            for i in range(0, min(365, temp.shape[0]), temp.chunks[0]):
                end = min(i + temp.chunks[0], temp.shape[0])
                chunk = temp[i:end, :, :]
                chunk_means.append(chunk.mean())

            print(f"   Processed {len(chunk_means)} time chunks")
            print(f"   Overall mean: {np.mean(chunk_means):.2f}")

    except Exception as e:
        print(f"\n⚠️  Could not access remote dataset: {e}")
        print("\nDemonstrating access pattern with local example instead...")

        # Fallback: Create local example that simulates remote access
        create_example_remote_dataset()

def create_example_remote_dataset():
    """Create a local dataset to demonstrate remote access patterns."""

    # Create a sample dataset locally
    root = zarr.open_group("demo_remote.zarr", mode="w")

    # Simulate climate data structure
    n_time = 365
    n_lat = 180
    n_lon = 360

    # Temperature array with realistic chunking for cloud access
    temp = root.create_array(
        "temperature",
        shape=(n_time, n_lat, n_lon),
        chunks=(30, 18, 36),  # Monthly time chunks, ~2 MB per chunk
        dtype="f4",
        fill_value=-999.0,
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )

    # Add CF metadata
    temp.attrs["long_name"] = "Sea Surface Temperature"
    temp.attrs["units"] = "Celsius"
    temp.attrs["standard_name"] = "sea_surface_temperature"

    # Create coordinate arrays
    time_arr = root.create_array("time", shape=(n_time,), dtype="f8")
    time_arr[:] = np.arange(n_time)
    time_arr.attrs["units"] = "days since 2020-01-01"

    lat_arr = root.create_array("lat", shape=(n_lat,), dtype="f4")
    lat_arr[:] = np.linspace(-90, 90, n_lat)
    lat_arr.attrs["units"] = "degrees_north"

    lon_arr = root.create_array("lon", shape=(n_lon,), dtype="f4")
    lon_arr[:] = np.linspace(-180, 180, n_lon, endpoint=False)
    lon_arr.attrs["units"] = "degrees_east"

    # Generate synthetic data
    print("Generating sample data...")
    for t in range(0, n_time, 30):
        end_t = min(t + 30, n_time)
        # Seasonal + spatial pattern
        lat_grid, lon_grid = np.meshgrid(lat_arr[:], lon_arr[:], indexing="ij")
        seasonal = 15 * np.cos(2 * np.pi * t / 365)
        spatial = 10 * np.cos(np.deg2rad(lat_grid))
        temps = seasonal + spatial + np.random.randn(n_lat, n_lon) * 2
        temp[t:end_t, :, :] = temps[np.newaxis, :, :].astype("f4")

    # Consolidate metadata (critical for cloud performance)
    zarr.consolidate_metadata("demo_remote.zarr")

    print("\n✅ Local example dataset created: demo_remote.zarr")
    print("\nNow demonstrating remote access patterns:")

    # Read with consolidated metadata
    root_read = zarr.open_consolidated("demo_remote.zarr", mode="r")

    print("\nDataset structure:")
    print(root_read.tree())

    # Efficient queries
    temp_read = root_read["temperature"]

    # Query 1: Spatial subset
    region = temp_read[0, 80:100, 160:200]  # 20x40 degree region
    print(f"\nSpatial subset: {region.shape}")
    print(f"Mean SST: {region.mean():.2f}°C")

    # Query 2: Time series at point
    point = temp_read[:, 90, 180]  # Equator, prime meridian
    print(f"\nTime series length: {len(point)}")
    print(f"Annual mean: {point.mean():.2f}°C")

    # Query 3: Monthly climatology
    monthly_means = []
    for month in range(12):
        # Select days for this month (simplified)
        month_data = temp_read[month*30:(month+1)*30, :, :]
        monthly_means.append(month_data.mean())

    print(f"\nMonthly climatology computed: {len(monthly_means)} months")
    print(f"Warmest month: {np.argmax(monthly_means) + 1}")
    print(f"Coolest month: {np.argmin(monthly_means) + 1}")

def demonstrate_pangeo_best_practices():
    """Best practices for accessing Pangeo datasets."""

    print("\n" + "=" * 70)
    print("BEST PRACTICES FOR PANGEO/AWS ZARR ACCESS")
    print("=" * 70)

    practices = [
        ("Use consolidated metadata", "zarr.open_consolidated() for v2 stores"),
        ("Anonymous access for public data", "s3fs.S3FileSystem(anon=True)"),
        ("Enable local caching", "fsspec caching for repeated access"),
        ("Request only needed data", "Use slicing to fetch minimal chunks"),
        ("Check chunk alignment", "Align queries with chunk boundaries"),
        ("Use Dask for large ops", "dask.array.from_zarr() for parallel processing"),
        ("Respect rate limits", "Batch requests, avoid tight loops"),
        ("Document data source", "Include dataset URL and access date")
    ]

    for i, (practice, detail) in enumerate(practices, 1):
        print(f"\n{i}. {practice}")
        print(f"   → {detail}")

    # Example: Dask integration for large-scale analysis
    print("\n" + "=" * 70)
    print("DASK INTEGRATION FOR CLOUD DATA")
    print("=" * 70)

    print("""
# Use Dask for efficient parallel access to large cloud datasets
import dask.array as da

# Wrap Zarr array as Dask array
store = s3fs.S3Map(root="bucket/data.zarr", s3=s3fs.S3FileSystem(anon=True))
darr = da.from_zarr(store)

# Lazy computations (no data loaded yet)
mean_temp = darr.mean(dim=0)
std_temp = darr.std(dim=0)

# Compute in parallel (fetches only needed chunks)
result_mean = mean_temp.compute()
result_std = std_temp.compute()

# Dask automatically:
# - Fetches chunks in parallel
# - Manages memory efficiently
# - Handles retries for network errors
    """)

if __name__ == "__main__":
    print("Exploring Pangeo datasets on AWS S3...\n")

    # Try to access remote dataset
    explore_pangeo_dataset()

    # Show best practices
    demonstrate_pangeo_best_practices()

    print("\n" + "=" * 70)
    print("Example complete!")
    print("\nFor real Pangeo datasets, visit:")
    print("  https://catalog.pangeo.io/")
    print("  https://registry.opendata.aws/ (search for 'zarr')")
    print("=" * 70)
```
