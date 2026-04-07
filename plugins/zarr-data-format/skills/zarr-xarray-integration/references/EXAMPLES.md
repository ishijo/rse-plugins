# Real-World Xarray-Zarr Integration Examples

Practical examples demonstrating production-ready workflows combining Xarray and Zarr.

## Example 1: Time-Series Append Workflow (Daily Updates)

**Scenario:** Operational weather model runs daily, appending new forecasts to existing Zarr store.

```python
import xarray as xr
import numpy as np
import zarr
from pathlib import Path

class DailyForecastArchive:
    """Manage incremental daily forecast appends to Zarr."""

    def __init__(self, zarr_path):
        self.zarr_path = Path(zarr_path)

    def initialize_archive(self, n_lat=180, n_lon=360):
        """Create initial archive with first 7 days."""
        # Day 0-6: Initialize
        ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'],
                           np.random.randn(7, n_lat, n_lon).astype('f4')),
            'pressure': (['time', 'lat', 'lon'],
                        (1013 + np.random.randn(7, n_lat, n_lon) * 5).astype('f4'))
        }, coords={
            'time': np.arange(7),
            'lat': np.linspace(-90, 90, n_lat),
            'lon': np.linspace(-180, 180, n_lon, endpoint=False)
        })

        # Add CF-compliant metadata
        ds['temperature'].attrs = {
            'long_name': '2m Temperature',
            'units': 'Celsius',
            'standard_name': 'air_temperature'
        }
        ds['pressure'].attrs = {
            'long_name': 'Surface Pressure',
            'units': 'hPa',
            'standard_name': 'surface_air_pressure'
        }

        # Encoding: chunking + compression
        encoding = {
            'temperature': {
                'chunks': (1, 90, 180),  # 1 day chunks
                'compressor': zarr.Blosc(cname='zstd', clevel=3),
                'dtype': 'float32'
            },
            'pressure': {
                'chunks': (1, 90, 180),
                'compressor': zarr.Blosc(cname='zstd', clevel=3),
                'dtype': 'float32'
            }
        }

        # Initial write
        ds.to_zarr(self.zarr_path, mode='w', encoding=encoding)

        # CRITICAL: Consolidate metadata for cloud performance
        zarr.consolidate_metadata(self.zarr_path)

        print(f"✓ Initialized archive: {self.zarr_path}")
        print(f"  Days: {ds.sizes['time']}")

    def append_daily_forecast(self, day_number):
        """Append one day of forecast data."""
        # Simulate new forecast for day N
        ds_new = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'],
                           np.random.randn(1, 180, 360).astype('f4')),
            'pressure': (['time', 'lat', 'lon'],
                        (1013 + np.random.randn(1, 180, 360) * 5).astype('f4'))
        }, coords={
            'time': [day_number],
            'lat': np.linspace(-90, 90, 180),
            'lon': np.linspace(-180, 180, 360, endpoint=False)
        })

        # Append mode with time dimension
        ds_new.to_zarr(
            self.zarr_path,
            mode='a',
            append_dim='time'
        )

        # CRITICAL: Re-consolidate after append
        zarr.consolidate_metadata(self.zarr_path)

        print(f"✓ Appended day {day_number}")

    def verify_archive(self):
        """Read and verify complete archive."""
        ds = xr.open_zarr(self.zarr_path, consolidated=True)

        print(f"\nArchive verification:")
        print(f"  Total days: {ds.sizes['time']}")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Temperature range: [{ds['temperature'].min().values:.2f}, "
              f"{ds['temperature'].max().values:.2f}]")

        return ds


# Usage
if __name__ == "__main__":
    archive = DailyForecastArchive("forecast_archive.zarr")

    # Day 1: Initialize
    archive.initialize_archive()

    # Days 2-10: Simulate daily operational runs
    for day in range(7, 17):
        archive.append_daily_forecast(day)

    # Verify
    ds_full = archive.verify_archive()

    # Query examples
    print("\nQuery examples:")

    # Latest forecast
    latest = ds_full.isel(time=-1)
    print(f"Latest forecast (day {latest.time.values}): "
          f"Mean temp = {latest['temperature'].mean().values:.2f}°C")

    # Time series at location
    point_ts = ds_full.sel(lat=0, lon=0, method='nearest')['temperature']
    print(f"Equator time series: {len(point_ts.time)} days")
```

---

## Example 2: Distributed Parallel Writes with Dask

**Scenario:** Process large climate model ensemble (100+ members) in parallel, writing to Zarr.

```python
import xarray as xr
import dask.array as da
import zarr
from dask.distributed import Client

def parallel_ensemble_processing():
    """Process 100-member climate ensemble in parallel."""

    # Step 1: Set up Dask cluster
    client = Client(n_workers=8, threads_per_worker=2)

    print("Dask dashboard:", client.dashboard_link)

    # Step 2: Create Dask-backed dataset (lazy)
    n_ensemble = 100
    n_time = 365
    n_lat = 180
    n_lon = 360

    # Lazy arrays (not computed yet)
    temperature = da.random.random(
        (n_ensemble, n_time, n_lat, n_lon),
        chunks=(10, 30, 90, 180)  # 10 members, 30 days, 90° lat, 180° lon
    ).astype('f4')

    precipitation = da.random.random(
        (n_ensemble, n_time, n_lat, n_lon),
        chunks=(10, 30, 90, 180)
    ).astype('f4') * 100  # mm/day

    # Create Xarray Dataset
    ds = xr.Dataset({
        'temperature': (['ensemble', 'time', 'lat', 'lon'], temperature),
        'precipitation': (['ensemble', 'time', 'lat', 'lon'], precipitation)
    }, coords={
        'ensemble': range(n_ensemble),
        'time': range(n_time),
        'lat': da.linspace(-90, 90, n_lat, chunks=90),
        'lon': da.linspace(-180, 180, n_lon, endpoint=False, chunks=180)
    })

    # Add metadata
    ds['temperature'].attrs = {'units': 'Celsius', 'long_name': 'Temperature'}
    ds['precipitation'].attrs = {'units': 'mm/day', 'long_name': 'Precipitation'}

    # Step 3: Define encoding
    encoding = {
        'temperature': {
            'compressor': zarr.Blosc(cname='zstd', clevel=3),
            'chunks': (10, 30, 90, 180),
            'dtype': 'float32'
        },
        'precipitation': {
            'compressor': zarr.Blosc(cname='lz4', clevel=5),
            'chunks': (10, 30, 90, 180),
            'dtype': 'float32'
        }
    }

    # Step 4: Initialize Zarr store (creates structure, no data written)
    print("Initializing Zarr store structure...")
    ds.to_zarr(
        'ensemble.zarr',
        mode='w',
        encoding=encoding,
        compute=False  # Don't compute yet
    )

    # Step 5: Parallel write with region='auto'
    print("Writing data in parallel (Dask coordinates chunk writes)...")
    write_task = ds.to_zarr(
        'ensemble.zarr',
        region='auto',  # Dask auto-detects regions
        compute=False
    )

    # Execute (Dask distributes across workers)
    write_task.compute()

    # Step 6: Consolidate metadata
    print("Consolidating metadata...")
    zarr.consolidate_metadata('ensemble.zarr')

    print("✓ Parallel write complete")

    # Step 7: Verify
    ds_read = xr.open_zarr('ensemble.zarr', consolidated=True, chunks={})

    print(f"\nVerification:")
    print(f"  Shape: {ds_read['temperature'].shape}")
    print(f"  Size: {ds_read['temperature'].nbytes / 1024**3:.2f} GB")
    print(f"  Mean temp (lazy): {ds_read['temperature'].mean()}")

    # Compute ensemble mean (parallel read)
    print("\nComputing ensemble mean (parallel)...")
    ensemble_mean = ds_read['temperature'].mean(dim='ensemble').compute()
    print(f"✓ Ensemble mean computed: shape={ensemble_mean.shape}")

    client.close()

if __name__ == "__main__":
    parallel_ensemble_processing()
```

---

## Example 3: Cloud Read/Write with S3 (Consolidated Metadata)

**Scenario:** Process global climate dataset stored on S3, apply corrections, write back.

```python
import xarray as xr
import zarr
import s3fs

def cloud_workflow_s3():
    """Read from S3, process, write corrected data back to S3."""

    # Step 1: Configure S3 access
    s3 = s3fs.S3FileSystem(
        anon=False,  # Use credentials
        client_kwargs={'region_name': 'us-west-2'}
    )

    input_bucket = "s3://climate-data-archive/raw/model-v1.zarr"
    output_bucket = "s3://climate-data-archive/corrected/model-v1.1.zarr"

    # Step 2: Read from S3 with consolidated metadata
    print("Reading from S3 (consolidated metadata = 1 network request)...")

    ds = xr.open_zarr(
        s3.get_mapper(input_bucket),
        consolidated=True,  # CRITICAL: Single metadata fetch
        chunks={}  # Preserve Zarr chunks
    )

    print(f"Dataset loaded (lazy):")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")

    # Step 3: Apply bias correction (lazy computation)
    print("\nApplying bias correction...")

    # Temperature bias correction (region-specific)
    temp = ds['temperature']

    # Correction: +2°C in tropics, 0°C elsewhere
    lat = ds['lat']
    tropical_mask = (lat >= -23.5) & (lat <= 23.5)

    temp_corrected = xr.where(
        tropical_mask,
        temp + 2.0,  # Tropical correction
        temp         # No correction elsewhere
    )

    ds_corrected = ds.copy()
    ds_corrected['temperature'] = temp_corrected

    # Update metadata
    ds_corrected.attrs['version'] = '1.1'
    ds_corrected.attrs['corrections'] = 'Tropical bias correction (+2°C)'
    ds_corrected['temperature'].attrs['bias_correction'] = 'Applied 2024-04-06'

    # Step 4: Write to S3 with same chunking
    print("Writing corrected data to S3...")

    # Create output mapper
    output_mapper = s3.get_mapper(output_bucket)

    # Encoding (match input chunks)
    encoding = {
        var: {
            'chunks': ds[var].encoding.get('chunks', ds[var].shape),
            'compressor': zarr.Blosc(cname='zstd', clevel=3)
        }
        for var in ds.data_vars
    }

    # Write (compute triggers actual data processing)
    ds_corrected.to_zarr(
        output_mapper,
        mode='w',
        encoding=encoding,
        consolidated=True  # Auto-consolidate after write
    )

    print("✓ Corrected data written to S3")

    # Step 5: Verify output
    print("\nVerifying output on S3...")
    ds_verify = xr.open_zarr(
        s3.get_mapper(output_bucket),
        consolidated=True
    )

    print(f"  Version: {ds_verify.attrs.get('version')}")
    print(f"  Corrections: {ds_verify.attrs.get('corrections')}")

    # Compare tropical mean before/after
    tropical_slice = ds_verify.sel(lat=slice(-23.5, 23.5))
    tropical_mean = tropical_slice['temperature'].mean().compute()

    print(f"  Tropical mean temp: {tropical_mean.values:.2f}°C")


def local_to_s3_upload():
    """Upload local Zarr store to S3 with optimization."""

    # Local dataset
    ds_local = xr.open_zarr('local_dataset.zarr', consolidated=True)

    # S3 configuration
    s3 = s3fs.S3FileSystem(
        client_kwargs={'region_name': 'us-east-1'}
    )
    s3_mapper = s3.get_mapper('s3://my-bucket/dataset.zarr')

    # Write to S3 (triggers upload)
    print("Uploading to S3...")
    ds_local.to_zarr(
        s3_mapper,
        mode='w',
        consolidated=True
    )

    print("✓ Upload complete")


if __name__ == "__main__":
    # Example 1: Cloud processing workflow
    cloud_workflow_s3()

    # Example 2: Local to S3 upload
    # local_to_s3_upload()
```

---

## Example 4: Region-Based Updates (Data Corrections)

**Scenario:** Correct erroneous data in specific time/space regions without rewriting entire dataset.

```python
import xarray as xr
import numpy as np
import zarr

def setup_example_dataset():
    """Create dataset with intentional errors."""

    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'],
                       np.random.randn(365, 180, 360).astype('f4'))
    }, coords={
        'time': np.arange(365),
        'lat': np.linspace(-90, 90, 180),
        'lon': np.linspace(-180, 180, 360, endpoint=False)
    })

    # Inject errors: days 100-109 have bad data
    ds['temperature'][100:110, :, :] = -999.0  # Missing value placeholder

    # Inject spatial error: Antarctic region (lat < -80) has sensor drift
    lat_mask = ds['lat'] < -80
    lat_indices = np.where(lat_mask.values)[0]
    ds['temperature'][:, lat_indices, :] += 10.0  # 10°C drift

    # Write with chunking
    encoding = {
        'temperature': {
            'chunks': (30, 90, 180),
            'compressor': zarr.Blosc(cname='zstd', clevel=3)
        }
    }

    ds.to_zarr('dataset_with_errors.zarr', mode='w', encoding=encoding)
    zarr.consolidate_metadata('dataset_with_errors.zarr')

    print("✓ Created dataset with intentional errors")
    print(f"  Error 1: Days 100-109 have missing data (-999)")
    print(f"  Error 2: Antarctic (lat < -80°) has +10°C sensor drift")


def temporal_region_correction():
    """Fix temporal region (days 100-109) with corrected data."""

    print("\n=== Temporal Region Correction ===")

    # Generate corrected data for days 100-109
    ds_corrected = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'],
                       np.random.randn(10, 180, 360).astype('f4'))
    }, coords={
        'time': np.arange(100, 110),
        'lat': np.linspace(-90, 90, 180),
        'lon': np.linspace(-180, 180, 360, endpoint=False)
    })

    # Write to specific time region
    ds_corrected.to_zarr(
        'dataset_with_errors.zarr',
        mode='a',
        region={'time': slice(100, 110)}  # Only update days 100-109
    )

    print("✓ Updated days 100-109 with corrected data")

    # Verify
    ds = xr.open_zarr('dataset_with_errors.zarr')
    print(f"  Days 100-109 mean: {ds['temperature'][100:110].mean().values:.2f}")
    print(f"  (Should no longer be -999)")


def spatial_region_correction():
    """Fix spatial region (Antarctic) by removing sensor drift."""

    print("\n=== Spatial Region Correction ===")

    # Read current data
    ds = xr.open_zarr('dataset_with_errors.zarr')

    # Select Antarctic region
    antarctic = ds.sel(lat=slice(-90, -80))

    # Remove +10°C drift
    antarctic_corrected = antarctic.copy()
    antarctic_corrected['temperature'] = antarctic['temperature'] - 10.0

    # Determine region indices
    lat_start = np.where(ds['lat'].values <= -80)[0][0]
    lat_end = np.where(ds['lat'].values <= -90)[0][-1] + 1

    print(f"  Antarctic region: lat indices {lat_start}-{lat_end}")

    # Write to spatial region
    antarctic_corrected.to_zarr(
        'dataset_with_errors.zarr',
        mode='a',
        region={'lat': slice(lat_start, lat_end)}
    )

    print("✓ Removed +10°C drift from Antarctic region")

    # Verify
    ds_verify = xr.open_zarr('dataset_with_errors.zarr')
    antarctic_verify = ds_verify.sel(lat=slice(-90, -80))
    print(f"  Antarctic mean: {antarctic_verify['temperature'].mean().values:.2f}")


def spatiotemporal_region_correction():
    """Fix specific space-time region (e.g., hurricane artifact)."""

    print("\n=== Spatiotemporal Region Correction ===")

    # Scenario: Hurricane artifact in Caribbean (days 200-210, lat 10-30°N, lon 260-290°E)

    # Generate corrected data for specific region
    ds_patch = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'],
                       np.random.randn(11, 20, 30).astype('f4'))
    }, coords={
        'time': np.arange(200, 211),
        'lat': np.linspace(10, 30, 20),
        'lon': np.linspace(260, 290, 30, endpoint=False)
    })

    # Determine region indices
    ds = xr.open_zarr('dataset_with_errors.zarr')

    time_start, time_end = 200, 211
    lat_start = np.argmin(np.abs(ds['lat'].values - 10))
    lat_end = np.argmin(np.abs(ds['lat'].values - 30)) + 1
    lon_start = np.argmin(np.abs(ds['lon'].values - 260))
    lon_end = np.argmin(np.abs(ds['lon'].values - 290)) + 1

    print(f"  Region: time={time_start}:{time_end}, "
          f"lat={lat_start}:{lat_end}, lon={lon_start}:{lon_end}")

    # Write to specific space-time region
    ds_patch.to_zarr(
        'dataset_with_errors.zarr',
        mode='a',
        region={
            'time': slice(time_start, time_end),
            'lat': slice(lat_start, lat_end),
            'lon': slice(lon_start, lon_end)
        }
    )

    print("✓ Corrected Caribbean hurricane artifact (days 200-210)")


def verification_report():
    """Generate verification report after all corrections."""

    print("\n=== Verification Report ===")

    ds = xr.open_zarr('dataset_with_errors.zarr')

    # Check temporal correction
    days_100_109 = ds['temperature'][100:110]
    print(f"Days 100-109:")
    print(f"  Mean: {days_100_109.mean().values:.2f}")
    print(f"  Contains -999: {(days_100_109 == -999.0).any().values}")

    # Check spatial correction
    antarctic = ds.sel(lat=slice(-90, -80))
    print(f"\nAntarctic (lat < -80°):")
    print(f"  Mean: {antarctic['temperature'].mean().values:.2f}")

    # Overall statistics
    print(f"\nOverall:")
    print(f"  Global mean: {ds['temperature'].mean().values:.2f}")
    print(f"  Global std: {ds['temperature'].std().values:.2f}")


if __name__ == "__main__":
    # Step 1: Create dataset with errors
    setup_example_dataset()

    # Step 2: Apply corrections
    temporal_region_correction()
    spatial_region_correction()
    spatiotemporal_region_correction()

    # Step 3: Verify
    verification_report()

    print("\n✓ All region-based corrections complete")
```

---

## Cross-References

- **zarr-fundamentals** — Zarr basics, chunking, compression
- **cloud-storage-backends** — S3/GCS/Azure configuration
- **compression-codecs** — Choosing compressors for encoding
- **xarray-for-multidimensional-data** (scientific-domain-applications) — Xarray fundamentals
