#!/usr/bin/env python3
"""
Production-ready data migration script for HDF5/NetCDF to Zarr.

Supports:
- HDF5 (.h5, .hdf5) and NetCDF (.nc, .nc4) sources
- Local or S3 destination
- Configurable chunking and compression
- Validation with shape, dtype, and random sampling
- Progress reporting and error handling

Usage:
    python migration-template.py --source data.h5 --dest output.zarr --variable temperature
    python migration-template.py --source data.nc --dest s3://bucket/data.zarr --variable temperature --chunks 10,90,180
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import zarr
import h5py
import xarray as xr


def detect_source_format(source_path):
    """Detect source file format from extension."""
    suffix = Path(source_path).suffix.lower()
    if suffix in ['.h5', '.hdf5']:
        return 'hdf5'
    elif suffix in ['.nc', '.nc4']:
        return 'netcdf'
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .h5/.hdf5 or .nc/.nc4")


def get_file_size(path):
    """Get file size in bytes."""
    if isinstance(path, str) and path.startswith('s3://'):
        return None  # Can't easily get S3 size
    return Path(path).stat().st_size


def format_size(size_bytes):
    """Format bytes to human-readable size."""
    if size_bytes is None:
        return "unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_zarr_size(zarr_path):
    """Calculate total Zarr store size."""
    if isinstance(zarr_path, str) and zarr_path.startswith('s3://'):
        return None  # Skip for S3

    total_size = 0
    zarr_dir = Path(zarr_path)
    if zarr_dir.exists():
        for file in zarr_dir.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
    return total_size


def migrate_hdf5_to_zarr(source_path, dest_path, variable, chunks, compressor):
    """Migrate HDF5 dataset to Zarr."""
    print(f"Opening HDF5 source: {source_path}")
    h5f = h5py.File(source_path, 'r')

    if variable not in h5f:
        h5f.close()
        raise ValueError(f"Variable '{variable}' not found. Available: {list(h5f.keys())}")

    source_dataset = h5f[variable]
    print(f"  Shape: {source_dataset.shape}")
    print(f"  Dtype: {source_dataset.dtype}")
    print(f"  HDF5 chunks: {source_dataset.chunks}")

    # Create Zarr group
    dest_group = zarr.open_group(dest_path, mode='a')

    # Determine chunks (use source chunks if not specified)
    if chunks is None:
        if source_dataset.chunks is not None:
            chunks = source_dataset.chunks
        else:
            # Auto-chunk: aim for ~4 MB chunks
            dtype_size = np.dtype(source_dataset.dtype).itemsize
            target_chunk_bytes = 4 * 1024 * 1024
            target_elements = target_chunk_bytes // dtype_size
            elements_per_dim = int(target_elements ** (1.0 / len(source_dataset.shape)))
            chunks = tuple(min(elements_per_dim, s) for s in source_dataset.shape)

    print(f"  Target chunks: {chunks}")
    print(f"  Compressor: {compressor}")

    # Copy data with zarr.copy (handles rechunking)
    print(f"Copying data to Zarr...")
    start_time = time.time()

    zarr.copy(
        source_dataset,
        dest_group,
        name=variable,
        chunks=chunks,
        compressor=compressor
    )

    elapsed = time.time() - start_time
    print(f"  Copied in {elapsed:.2f}s")

    # Copy attributes
    dest_array = dest_group[variable]
    for key, value in source_dataset.attrs.items():
        try:
            dest_array.attrs[key] = value
        except Exception as e:
            print(f"  Warning: Could not copy attribute '{key}': {e}")

    h5f.close()
    return dest_array


def migrate_netcdf_to_zarr(source_path, dest_path, variable, chunks, compressor):
    """Migrate NetCDF dataset to Zarr using Xarray."""
    print(f"Opening NetCDF source: {source_path}")
    ds = xr.open_dataset(source_path)

    if variable not in ds:
        ds.close()
        raise ValueError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")

    data_array = ds[variable]
    print(f"  Shape: {data_array.shape}")
    print(f"  Dtype: {data_array.dtype}")
    print(f"  Dimensions: {data_array.dims}")

    # Determine chunks
    if chunks is None:
        # Use dask chunks if present, else auto-chunk
        if hasattr(data_array.data, 'chunks'):
            chunks = data_array.data.chunks
        else:
            dtype_size = np.dtype(data_array.dtype).itemsize
            target_chunk_bytes = 4 * 1024 * 1024
            target_elements = target_chunk_bytes // dtype_size
            elements_per_dim = int(target_elements ** (1.0 / len(data_array.shape)))
            chunks = tuple(min(elements_per_dim, s) for s in data_array.shape)

    print(f"  Target chunks: {chunks}")
    print(f"  Compressor: {compressor}")

    # Create encoding
    encoding = {
        variable: {
            'chunks': chunks,
            'compressor': compressor
        }
    }

    # Check if destination is S3
    storage_options = None
    if isinstance(dest_path, str) and dest_path.startswith('s3://'):
        storage_options = {'anon': False}

    print(f"Writing to Zarr...")
    start_time = time.time()

    if storage_options:
        ds.to_zarr(dest_path, mode='w', encoding=encoding, storage_options=storage_options)
    else:
        ds.to_zarr(dest_path, mode='w', encoding=encoding)

    elapsed = time.time() - start_time
    print(f"  Written in {elapsed:.2f}s")

    # Consolidate metadata
    print("Consolidating metadata...")
    if storage_options:
        zarr.consolidate_metadata(dest_path, storage_options=storage_options)
    else:
        zarr.consolidate_metadata(dest_path)

    ds.close()

    # Return array for validation
    z = zarr.open(dest_path, mode='r')
    return z[variable]


def validate_migration(source_path, source_format, dest_array, variable):
    """Validate migrated data matches source."""
    print("\nValidating migration...")

    # Open source
    if source_format == 'hdf5':
        h5f = h5py.File(source_path, 'r')
        source_array = h5f[variable]
    else:  # netcdf
        ds = xr.open_dataset(source_path)
        source_array = ds[variable].values

    # Check 1: Shape
    print(f"  Checking shape...")
    if hasattr(source_array, 'shape'):
        assert source_array.shape == dest_array.shape, \
            f"Shape mismatch: {source_array.shape} vs {dest_array.shape}"
        print(f"    ✓ Shape matches: {dest_array.shape}")

    # Check 2: Dtype
    print(f"  Checking dtype...")
    source_dtype = source_array.dtype if hasattr(source_array, 'dtype') else type(source_array)
    assert source_dtype == dest_array.dtype, \
        f"Dtype mismatch: {source_dtype} vs {dest_array.dtype}"
    print(f"    ✓ Dtype matches: {dest_array.dtype}")

    # Check 3: Random sampling (10 points)
    print(f"  Checking data integrity (10 random samples)...")
    np.random.seed(42)

    shape = dest_array.shape
    n_samples = min(10, np.prod(shape))

    for i in range(n_samples):
        # Generate random index
        idx = tuple(np.random.randint(0, s) for s in shape)

        # Read values
        if source_format == 'hdf5':
            source_val = source_array[idx]
        else:
            source_val = source_array[idx]

        dest_val = dest_array[idx]

        # Compare (allow small floating point differences)
        if np.isnan(source_val) and np.isnan(dest_val):
            continue

        try:
            np.testing.assert_allclose(
                source_val, dest_val,
                rtol=1e-5, atol=1e-8,
                err_msg=f"Value mismatch at {idx}"
            )
        except AssertionError as e:
            print(f"    ✗ {e}")
            if source_format == 'hdf5':
                h5f.close()
            else:
                ds.close()
            raise

    print(f"    ✓ All {n_samples} samples match")

    # Check 4: Metadata (if HDF5)
    if source_format == 'hdf5':
        print(f"  Checking metadata...")
        attrs_match = 0
        for key in source_array.attrs:
            if key in dest_array.attrs:
                attrs_match += 1
        print(f"    ✓ {attrs_match}/{len(source_array.attrs.keys())} attributes preserved")

    # Cleanup
    if source_format == 'hdf5':
        h5f.close()
    else:
        ds.close()

    print("\n✓ Validation passed!")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate HDF5/NetCDF data to Zarr format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local HDF5 to local Zarr
  python migration-template.py --source data.h5 --dest output.zarr --variable temperature

  # NetCDF to S3
  python migration-template.py --source data.nc --dest s3://bucket/data.zarr --variable temperature

  # Custom chunking and compression
  python migration-template.py --source data.h5 --dest output.zarr --variable temperature \\
      --chunks 10,90,180 --compression-level 5
        """
    )

    parser.add_argument('--source', required=True, help='Source HDF5 or NetCDF file path')
    parser.add_argument('--dest', required=True, help='Destination Zarr path (local or s3://)')
    parser.add_argument('--variable', required=True, help='Variable name to migrate')
    parser.add_argument('--chunks', type=str, help='Comma-separated chunk sizes (e.g., "10,90,180")')
    parser.add_argument('--compression-level', type=int, default=3, help='Zstd compression level (default: 3)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip post-migration validation')

    args = parser.parse_args()

    # Parse chunks
    chunks = None
    if args.chunks:
        try:
            chunks = tuple(int(x.strip()) for x in args.chunks.split(','))
        except ValueError:
            print(f"Error: Invalid chunk format. Use comma-separated integers (e.g., '10,90,180')")
            sys.exit(1)

    # Setup compressor
    compressor = zarr.Blosc(cname='zstd', clevel=args.compression_level, shuffle=zarr.Blosc.BITSHUFFLE)

    # Detect format
    try:
        source_format = detect_source_format(args.source)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 70)
    print("DATA MIGRATION TO ZARR")
    print("=" * 70)
    print(f"Source: {args.source} ({source_format.upper()})")
    print(f"Destination: {args.dest}")
    print(f"Variable: {args.variable}")
    print()

    # Get source size
    source_size = get_file_size(args.source)
    if source_size:
        print(f"Source file size: {format_size(source_size)}")

    # Migrate
    start_total = time.time()

    try:
        if source_format == 'hdf5':
            dest_array = migrate_hdf5_to_zarr(args.source, args.dest, args.variable, chunks, compressor)
        else:
            dest_array = migrate_netcdf_to_zarr(args.source, args.dest, args.variable, chunks, compressor)

        # Validation
        if not args.skip_validation:
            validate_migration(args.source, source_format, dest_array, args.variable)

        # Report
        total_elapsed = time.time() - start_total
        dest_size = get_zarr_size(args.dest)

        print("\n" + "=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_elapsed:.2f}s")

        if source_size and dest_size:
            ratio = source_size / dest_size
            print(f"Source size: {format_size(source_size)}")
            print(f"Destination size: {format_size(dest_size)}")
            print(f"Compression ratio: {ratio:.2f}x")

        print(f"\nOutput: {args.dest}")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
