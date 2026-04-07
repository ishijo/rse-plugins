#!/usr/bin/env python3
"""
Zarr Quickstart Example

Runnable end-to-end demonstration of Zarr fundamentals:
- Array creation and initialization
- Writing and reading data
- Group hierarchies
- Metadata management
- Indexing modes
- Compression configuration

Dependencies: zarr, numpy

Usage:
    python zarr-quickstart.py
"""

import numpy as np
import zarr
from pathlib import Path


def main():
    """Run complete Zarr workflow demonstration."""

    print("=" * 60)
    print("ZARR FUNDAMENTALS QUICKSTART")
    print("=" * 60)

    # Clean up any existing demo data
    demo_path = Path("demo.zarr")
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)

    # ========================================
    # 1. ARRAY CREATION
    # ========================================
    print("\n1. Creating Zarr arrays...")

    # Create a simple 2D array
    z_simple = zarr.zeros(
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4",
        store="demo.zarr/simple"
    )
    print(f"   Created simple array: shape={z_simple.shape}, chunks={z_simple.chunks}")

    # Create 3D array with compression
    z_3d = zarr.open_array(
        "demo.zarr/data_3d",
        mode="w",
        shape=(100, 200, 300),
        chunks=(10, 20, 30),
        dtype="f4",
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )
    print(f"   Created 3D array: shape={z_3d.shape}, dtype={z_3d.dtype}")
    print(f"   Compression: {z_3d.compressor}")

    # ========================================
    # 2. WRITING DATA
    # ========================================
    print("\n2. Writing data...")

    # Fill simple array with random data
    data_2d = np.random.randn(1000, 1000).astype("f4")
    z_simple[:] = data_2d
    print(f"   Wrote {data_2d.nbytes / 1024**2:.2f} MB to simple array")

    # Write to 3D array using slicing
    for i in range(100):
        z_3d[i, :, :] = np.random.randn(200, 300).astype("f4") * 10 + 15
    print(f"   Wrote 100 time slices to 3D array")

    # Partial write
    z_3d[50:60, 100:120, 150:180] = 99.0
    print(f"   Filled sub-region with value 99.0")

    # ========================================
    # 3. READING DATA
    # ========================================
    print("\n3. Reading data...")

    # Read entire array
    data_full = z_simple[:]
    print(f"   Read full array: shape={data_full.shape}")

    # Read slice
    row_slice = z_simple[0, :]
    print(f"   Read single row: shape={row_slice.shape}")

    # Read sub-region
    block = z_3d[10:20, 50:100, 100:150]
    print(f"   Read 3D block: shape={block.shape}")

    # Check the filled region
    filled_region = z_3d[50:60, 100:120, 150:180]
    assert np.all(filled_region == 99.0), "Filled region check failed"
    print(f"   Verified filled region contains 99.0")

    # ========================================
    # 4. GROUP HIERARCHIES
    # ========================================
    print("\n4. Creating group hierarchies...")

    # Create root group
    root = zarr.open_group("demo.zarr", mode="a")

    # Create nested groups
    observations = root.create_group("observations", overwrite=True)
    models = root.create_group("models", overwrite=True)

    # Add arrays to groups
    obs_temp = observations.create_array(
        "temperature",
        shape=(365, 100, 100),
        chunks=(30, 10, 10),
        dtype="f4",
        fill_value=-999.0
    )

    obs_precip = observations.create_array(
        "precipitation",
        shape=(365, 100, 100),
        chunks=(30, 10, 10),
        dtype="f4",
        fill_value=0.0
    )

    # Create subgroup
    station_a = observations.create_group("station_a", overwrite=True)
    station_data = station_a.create_array(
        "measurements",
        shape=(10000,),
        chunks=(1000,),
        dtype="f4"
    )

    print("   Created group hierarchy:")
    print(root.tree())

    # ========================================
    # 5. METADATA MANAGEMENT
    # ========================================
    print("\n5. Adding metadata...")

    # CF-compliant metadata on temperature array
    obs_temp.attrs["long_name"] = "Air Temperature"
    obs_temp.attrs["units"] = "Celsius"
    obs_temp.attrs["standard_name"] = "air_temperature"
    obs_temp.attrs["coordinates"] = "time lat lon"
    obs_temp.attrs["_FillValue"] = -999.0
    obs_temp.attrs["valid_range"] = [-100.0, 60.0]

    # Metadata on precipitation array
    obs_precip.attrs["long_name"] = "Precipitation"
    obs_precip.attrs["units"] = "mm"
    obs_precip.attrs["standard_name"] = "precipitation_amount"

    # Group-level metadata
    observations.attrs["project"] = "Climate Study 2024"
    observations.attrs["institution"] = "Research Center"
    observations.attrs["contact"] = "researcher@example.org"

    print(f"   Temperature metadata: {dict(obs_temp.attrs)}")
    print(f"   Group metadata: {dict(observations.attrs)}")

    # ========================================
    # 6. INDEXING MODES
    # ========================================
    print("\n6. Demonstrating indexing modes...")

    # Create test array
    test_arr = zarr.zeros((1000, 500), chunks=(100, 50), dtype="f4", store="demo.zarr/indexing")
    test_arr[:] = np.arange(1000 * 500).reshape(1000, 500)

    # Mode 1: Basic slicing
    basic_slice = test_arr[10:20, 100:200]
    print(f"   Basic slicing: shape={basic_slice.shape}")

    # Mode 2: Coordinate indexing (vindex)
    coord_indices = [0, 10, 50, 99, 500]
    coord_select = test_arr.vindex[coord_indices, 100]
    print(f"   Coordinate indexing: shape={coord_select.shape}")

    # Mode 3: Boolean mask (vindex)
    first_col = test_arr[:, 0]
    mask = first_col > 500000
    mask_select = test_arr.vindex[mask, 0]
    print(f"   Boolean mask: selected {len(mask_select)} elements")

    # Mode 4: Orthogonal indexing (oindex)
    rows = [0, 5, 10]
    cols = [100, 200, 300]
    orth_select = test_arr.oindex[rows, cols]
    print(f"   Orthogonal indexing: shape={orth_select.shape}")

    # Mode 5: Block indexing
    block_00 = test_arr.blocks[0, 0]
    print(f"   Block indexing: first block shape={block_00.shape}")

    # Mode 6: Structured arrays
    struct_dtype = np.dtype([
        ("name", "U20"),
        ("temperature", "f4"),
        ("humidity", "f4")
    ])
    struct_arr = zarr.zeros((100,), chunks=(10,), dtype=struct_dtype, store="demo.zarr/structured")
    struct_arr["temperature"][:] = np.random.randn(100) * 5 + 20
    struct_arr["humidity"][:] = np.random.rand(100) * 100
    temps = struct_arr["temperature"][:]
    print(f"   Structured field access: mean temperature={temps.mean():.2f}")

    # ========================================
    # 7. COMPRESSION COMPARISON
    # ========================================
    print("\n7. Comparing compression codecs...")

    # Create test data (1000x1000 float32)
    test_data = np.random.randn(1000, 1000).astype("f4")

    # No compression
    z_none = zarr.open_array(
        "demo.zarr/compress_none",
        mode="w",
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4",
        compressor=None
    )
    z_none[:] = test_data

    # Blosc+Zstd (default for v2)
    z_blosc = zarr.open_array(
        "demo.zarr/compress_blosc",
        mode="w",
        shape=(1000, 1000),
        chunks=(100, 100),
        dtype="f4",
        compressor=zarr.Blosc(cname="zstd", clevel=3)
    )
    z_blosc[:] = test_data

    # Report sizes
    print(f"   Uncompressed: {z_none.nbytes / 1024**2:.2f} MB")
    print(f"   Blosc+Zstd:   {z_blosc.nbytes_stored / 1024**2:.2f} MB")
    ratio = z_none.nbytes / z_blosc.nbytes_stored
    print(f"   Compression ratio: {ratio:.2f}x")

    # ========================================
    # 8. SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Created Zarr store at: {demo_path}")
    print(f"Total arrays created: {len(list(root.arrays()))}")
    print(f"Total groups: {len(list(root.groups()))}")
    print("\nGroup structure:")
    print(root.tree())

    print("\nQuickstart complete! Explore the created arrays:")
    print("  >>> import zarr")
    print("  >>> root = zarr.open_group('demo.zarr')")
    print("  >>> print(root.tree())")
    print("  >>> temp = root['observations/temperature']")
    print("  >>> print(temp.attrs.asdict())")


if __name__ == "__main__":
    main()
