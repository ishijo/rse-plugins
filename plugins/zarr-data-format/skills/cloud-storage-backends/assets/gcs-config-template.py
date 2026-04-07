#!/usr/bin/env python3
"""
Google Cloud Storage (GCS) Zarr Configuration Template

Production-ready template for storing and accessing Zarr arrays on GCS.
Demonstrates authentication, metadata consolidation, and error handling.

Usage:
    # Option 1: Application Default Credentials (recommended)
    gcloud auth application-default login
    python gcs-config-template.py

    # Option 2: Service Account
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
    export GCP_PROJECT="my-project"
    python gcs-config-template.py

Dependencies: zarr, fsspec, gcsfs, numpy
"""

import os
import sys
import zarr
import numpy as np
from fsspec import filesystem


# ============================================================================
# CONFIGURATION
# ============================================================================

GCS_BUCKET = os.getenv('GCS_BUCKET', 'my-zarr-bucket')
GCS_PREFIX = os.getenv('GCS_PREFIX', 'data/example-array.zarr')
GCP_PROJECT = os.getenv('GCP_PROJECT', None)  # Required for some operations

# Authentication
# - Default: Uses Application Default Credentials (gcloud auth application-default login)
# - Service Account: Set GOOGLE_APPLICATION_CREDENTIALS env var
# - Anonymous: For public buckets
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', None)
ANONYMOUS = os.getenv('GCS_ANONYMOUS', 'false').lower() == 'true'

# Performance tuning
ENABLE_CACHING = True
CACHE_DIR = '/tmp/zarr-gcs-cache'
CONSOLIDATE_METADATA = True


# ============================================================================
# GCS FILESYSTEM CONFIGURATION
# ============================================================================

def create_gcs_filesystem(project=GCP_PROJECT, anonymous=ANONYMOUS):
    """
    Create fsspec GCS filesystem with proper authentication.

    Parameters
    ----------
    project : str or None
        GCP project ID (required for some operations)
    anonymous : bool
        Use anonymous access (for public buckets)

    Returns
    -------
    fs : fsspec.AbstractFileSystem
        Configured GCS filesystem
    """
    if anonymous:
        print(f"Using anonymous GCS access (public bucket)")
        fs = filesystem('gcs', token='anon')
    elif SERVICE_ACCOUNT_JSON:
        print(f"Using service account: {SERVICE_ACCOUNT_JSON}")
        fs = filesystem(
            'gcs',
            project=project,
            token=SERVICE_ACCOUNT_JSON
        )
    else:
        print(f"Using Application Default Credentials")
        # Uses gcloud auth application-default login
        fs = filesystem('gcs', project=project)

    return fs


# ============================================================================
# WRITE EXAMPLE: Create and Upload Zarr Array
# ============================================================================

def write_to_gcs(bucket, prefix):
    """
    Create a Zarr array and write to GCS with metadata consolidation.
    """
    print("\n" + "="*80)
    print("WRITING ZARR ARRAY TO GCS")
    print("="*80)

    # Create GCS store
    gcs_path = f'gs://{bucket}/{prefix}'
    print(f"\nTarget: {gcs_path}")
    if GCP_PROJECT:
        print(f"Project: {GCP_PROJECT}")

    fs = create_gcs_filesystem(GCP_PROJECT)
    store = zarr.storage.FsspecStore(f'{bucket}/{prefix}', fs=fs)

    # Create Zarr group
    print("\nCreating Zarr group...")
    root = zarr.open_group(store, mode='w')

    # Create sample dataset
    print("Creating precipitation array (365 x 180 x 360)...")
    precip = root.create_dataset(
        'precipitation',
        shape=(365, 180, 360),
        chunks=(10, 90, 180),  # ~1.4 MB chunks
        dtype='f4',
        compressor=zarr.codecs.Blosc(cname='zstd', clevel=3),
        fill_value=-9999.0
    )

    # Add metadata
    precip.attrs['units'] = 'mm/day'
    precip.attrs['long_name'] = 'Daily precipitation rate'
    precip.attrs['coordinates'] = 'time lat lon'
    precip.attrs['institution'] = 'Example Climate Center'

    # Write synthetic data
    print("Writing synthetic data...")
    data = np.abs(np.random.randn(365, 180, 360).astype('f4')) * 5.0
    precip[:] = data

    print(f"✓ Wrote {precip.nbytes / 1024**2:.1f} MB (uncompressed)")
    print(f"✓ Stored as {precip.nbytes_stored / 1024**2:.1f} MB (compressed)")
    print(f"✓ Compression ratio: {precip.nbytes / precip.nbytes_stored:.2f}x")

    # Consolidate metadata (CRITICAL for cloud performance)
    if CONSOLIDATE_METADATA:
        print("\nConsolidating metadata...")
        zarr.consolidate_metadata(store)
        print("✓ Metadata consolidated (.zmetadata created)")

    print("\n✓ Write complete!")
    return gcs_path


# ============================================================================
# READ EXAMPLE: Access Zarr Array from GCS
# ============================================================================

def read_from_gcs(bucket, prefix, use_caching=ENABLE_CACHING):
    """
    Read Zarr array from GCS with optional caching.
    """
    print("\n" + "="*80)
    print("READING ZARR ARRAY FROM GCS")
    print("="*80)

    if use_caching:
        # Use file cache for persistent local cache
        gcs_path = f'filecache::gs://{bucket}/{prefix}'
        storage_options = {
            'gcs': {
                'token': 'anon' if ANONYMOUS else None,
                'project': GCP_PROJECT
            },
            'filecache': {
                'cache_storage': CACHE_DIR,
                'expiry_time': 86400  # 24 hours
            }
        }
        print(f"\nSource: {gcs_path}")
        print(f"Cache: {CACHE_DIR} (24 hour expiry)")
    else:
        gcs_path = f'gs://{bucket}/{prefix}'
        storage_options = None
        print(f"\nSource: {gcs_path}")
        print("Cache: Disabled")

    # Open with consolidated metadata
    print("\nOpening Zarr store...")
    try:
        root = zarr.open_consolidated(
            gcs_path,
            mode='r',
            storage_options=storage_options
        )
        print("✓ Opened with consolidated metadata")
    except (KeyError, FileNotFoundError):
        print("⚠ No consolidated metadata found, opening normally")
        root = zarr.open_group(
            gcs_path,
            mode='r',
            storage_options=storage_options
        )

    # Access precipitation array
    precip = root['precipitation']
    print(f"\nArray: precipitation")
    print(f"Shape: {precip.shape}")
    print(f"Chunks: {precip.chunks}")
    print(f"Dtype: {precip.dtype}")
    print(f"Size: {precip.nbytes / 1024**2:.1f} MB (uncompressed)")

    # Read metadata
    print("\nMetadata:")
    for key, value in precip.attrs.items():
        print(f"  {key}: {value}")

    # Read sample data (triggers chunk download)
    print("\nReading sample: first day, global mean...")
    sample = precip[0, :, :].mean()
    print(f"✓ Global mean precipitation (day 0): {sample:.2f} mm/day")

    print("\n✓ Read complete!")


# ============================================================================
# BUCKET OPERATIONS: List and Manage
# ============================================================================

def list_gcs_buckets(project=GCP_PROJECT):
    """
    List all GCS buckets in the project.
    """
    print("\n" + "="*80)
    print("LISTING GCS BUCKETS")
    print("="*80)

    if not project:
        print("⚠ GCP_PROJECT not set, skipping bucket listing")
        return

    print(f"\nProject: {project}")

    fs = create_gcs_filesystem(project)

    try:
        buckets = fs.ls('/')
        print(f"\nFound {len(buckets)} buckets:")
        for bucket in buckets[:10]:  # Show first 10
            print(f"  - {bucket}")
        if len(buckets) > 10:
            print(f"  ... and {len(buckets) - 10} more")
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")


def list_zarr_arrays(bucket, prefix=''):
    """
    List Zarr arrays in a GCS bucket.
    """
    print("\n" + "="*80)
    print("LISTING ZARR ARRAYS")
    print("="*80)

    fs = create_gcs_filesystem(GCP_PROJECT)

    print(f"\nBucket: {bucket}")
    print(f"Prefix: {prefix or '(root)'}")

    try:
        # Look for .zarray or .zgroup files
        files = fs.ls(f'{bucket}/{prefix}' if prefix else bucket, detail=False)
        zarr_arrays = [f for f in files if f.endswith('.zarray') or f.endswith('zarr.json')]

        print(f"\nFound {len(zarr_arrays)} Zarr arrays:")
        for path in zarr_arrays[:20]:
            # Remove .zarray to get array path
            array_path = path.replace('/.zarray', '').replace('/zarr.json', '')
            print(f"  - gs://{array_path}")

    except Exception as e:
        print(f"❌ Error listing arrays: {e}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run complete GCS Zarr workflow."""

    print("\n" + "="*80)
    print("GCS ZARR CONFIGURATION TEMPLATE")
    print("="*80)
    print(f"\nBucket: {GCS_BUCKET}")
    print(f"Prefix: {GCS_PREFIX}")
    print(f"Project: {GCP_PROJECT or '(not set)'}")
    print(f"Service Account: {SERVICE_ACCOUNT_JSON or '(using ADC)'}")
    print(f"Anonymous: {ANONYMOUS}")
    print(f"Caching: {ENABLE_CACHING}")
    print(f"Consolidate Metadata: {CONSOLIDATE_METADATA}")

    # Verify credentials
    if not ANONYMOUS and not SERVICE_ACCOUNT_JSON:
        print("\nℹ Using Application Default Credentials")
        print("  Run: gcloud auth application-default login")

    try:
        # Optional: List buckets
        if GCP_PROJECT:
            list_gcs_buckets(GCP_PROJECT)

        # 1. Write to GCS
        gcs_path = write_to_gcs(GCS_BUCKET, GCS_PREFIX)

        # 2. Read from GCS
        read_from_gcs(GCS_BUCKET, GCS_PREFIX, use_caching=ENABLE_CACHING)

        # 3. List arrays in bucket
        list_zarr_arrays(GCS_BUCKET, prefix='data')

        print("\n" + "="*80)
        print("✓ WORKFLOW COMPLETE")
        print("="*80)
        print(f"\nYour Zarr array is stored at: {gcs_path}")
        print("Access it with:")
        print(f"  zarr.open_consolidated('gs://{GCS_BUCKET}/{GCS_PREFIX}', mode='r')")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
