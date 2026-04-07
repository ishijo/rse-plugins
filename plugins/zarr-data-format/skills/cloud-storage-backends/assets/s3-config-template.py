#!/usr/bin/env python3
"""
Amazon S3 Zarr Configuration Template

Production-ready template for storing and accessing Zarr arrays on Amazon S3.
Demonstrates authentication, metadata consolidation, caching, and error handling.

Usage:
    # Set environment variables (or use IAM role)
    export AWS_ACCESS_KEY_ID="your-key"
    export AWS_SECRET_ACCESS_KEY="your-secret"
    export AWS_DEFAULT_REGION="us-west-2"

    python s3-config-template.py

Dependencies: zarr, fsspec, s3fs, numpy
"""

import os
import sys
import zarr
import numpy as np
from fsspec import filesystem


# ============================================================================
# CONFIGURATION
# ============================================================================

S3_BUCKET = os.getenv('S3_BUCKET', 'my-zarr-bucket')
S3_PREFIX = os.getenv('S3_PREFIX', 'data/example-array.zarr')
S3_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')

# Credentials: use environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
# OR use IAM role (recommended for EC2/Lambda)
USE_IAM_ROLE = os.getenv('USE_IAM_ROLE', 'false').lower() == 'true'
ANONYMOUS = os.getenv('S3_ANONYMOUS', 'false').lower() == 'true'

# Performance tuning
ENABLE_CACHING = True
CACHE_DIR = '/tmp/zarr-s3-cache'
CONSOLIDATE_METADATA = True


# ============================================================================
# S3 FILESYSTEM CONFIGURATION
# ============================================================================

def create_s3_filesystem(bucket, region=S3_REGION, anonymous=ANONYMOUS):
    """
    Create fsspec S3 filesystem with proper authentication.

    Parameters
    ----------
    bucket : str
        S3 bucket name
    region : str
        AWS region (e.g., 'us-west-2')
    anonymous : bool
        Use anonymous access (for public buckets)

    Returns
    -------
    fs : fsspec.AbstractFileSystem
        Configured S3 filesystem
    """
    if anonymous:
        print(f"Using anonymous S3 access (public bucket)")
        fs = filesystem('s3', anon=True)
    elif USE_IAM_ROLE:
        print(f"Using IAM role authentication")
        fs = filesystem('s3', anon=False)
    else:
        print(f"Using environment variable credentials")
        # Credentials from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        fs = filesystem(
            's3',
            key=os.getenv('AWS_ACCESS_KEY_ID'),
            secret=os.getenv('AWS_SECRET_ACCESS_KEY'),
            token=os.getenv('AWS_SESSION_TOKEN'),  # Optional
            client_kwargs={
                'region_name': region,
                'config': {
                    'signature_version': 's3v4',
                    'max_pool_connections': 50
                }
            }
        )

    return fs


# ============================================================================
# WRITE EXAMPLE: Create and Upload Zarr Array
# ============================================================================

def write_to_s3(bucket, prefix):
    """
    Create a Zarr array and write to S3 with metadata consolidation.
    """
    print("\n" + "="*80)
    print("WRITING ZARR ARRAY TO S3")
    print("="*80)

    # Create S3 store
    s3_path = f's3://{bucket}/{prefix}'
    print(f"\nTarget: {s3_path}")
    print(f"Region: {S3_REGION}")

    fs = create_s3_filesystem(bucket, S3_REGION)
    store = zarr.storage.FsspecStore(f'{bucket}/{prefix}', fs=fs)

    # Create Zarr group
    print("\nCreating Zarr group...")
    root = zarr.open_group(store, mode='w')

    # Create sample dataset
    print("Creating temperature array (365 x 180 x 360)...")
    temp = root.create_dataset(
        'temperature',
        shape=(365, 180, 360),
        chunks=(10, 90, 180),  # ~1.4 MB chunks
        dtype='f4',
        compressor=zarr.codecs.Blosc(cname='zstd', clevel=3),
        fill_value=-9999.0
    )

    # Add metadata
    temp.attrs['units'] = 'Kelvin'
    temp.attrs['long_name'] = 'Daily mean temperature'
    temp.attrs['coordinates'] = 'time lat lon'

    # Write synthetic data
    print("Writing synthetic data...")
    data = np.random.randn(365, 180, 360).astype('f4') * 10 + 273.15
    temp[:] = data

    print(f"✓ Wrote {temp.nbytes / 1024**2:.1f} MB (uncompressed)")
    print(f"✓ Stored as {temp.nbytes_stored / 1024**2:.1f} MB (compressed)")
    print(f"✓ Compression ratio: {temp.nbytes / temp.nbytes_stored:.2f}x")

    # Consolidate metadata (CRITICAL for cloud performance)
    if CONSOLIDATE_METADATA:
        print("\nConsolidating metadata...")
        zarr.consolidate_metadata(store)
        print("✓ Metadata consolidated (.zmetadata created)")

    print("\n✓ Write complete!")
    return s3_path


# ============================================================================
# READ EXAMPLE: Access Zarr Array from S3
# ============================================================================

def read_from_s3(bucket, prefix, use_caching=ENABLE_CACHING):
    """
    Read Zarr array from S3 with optional caching.
    """
    print("\n" + "="*80)
    print("READING ZARR ARRAY FROM S3")
    print("="*80)

    if use_caching:
        # Use file cache for persistent local cache
        s3_path = f'filecache::s3://{bucket}/{prefix}'
        storage_options = {
            's3': {
                'anon': ANONYMOUS,
                'client_kwargs': {'region_name': S3_REGION}
            },
            'filecache': {
                'cache_storage': CACHE_DIR,
                'expiry_time': 86400  # 24 hours
            }
        }
        print(f"\nSource: {s3_path}")
        print(f"Cache: {CACHE_DIR} (24 hour expiry)")
    else:
        s3_path = f's3://{bucket}/{prefix}'
        storage_options = None
        print(f"\nSource: {s3_path}")
        print("Cache: Disabled")

    # Open with consolidated metadata
    print("\nOpening Zarr store...")
    try:
        root = zarr.open_consolidated(
            s3_path,
            mode='r',
            storage_options=storage_options
        )
        print("✓ Opened with consolidated metadata")
    except (KeyError, FileNotFoundError):
        print("⚠ No consolidated metadata found, opening normally")
        root = zarr.open_group(
            s3_path,
            mode='r',
            storage_options=storage_options
        )

    # Access temperature array
    temp = root['temperature']
    print(f"\nArray: temperature")
    print(f"Shape: {temp.shape}")
    print(f"Chunks: {temp.chunks}")
    print(f"Dtype: {temp.dtype}")
    print(f"Size: {temp.nbytes / 1024**2:.1f} MB (uncompressed)")

    # Read metadata
    print("\nMetadata:")
    for key, value in temp.attrs.items():
        print(f"  {key}: {value}")

    # Read sample data (triggers chunk download)
    print("\nReading sample: first day, global mean...")
    sample = temp[0, :, :].mean()
    print(f"✓ Global mean temperature (day 0): {sample:.2f} K")

    print("\n✓ Read complete!")


# ============================================================================
# UPDATE EXAMPLE: Append or Modify Data
# ============================================================================

def update_on_s3(bucket, prefix):
    """
    Update existing Zarr array on S3 (modify data, add attributes).
    """
    print("\n" + "="*80)
    print("UPDATING ZARR ARRAY ON S3")
    print("="*80)

    s3_path = f's3://{bucket}/{prefix}'
    print(f"\nTarget: {s3_path}")

    fs = create_s3_filesystem(bucket, S3_REGION)
    store = zarr.storage.FsspecStore(f'{bucket}/{prefix}', fs=fs)

    # Open in append mode
    print("\nOpening in append mode...")
    root = zarr.open_group(store, mode='a')
    temp = root['temperature']

    # Modify a subset
    print("Updating first 10 days (region write)...")
    new_data = np.random.randn(10, 180, 360).astype('f4') * 10 + 280.0
    temp[0:10, :, :] = new_data
    print("✓ Updated days 0-9")

    # Update metadata
    print("Adding version attribute...")
    temp.attrs['version'] = '2.0'
    temp.attrs['last_modified'] = '2026-04-06'

    # Re-consolidate metadata (required after attribute changes)
    if CONSOLIDATE_METADATA:
        print("\nRe-consolidating metadata...")
        zarr.consolidate_metadata(store)
        print("✓ Metadata re-consolidated")

    print("\n✓ Update complete!")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run complete S3 Zarr workflow."""

    print("\n" + "="*80)
    print("S3 ZARR CONFIGURATION TEMPLATE")
    print("="*80)
    print(f"\nBucket: {S3_BUCKET}")
    print(f"Prefix: {S3_PREFIX}")
    print(f"Region: {S3_REGION}")
    print(f"IAM Role: {USE_IAM_ROLE}")
    print(f"Anonymous: {ANONYMOUS}")
    print(f"Caching: {ENABLE_CACHING}")
    print(f"Consolidate Metadata: {CONSOLIDATE_METADATA}")

    # Verify credentials
    if not ANONYMOUS and not USE_IAM_ROLE:
        if not os.getenv('AWS_ACCESS_KEY_ID'):
            print("\n❌ ERROR: AWS_ACCESS_KEY_ID not set")
            print("Set credentials or use USE_IAM_ROLE=true")
            sys.exit(1)

    try:
        # 1. Write to S3
        s3_path = write_to_s3(S3_BUCKET, S3_PREFIX)

        # 2. Read from S3
        read_from_s3(S3_BUCKET, S3_PREFIX, use_caching=ENABLE_CACHING)

        # 3. Update on S3
        update_on_s3(S3_BUCKET, S3_PREFIX)

        # 4. Verify update
        print("\n" + "="*80)
        print("VERIFYING UPDATE")
        print("="*80)
        read_from_s3(S3_BUCKET, S3_PREFIX, use_caching=False)

        print("\n" + "="*80)
        print("✓ WORKFLOW COMPLETE")
        print("="*80)
        print(f"\nYour Zarr array is stored at: {s3_path}")
        print("Access it with:")
        print(f"  zarr.open_consolidated('s3://{S3_BUCKET}/{S3_PREFIX}', mode='r')")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
