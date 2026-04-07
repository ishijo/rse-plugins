#!/usr/bin/env python3
"""
Azure Blob Storage Zarr Configuration Template

Production-ready template for storing and accessing Zarr arrays on Azure Blob Storage.
Demonstrates authentication, metadata consolidation, and error handling.

Usage:
    # Option 1: Connection String
    export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
    python azure-config-template.py

    # Option 2: Account Key
    export AZURE_STORAGE_ACCOUNT="mystorageaccount"
    export AZURE_STORAGE_KEY="your-account-key"
    python azure-config-template.py

    # Option 3: SAS Token
    export AZURE_STORAGE_ACCOUNT="mystorageaccount"
    export AZURE_STORAGE_SAS_TOKEN="?sv=2021-06-08&ss=..."
    python azure-config-template.py

Dependencies: zarr, fsspec, adlfs, numpy
"""

import os
import sys
import zarr
import numpy as np
from fsspec import filesystem


# ============================================================================
# CONFIGURATION
# ============================================================================

AZURE_CONTAINER = os.getenv('AZURE_CONTAINER', 'zarr-data')
AZURE_PREFIX = os.getenv('AZURE_PREFIX', 'arrays/example-array.zarr')
AZURE_STORAGE_ACCOUNT = os.getenv('AZURE_STORAGE_ACCOUNT', None)

# Authentication (priority order)
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING', None)
AZURE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_KEY', None)
AZURE_SAS_TOKEN = os.getenv('AZURE_STORAGE_SAS_TOKEN', None)
USE_MANAGED_IDENTITY = os.getenv('USE_MANAGED_IDENTITY', 'false').lower() == 'true'
ANONYMOUS = os.getenv('AZURE_ANONYMOUS', 'false').lower() == 'true'

# Performance tuning
ENABLE_CACHING = True
CACHE_DIR = '/tmp/zarr-azure-cache'
CONSOLIDATE_METADATA = True


# ============================================================================
# AZURE FILESYSTEM CONFIGURATION
# ============================================================================

def create_azure_filesystem(account=AZURE_STORAGE_ACCOUNT, anonymous=ANONYMOUS):
    """
    Create fsspec Azure filesystem with proper authentication.

    Parameters
    ----------
    account : str
        Azure storage account name
    anonymous : bool
        Use anonymous access (for public containers)

    Returns
    -------
    fs : fsspec.AbstractFileSystem
        Configured Azure filesystem
    """
    if anonymous:
        print(f"Using anonymous Azure access (public container)")
        fs = filesystem('az', account_name=account, anon=True)

    elif AZURE_CONNECTION_STRING:
        print(f"Using connection string authentication")
        fs = filesystem('az', connection_string=AZURE_CONNECTION_STRING)

    elif AZURE_ACCOUNT_KEY:
        print(f"Using account key authentication")
        fs = filesystem(
            'az',
            account_name=account,
            account_key=AZURE_ACCOUNT_KEY
        )

    elif AZURE_SAS_TOKEN:
        print(f"Using SAS token authentication")
        fs = filesystem(
            'az',
            account_name=account,
            sas_token=AZURE_SAS_TOKEN
        )

    elif USE_MANAGED_IDENTITY:
        print(f"Using Azure Managed Identity")
        # Requires Azure VM or App Service with managed identity
        fs = filesystem('az', account_name=account, anon=False)

    else:
        print("❌ ERROR: No authentication method configured")
        print("Set one of: AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_KEY, ")
        print("           AZURE_STORAGE_SAS_TOKEN, or USE_MANAGED_IDENTITY=true")
        sys.exit(1)

    return fs


# ============================================================================
# WRITE EXAMPLE: Create and Upload Zarr Array
# ============================================================================

def write_to_azure(container, prefix):
    """
    Create a Zarr array and write to Azure Blob Storage with metadata consolidation.
    """
    print("\n" + "="*80)
    print("WRITING ZARR ARRAY TO AZURE")
    print("="*80)

    # Create Azure store
    azure_path = f'az://{container}/{prefix}'
    print(f"\nTarget: {azure_path}")
    print(f"Account: {AZURE_STORAGE_ACCOUNT}")

    fs = create_azure_filesystem(AZURE_STORAGE_ACCOUNT)
    store = zarr.storage.FsspecStore(f'{container}/{prefix}', fs=fs)

    # Create Zarr group
    print("\nCreating Zarr group...")
    root = zarr.open_group(store, mode='w')

    # Create sample dataset
    print("Creating wind_speed array (8760 x 100 x 100)...")
    wind = root.create_dataset(
        'wind_speed',
        shape=(8760, 100, 100),  # Hourly for 1 year
        chunks=(24, 50, 50),     # Daily chunks, ~1.1 MB
        dtype='f4',
        compressor=zarr.codecs.Blosc(cname='zstd', clevel=3),
        fill_value=-9999.0
    )

    # Add metadata
    wind.attrs['units'] = 'm/s'
    wind.attrs['long_name'] = 'Hourly wind speed at 10m'
    wind.attrs['coordinates'] = 'time y x'
    wind.attrs['grid_mapping'] = 'crs'

    # Write synthetic data
    print("Writing synthetic data...")
    data = np.abs(np.random.randn(8760, 100, 100).astype('f4')) * 5.0 + 3.0
    wind[:] = data

    print(f"✓ Wrote {wind.nbytes / 1024**2:.1f} MB (uncompressed)")
    print(f"✓ Stored as {wind.nbytes_stored / 1024**2:.1f} MB (compressed)")
    print(f"✓ Compression ratio: {wind.nbytes / wind.nbytes_stored:.2f}x")

    # Consolidate metadata (CRITICAL for cloud performance)
    if CONSOLIDATE_METADATA:
        print("\nConsolidating metadata...")
        zarr.consolidate_metadata(store)
        print("✓ Metadata consolidated (.zmetadata created)")

    print("\n✓ Write complete!")
    return azure_path


# ============================================================================
# READ EXAMPLE: Access Zarr Array from Azure
# ============================================================================

def read_from_azure(container, prefix, use_caching=ENABLE_CACHING):
    """
    Read Zarr array from Azure Blob Storage with optional caching.
    """
    print("\n" + "="*80)
    print("READING ZARR ARRAY FROM AZURE")
    print("="*80)

    if use_caching:
        # Use file cache for persistent local cache
        azure_path = f'filecache::az://{container}/{prefix}'
        storage_options = {
            'az': {
                'account_name': AZURE_STORAGE_ACCOUNT,
                'account_key': AZURE_ACCOUNT_KEY,
                'sas_token': AZURE_SAS_TOKEN,
                'connection_string': AZURE_CONNECTION_STRING,
                'anon': ANONYMOUS
            },
            'filecache': {
                'cache_storage': CACHE_DIR,
                'expiry_time': 86400  # 24 hours
            }
        }
        print(f"\nSource: {azure_path}")
        print(f"Cache: {CACHE_DIR} (24 hour expiry)")
    else:
        azure_path = f'az://{container}/{prefix}'
        storage_options = None
        print(f"\nSource: {azure_path}")
        print("Cache: Disabled")

    # Open with consolidated metadata
    print("\nOpening Zarr store...")
    try:
        root = zarr.open_consolidated(
            azure_path,
            mode='r',
            storage_options=storage_options
        )
        print("✓ Opened with consolidated metadata")
    except (KeyError, FileNotFoundError):
        print("⚠ No consolidated metadata found, opening normally")
        root = zarr.open_group(
            azure_path,
            mode='r',
            storage_options=storage_options
        )

    # Access wind_speed array
    wind = root['wind_speed']
    print(f"\nArray: wind_speed")
    print(f"Shape: {wind.shape}")
    print(f"Chunks: {wind.chunks}")
    print(f"Dtype: {wind.dtype}")
    print(f"Size: {wind.nbytes / 1024**2:.1f} MB (uncompressed)")

    # Read metadata
    print("\nMetadata:")
    for key, value in wind.attrs.items():
        print(f"  {key}: {value}")

    # Read sample data (triggers chunk download)
    print("\nReading sample: first day (24 hours), spatial mean...")
    sample = wind[0:24, :, :].mean()
    print(f"✓ Mean wind speed (day 0): {sample:.2f} m/s")

    print("\n✓ Read complete!")


# ============================================================================
# CONTAINER OPERATIONS: List and Manage
# ============================================================================

def list_azure_containers(account=AZURE_STORAGE_ACCOUNT):
    """
    List all containers in the Azure storage account.
    """
    print("\n" + "="*80)
    print("LISTING AZURE CONTAINERS")
    print("="*80)

    if not account:
        print("⚠ AZURE_STORAGE_ACCOUNT not set, skipping container listing")
        return

    print(f"\nAccount: {account}")

    fs = create_azure_filesystem(account)

    try:
        containers = fs.ls('/')
        print(f"\nFound {len(containers)} containers:")
        for container in containers[:10]:  # Show first 10
            print(f"  - {container}")
        if len(containers) > 10:
            print(f"  ... and {len(containers) - 10} more")
    except Exception as e:
        print(f"❌ Error listing containers: {e}")


def list_zarr_arrays(container, prefix=''):
    """
    List Zarr arrays in an Azure container.
    """
    print("\n" + "="*80)
    print("LISTING ZARR ARRAYS")
    print("="*80)

    fs = create_azure_filesystem(AZURE_STORAGE_ACCOUNT)

    print(f"\nContainer: {container}")
    print(f"Prefix: {prefix or '(root)'}")

    try:
        # Look for .zarray or .zgroup files
        path = f'{container}/{prefix}' if prefix else container
        files = fs.ls(path, detail=False)
        zarr_arrays = [f for f in files if f.endswith('.zarray') or f.endswith('zarr.json')]

        print(f"\nFound {len(zarr_arrays)} Zarr arrays:")
        for path in zarr_arrays[:20]:
            # Remove .zarray to get array path
            array_path = path.replace('/.zarray', '').replace('/zarr.json', '')
            print(f"  - az://{array_path}")

    except Exception as e:
        print(f"❌ Error listing arrays: {e}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run complete Azure Blob Storage Zarr workflow."""

    print("\n" + "="*80)
    print("AZURE BLOB STORAGE ZARR CONFIGURATION TEMPLATE")
    print("="*80)
    print(f"\nContainer: {AZURE_CONTAINER}")
    print(f"Prefix: {AZURE_PREFIX}")
    print(f"Account: {AZURE_STORAGE_ACCOUNT or '(not set)'}")
    print(f"Auth Method: ", end='')
    if AZURE_CONNECTION_STRING:
        print("Connection String")
    elif AZURE_ACCOUNT_KEY:
        print("Account Key")
    elif AZURE_SAS_TOKEN:
        print("SAS Token")
    elif USE_MANAGED_IDENTITY:
        print("Managed Identity")
    elif ANONYMOUS:
        print("Anonymous")
    else:
        print("None (ERROR)")

    print(f"Caching: {ENABLE_CACHING}")
    print(f"Consolidate Metadata: {CONSOLIDATE_METADATA}")

    # Verify configuration
    if not AZURE_STORAGE_ACCOUNT and not AZURE_CONNECTION_STRING:
        print("\n❌ ERROR: AZURE_STORAGE_ACCOUNT or AZURE_STORAGE_CONNECTION_STRING required")
        sys.exit(1)

    try:
        # Optional: List containers
        if AZURE_STORAGE_ACCOUNT:
            list_azure_containers(AZURE_STORAGE_ACCOUNT)

        # 1. Write to Azure
        azure_path = write_to_azure(AZURE_CONTAINER, AZURE_PREFIX)

        # 2. Read from Azure
        read_from_azure(AZURE_CONTAINER, AZURE_PREFIX, use_caching=ENABLE_CACHING)

        # 3. List arrays in container
        list_zarr_arrays(AZURE_CONTAINER, prefix='arrays')

        print("\n" + "="*80)
        print("✓ WORKFLOW COMPLETE")
        print("="*80)
        print(f"\nYour Zarr array is stored at: {azure_path}")
        print("Access it with:")
        print(f"  zarr.open_consolidated('az://{AZURE_CONTAINER}/{AZURE_PREFIX}', mode='r')")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
