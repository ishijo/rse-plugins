# Zarr Data Format Plugin

Comprehensive agents and skills for working with the Zarr array storage format for cloud-native scientific data.

**Version:** 0.1.0

**Contents:**
- 2 Agents: Zarr Expert, Zarr Cloud Architect
- 5 Skills: Zarr Fundamentals, Compression Codecs, Cloud Storage Backends, Zarr-Xarray Integration, Data Migration

## Installation

This plugin is part of the RSE Plugins collection. To use it with Claude Code:

1. Clone the repository:
   ```bash
   git clone https://github.com/uw-ssec/rse-plugins.git
   ```

2. The plugin will be automatically available in the repository's marketplace at:
   ```
   plugins/zarr-data-format/
   ```

3. Load the Zarr Expert agent or individual skills through Claude Code's plugin interface.

## Plugin Structure

```
zarr-data-format/
├── .claude-plugin/
│   └── plugin.json                                # Plugin metadata and configuration
├── agents/
│   ├── zarr-expert.md                            # Comprehensive Zarr expert agent
│   └── zarr-cloud-architect.md                   # Cloud storage specialist agent
├── skills/
│   ├── zarr-fundamentals/                        # Core Zarr concepts and operations
│   │   ├── SKILL.md
│   │   ├── assets/
│   │   └── references/
│   ├── compression-codecs/                       # Compression strategies and codecs
│   │   ├── SKILL.md
│   │   ├── assets/
│   │   └── references/
│   ├── cloud-storage-backends/                   # S3, GCS, Azure backends
│   │   ├── SKILL.md
│   │   ├── assets/
│   │   └── references/
│   ├── zarr-xarray-integration/                  # Zarr with Xarray workflows
│   │   ├── SKILL.md
│   │   └── references/
│   └── data-migration/                           # HDF5, NetCDF to Zarr migration
│       ├── SKILL.md
│       ├── assets/
│       └── references/
├── LICENSE -> ../../LICENSE                       # Symlink to main repository license
└── README.md                                     # This file
```

## Available Agents

### Zarr Expert

**File:** [agents/zarr-expert.md](agents/zarr-expert.md)

**Description:** Comprehensive Zarr specialist for array creation, compression optimization, cloud storage configuration, and data migration. Expert in Zarr v2/v3 differences, chunking strategies, codec selection, and integration with Xarray and Dask. Use for any Zarr-related task requiring deep knowledge of the format and ecosystem.

**Integrated Skills:**
- zarr-fundamentals
- compression-codecs
- cloud-storage-backends
- zarr-xarray-integration
- data-migration

**When to use:**
- Creating or optimizing Zarr arrays and stores
- Migrating data from HDF5, NetCDF, or other formats to Zarr
- Choosing compression strategies and chunking patterns
- Troubleshooting Zarr performance or compatibility issues
- Any complex Zarr workflow requiring comprehensive expertise

### Zarr Cloud Architect

**File:** [agents/zarr-cloud-architect.md](agents/zarr-cloud-architect.md)

**Description:** Specialist in cloud storage backends for Zarr data (S3, GCS, Azure). Expert in fsspec, obstore, Icechunk, metadata consolidation, authentication patterns, and performance optimization for cloud-native workflows. Use for cloud deployment and optimization tasks.

**Integrated Skills:**
- cloud-storage-backends
- zarr-fundamentals

**When to use:**
- Deploying Zarr data to cloud storage (S3, GCS, Azure)
- Optimizing cloud read/write performance
- Configuring authentication and access patterns
- Setting up versioned or ACID-compliant Zarr stores with Icechunk
- Implementing metadata consolidation for cloud efficiency

## Available Skills

### Zarr Fundamentals

**File:** [skills/zarr-fundamentals/SKILL.md](skills/zarr-fundamentals/SKILL.md)

**Description:** Core Zarr concepts including array creation, group management, indexing modes, data types, thread/process safety, and Zarr v2 vs v3 differences. Essential foundation for all Zarr work.

**Key topics:**
- Zarr v2 vs v3 format differences
- Array creation functions and parameters
- All 6 indexing modes (basic, coordinate, mask, orthogonal, block, structured)
- Data types including variable-length and datetime
- Thread and process synchronization
- Sharding in Zarr v3

**When to use:**
- Creating new Zarr arrays or groups
- Understanding Zarr format versions
- Working with different data types or indexing patterns
- Setting up concurrent access to Zarr stores

### Compression Codecs

**File:** [skills/compression-codecs/SKILL.md](skills/compression-codecs/SKILL.md)

**Description:** Comprehensive guide to compression codecs (Blosc, Zstd, LZ4, Gzip, etc.), filters, codec selection strategies, and performance optimization. Includes critical safety warnings for multi-process scenarios.

**Key topics:**
- Codec selection matrix (speed vs ratio trade-offs)
- Blosc configuration and shuffle modes
- Standalone codecs (Zstd, LZ4, Gzip, LZMA)
- Filters (Delta, Quantize, etc.)
- Multi-process safety considerations
- v2 vs v3 codec pipeline differences

**When to use:**
- Choosing compression settings for new Zarr arrays
- Optimizing storage size or I/O performance
- Benchmarking different compression strategies
- Troubleshooting compression-related issues

### Cloud Storage Backends

**File:** [skills/cloud-storage-backends/SKILL.md](skills/cloud-storage-backends/SKILL.md)

**Description:** Complete guide to cloud storage backends including fsspec, obstore, Icechunk, authentication patterns, metadata consolidation, and caching strategies for S3, GCS, and Azure.

**Key topics:**
- Backend comparison (fsspec, obstore, Icechunk)
- S3, GCS, Azure configuration and authentication
- Metadata consolidation for cloud performance
- Caching strategies (simplecache, filecache, blockcache)
- Concurrency tuning for different providers
- Versioned stores with Icechunk

**When to use:**
- Storing Zarr data on cloud object storage
- Configuring cloud authentication
- Optimizing cloud read/write performance
- Implementing versioning or ACID compliance

### Zarr-Xarray Integration

**File:** [skills/zarr-xarray-integration/SKILL.md](skills/zarr-xarray-integration/SKILL.md)

**Description:** Using Zarr with Xarray for labeled multidimensional arrays, including all write modes, encoding control, distributed writes, and Dask integration. Cross-references the xarray-for-multidimensional-data skill for Xarray basics.

**Key topics:**
- Reading Zarr with Xarray (chunking strategies)
- All 6 write modes (mode='w'/'a', append_dim, region writes)
- Encoding precedence and control
- Distributed parallel writes
- Consolidated metadata for cloud access
- Dask lazy loading patterns

**When to use:**
- Reading or writing Zarr data with Xarray
- Implementing append or update workflows
- Performing distributed writes to Zarr stores
- Working with large Zarr datasets via Dask

### Data Migration

**File:** [skills/data-migration/SKILL.md](skills/data-migration/SKILL.md)

**Description:** Migrating data from HDF5, NetCDF, or other formats to Zarr. Covers zarr.copy functions, VirtualiZarr for zero-copy references, Icechunk integration, and validation strategies.

**Key topics:**
- HDF5 to Zarr migration strategies
- NetCDF to Zarr with Xarray
- VirtualiZarr for zero-copy metadata references
- Zarr v2 to v3 conversion
- Rechunking and recompression
- Validation and testing approaches

**When to use:**
- Converting legacy HDF5 or NetCDF data to Zarr
- Creating virtual Zarr references to existing data
- Rechunking or recompressing existing Zarr stores
- Validating migration results

## Usage

### Using the Zarr Expert Agent

The Zarr Expert agent provides comprehensive guidance for all Zarr tasks. Load it through Claude Code's interface for:
- End-to-end Zarr workflows (creation, compression, migration)
- Architectural decisions for Zarr-based systems
- Performance optimization and troubleshooting
- Learning Zarr best practices

### Using the Zarr Cloud Architect Agent

The Zarr Cloud Architect specializes in cloud deployments. Use it for:
- Deploying Zarr to S3, GCS, or Azure
- Optimizing cloud performance
- Implementing authentication and security
- Setting up versioned or ACID stores

### Using Individual Skills

Skills can be loaded independently for focused expertise:

- **Load zarr-fundamentals** for core array operations and format knowledge
- **Load compression-codecs** for compression selection and optimization
- **Load cloud-storage-backends** for cloud deployment
- **Load zarr-xarray-integration** for Xarray workflows
- **Load data-migration** for format conversion tasks

## When to Use This Plugin

Use the Zarr Data Format plugin when working on:

- **Large-scale scientific data** requiring chunked, cloud-optimized storage
- **Data migration** from HDF5, NetCDF, or other formats to Zarr
- **Cloud-native workflows** with data stored on S3, GCS, or Azure
- **Performance optimization** for I/O-intensive scientific computing
- **Multi-dimensional arrays** requiring efficient partial reads
- **Distributed computing** with Dask and parallel I/O
- **Data archiving** with compression and long-term storage requirements
- **Collaborative research** with shared cloud-based datasets

## Resources

### Official Documentation

- [Zarr Specification](https://zarr.dev/)
- [zarr-python Documentation](https://zarr.readthedocs.io/)
- [Xarray Zarr Tutorial](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr)
- [fsspec Documentation](https://filesystem-spec.readthedocs.io/)
- [Icechunk Documentation](https://icechunk.io/)

### Community

- [Zarr GitHub Discussions](https://github.com/zarr-developers/zarr-python/discussions)
- [Pangeo Community](https://pangeo.io/) - Cloud-native geoscience
- [Xarray GitHub Discussions](https://github.com/pydata/xarray/discussions)

### Research

- Nguyen et al. (2023) - Chunking optimization for time-series and spatial access patterns
- [Pangeo Best Practices](https://pangeo.io/data.html)
- [CarbonPlan Guides](https://carbonplan.org/)

Licensed under BSD-3-Clause. See [LICENSE](../../LICENSE).
