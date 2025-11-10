# EffiDict Benchmarking Plan

## Overview

Create a comprehensive benchmarking suite that evaluates EffiDict's performance across all replacement strategies (LRU, MRU, FIFO, LIFO, LFU, MFU, Random) and backends (SQLite, JSON, Pickle, HDF5). The benchmarks compare EffiDict against two baselines: standard Python dict (in-memory) and Python shelve (persistent storage). The primary focus is on generating specific plots: runtime for read/write/delete/modification operations and memory usage, all plotted against increasing dataset sizes. Includes a specific use case from the oligo-designer-toolsuite with nested dictionary structures and sequential circular access patterns.

## Structure

### 1. Benchmark Framework (`benchmarks/benchmark_framework.py`)

- Create a reusable benchmarking framework with:
- Timer utilities for measuring operation latency
- Memory profiler integration (track RAM/swap usage)
- Results collection and aggregation
- CSV/JSON export functionality
- Comparison utilities against:
- Standard Python dict (baseline for in-memory)
- Python shelve (baseline for persistent storage)
- EffiDict (all strategy/backend combinations)

### 2. Dataset Generators (`benchmarks/datasets.py`)

Generate test data with different value sizes:

- **Small uniform objects**: Small strings, integers, small dicts (baseline comparison)
- Example: `{"key1": "value", "key2": 123}` (~100 bytes)
- Sizes: [100 bytes, 1KB, 10KB]
- **Large objects**: Large arrays, nested dicts, binary data (EffiDict's strength)
- Example: NumPy arrays (1MB-100MB), large JSON objects, binary blobs
- Sizes: [100KB, 1MB, 10MB, 100MB, 500MB, 1GB]
- Configurable to demonstrate RAM savings
- **Oligo Designer Toolsuite dataset**: Nested dictionary structure
- Outer structure: region_id -> inner dictionary
- Inner structure: DNA_sequence -> metadata dictionary
- Example: `{"region_1": {"ACGTACGT": {"chromosome": 1, "start": 100, "end": 108, "strand": "+", ...}, ...}, ...}`
- Configurable number of regions and sequences per region
- Simulates real oligo design pipeline data structure
- Reference: https://github.com/HelmholtzAI-Consultants-Munich/oligo-designer-toolsuite

### 3. Workload Generators (`benchmarks/workloads.py`)

Implement access pattern generators:

- **Uniform distribution**: All keys equally likely to be accessed
- Random selection from all keys with equal probability
- Baseline access pattern
- **Zipf distribution**: Small subset of "hot" keys get most accesses
- Power-law distribution where few keys are accessed frequently
- Configurable skewness parameter (e.g., s=1.0, s=1.5, s=2.0)
- Majority of keys rarely touched (demonstrates eviction effectiveness)
- Critical for testing replacement strategies (hot keys should stay in memory)
- **Sequential circular access**: Keys accessed in order, wrapping around
- Access keys sequentially: key_1, key_2, ..., key_N, then back to key_1
- Simulates oligo designer toolsuite access pattern
- Useful for testing LRU and FIFO strategies

### 4. Benchmark Suites

#### 4.1 Runtime Benchmarks (`benchmarks/test_runtime.py`)

Measure runtime for read, write, delete, and modification operations across increasing dataset sizes:

**Write Operations:**

- Measure total time to insert N entries (i.e. N key/value pairs)
- Dataset sizes: [1K, 10K, 100K, 1M, 10M entries] (configurable)
- Compare: EffiDict (multiple configs) vs dict vs shelve
- Track time for each dataset size
- Output: Data for write operations plot (entries vs time[s])

**Read Operations:**

- Measure total time to read/access N entries (i.e. N key/value pairs)
- Dataset sizes: [1K, 10K, 100K, 1M, 10M entries] (configurable)
- Compare: EffiDict (multiple configs) vs dict vs shelve
- Track time for each dataset size
- Output: Data for read operations plot (entries vs time[s])

**Delete Operations:**

- Measure total time to delete N entries (i.e. N key/value pairs)
- Dataset sizes: [1K, 10K, 100K, 1M, 10M entries] (configurable)
- Compare: EffiDict (multiple configs) vs dict vs shelve
- Track time for each dataset size
- Output: Data for delete operations plot (entries vs time[s])

**Modification Operations:**

- Measure total time to modify/update N entries (i.e. N key/value pairs)
- Dataset sizes: [1K, 10K, 100K, 1M, 10M entries] (configurable)
- Compare: EffiDict (multiple configs) vs dict vs shelve
- Track time for each dataset size
- Output: Data for modification operations plot (entries vs time[s])

#### 4.2 Memory Benchmarks (`benchmarks/test_memory.py`)

Measure memory usage (RAM/swap) across increasing dataset sizes:

**Memory Usage:**

- Track RAM usage for datasets of increasing size
- Track swap usage (if applicable)
- Dataset sizes: [1K, 10K, 100K, 1M, 10M entries] (configurable)
- Measure peak memory usage for each dataset size
- Compare: EffiDict (multiple configs) vs dict vs shelve
- Output: Data for memory plot (entries vs RAM/swap)

#### 4.3 Replacement Strategy Comparison (`benchmarks/test_replacement_strategies.py`)

Compare all 7 strategies under:

- **Dataset types**: Small uniform objects vs large objects
- **Access patterns**: Uniform distribution vs Zipf distribution vs sequential circular
- Different memory limits (10, 100, 1000, 10000, 100000 items)
- Different value sizes (1KB, 10KB, 100KB, 1MB, 10MB, 100MB, 500MB, 1GB)
- Cache hit rates for each workload pattern (especially Zipf and sequential circular)
- Eviction overhead and effectiveness
- RAM usage reduction compared to standard dict and shelve
- Hot key retention under Zipf distribution (which strategies keep hot keys in memory)
- Performance comparison: Best EffiDict strategy vs shelve for each access pattern

#### 4.4 Backend Comparison (`benchmarks/test_backends.py`)

Compare all 4 backends:

- Serialization/deserialization speed (especially for large values)
- Disk I/O performance
- Storage efficiency (file size comparison)
- Performance with large objects (100KB-100MB)
- Comparison with shelve (which uses pickle by default)
- Backend-specific optimizations (e.g., HDF5 for arrays)

#### 4.5 Use Case Scenarios (`benchmarks/test_use_cases.py`)

Real-world scenarios emphasizing large values:

- **Machine learning feature cache**: Large NumPy arrays (1MB-100MB per key)
- Zipf access pattern: frequently used features stay in memory
- HDF5 backend focus
- Comparison: EffiDict vs shelve vs dict
- **Image/Media cache**: Large binary objects (images, videos)
- Hot content (popular items) vs cold content
- Zipf distribution of access
- Comparison: EffiDict vs shelve vs dict
- **Oligo Designer Toolsuite use case**: Nested dictionary structure for oligo design
- Structure: region_id (key) -> dictionary (value) containing DNA sequences (keys) and metadata (values)
- Many key/value pairs: region IDs as keys, dictionaries as values
- Each inner dictionary contains many short DNA sequences as keys with their metadata as values
- Access pattern: Sequential circular access (regions accessed in order, wrapping around)
- Comparison: EffiDict vs shelve vs dict
- Demonstrates EffiDict's ability to handle nested structures with sequential access patterns
- Reference: https://github.com/HelmholtzAI-Consultants-Munich/oligo-designer-toolsuite

### 5. Visualization (`benchmarks/visualize_results.py`)

Create visualization utilities with specific plots:

**Runtime Plots:**

- **Read operations plot**:
- x-axis: different number of entries (dataset sizes: 1K, 10K, 100K, 1M, 10M)
- y-axis: time [s]
- Lines for: EffiDict (best config), dict, shelve
- Multiple EffiDict configurations can be shown (different strategies/backends)
- Clear legend distinguishing implementations
- **Write operations plot**:
- x-axis: different number of entries (dataset sizes: 1K, 10K, 100K, 1M, 10M)
- y-axis: time [s]
- Lines for: EffiDict (best config), dict, shelve
- Multiple EffiDict configurations can be shown
- Clear legend distinguishing implementations
- **Delete operations plot**:
- x-axis: different number of entries (dataset sizes: 1K, 10K, 100K, 1M, 10M)
- y-axis: time [s]
- Lines for: EffiDict (best config), dict, shelve
- Multiple EffiDict configurations can be shown
- Clear legend distinguishing implementations
- **Modification operations plot**:
- x-axis: different number of entries (dataset sizes: 1K, 10K, 100K, 1M, 10M)
- y-axis: time [s]
- Lines for: EffiDict (best config), dict, shelve
- Multiple EffiDict configurations can be shown
- Clear legend distinguishing implementations

**Memory Plots:**

- **Memory usage plot**:
- x-axis: different number of entries (dataset sizes: 1K, 10K, 100K, 1M, 10M)
- y-axis: RAM/swap usage (MB or GB)
- Lines for: EffiDict (best config), dict, shelve
- Separate tracking for RAM vs swap if applicable
- Multiple EffiDict configurations can be shown
- Clear legend distinguishing implementations

**Additional visualizations (optional):**

- Cache hit rate heatmaps (strategy × workload, especially Zipf and sequential circular)
- Performance vs memory limit curves
- Backend comparison charts
- Side-by-side comparison tables: EffiDict (best config) vs dict vs shelve

### 6. Main Benchmark Runner (`benchmarks/run_benchmarks.py`)

Orchestrate all benchmarks:

- Command-line interface for selective execution
- Results aggregation
- Report generation (HTML/Markdown)
- Comparison with baselines (standard dict and shelve)
- Summary report highlighting EffiDict advantages over shelve
- Generate the primary plots (read runtime, write runtime, delete runtime, modification runtime, memory)

### 7. Configuration (`benchmarks/config.py`)

Centralized configuration:

- Default parameters (memory limits, dataset sizes)
- Backend/strategy combinations to test
- Workload parameters (Zipf skewness values)
- Value size configurations
- Output directories
- Baseline configurations (dict, shelve)
- **Dataset sizes for plots**: [1K, 10K, 100K, 1M, 10M entries]
- **Oligo Designer Toolsuite parameters**: Number of regions, sequences per region

## Implementation Details

### Key Metrics to Track

- **Runtime for write operations**: Total time to insert N entries
- **Runtime for read operations**: Total time to access N entries
- **Runtime for delete operations**: Total time to delete N entries
- **Runtime for modification operations**: Total time to modify/update N entries
- **Memory usage**: Peak RAM/swap usage for N entries
- **Latency**: Mean, median, p95, p99 for each operation type
- Separate metrics for memory hits vs disk misses
- **Throughput**: Operations per second
- **Cache efficiency**: 
- Hit rate, miss rate, eviction frequency
- Effectiveness under Zipf distribution (hot keys stay in memory)
- Effectiveness under sequential circular access
- **Comparative analysis**:
- EffiDict vs shelve: Performance, memory usage, features
- EffiDict vs dict: Memory savings, persistence capabilities

### Test Parameters

- **Dataset sizes for main plots**: [1K, 10K, 100K, 1M, 10M entries]
- These are the x-axis values for runtime and memory plots
- Memory limits (for EffiDict): [10, 100, 1000, 10000 items]
- **Value sizes**: 
- Small uniform: [100 bytes, 1KB, 10KB]
- Large objects: [100KB, 1MB, 10MB, 100MB, 500MB, 1GB]
- **Access patterns**:
- Uniform distribution: Equal probability for all keys
- Zipf distribution: s=[1.0, 1.5, 2.0] (higher = more skewed)
- Sequential circular: Keys accessed in order, wrapping around
- Workload sizes: [1K, 10K, 100K, 1M operations]
- **Oligo Designer Toolsuite parameters**:
- Number of regions: [10, 100, 1000, 10000]
- Sequences per region: [10, 100, 1000]

### Output Format

- CSV files for raw data (runtime and memory measurements per dataset size)
- JSON files for structured results
- **Primary plots** (as specified):
- Runtime plot for read operations (entries vs time[s])
- Runtime plot for write operations (entries vs time[s])
- Runtime plot for delete operations (entries vs time[s])
- Runtime plot for modification operations (entries vs time[s])
- Memory plot (entries vs RAM/swap)
- HTML reports with visualizations
- Summary statistics tables
- Comparison matrices (EffiDict vs dict vs shelve)

## Files to Create/Modify

1. `benchmarks/benchmark_framework.py` - Core benchmarking utilities (supports dict, shelve, EffiDict)
2. `benchmarks/datasets.py` - Dataset generators (small uniform vs large objects, oligo designer toolsuite dataset)
3. `benchmarks/workloads.py` - Workload pattern generators (uniform vs Zipf vs sequential circular)
4. `benchmarks/test_runtime.py` - Runtime benchmarks (read/write/delete/modification operations)
5. `benchmarks/test_memory.py` - Memory usage benchmarks
6. `benchmarks/test_replacement_strategies.py` - Strategy comparison (focus on Zipf and sequential circular, includes shelve baseline)
7. `benchmarks/test_backends.py` - Backend comparison (includes shelve comparison)
8. `benchmarks/test_use_cases.py` - Real-world scenario tests (large values, oligo designer toolsuite, dict vs shelve vs EffiDict)
9. `benchmarks/visualize_results.py` - Visualization utilities (three-way comparisons with specific plots)
10. `benchmarks/run_benchmarks.py` - Main benchmark runner
11. `benchmarks/config.py` - Configuration management (includes shelve config, oligo designer toolsuite params)
12. `benchmarks/requirements.txt` - Additional dependencies (memory_profiler, psutil, etc.)
13. Update `benchmarks/utils.py` if needed for compatibility

## Dependencies

- `memory_profiler` or `psutil` for memory tracking
- `matplotlib` and `seaborn` for visualization (already in utils.py)
- `pandas` for data manipulation
- `numpy` for statistical analysis and Zipf distribution
- `tqdm` for progress bars
- Python standard library `shelve` (no additional install needed)