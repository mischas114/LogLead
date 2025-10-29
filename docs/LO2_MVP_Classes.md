# LO2 MVP Classes Documentation

## Overview

This document describes the Python classes that have been added to support the LO2 (OAuth2 Log Analysis) Minimum Viable Product (MVP) in LogLead. The LO2 MVP enables log anomaly detection for OAuth2 service logs from microservice-based systems.

## Classes Added

### 1. `LO2Loader` Class

**Location:** `loglead/loaders/lo2.py`

**Purpose:** The `LO2Loader` is a custom loader class designed to process and analyze log files from LO2 pipeline runs. It enables LogLead to work with OAuth2 service logs from distributed microservice architectures.

#### Why It Is Necessary

The `LO2Loader` class is necessary because:

1. **Specialized Data Structure**: LO2 pipeline data has a unique directory structure organizing logs by runs, test cases (correct vs. error scenarios), and individual services. This requires custom parsing logic.

2. **OAuth2 Microservice Focus**: Unlike traditional monolithic log systems, LO2 deals with OAuth2 authentication service logs from multiple microservices (client, code, key, refresh-token, service, token, user). Each service has distinct log characteristics.

3. **Experimental Flexibility**: The loader supports various experimental configurations:
   - Selecting specific numbers of runs and errors per run
   - Controlling duplicate error types across runs
   - Focusing on single error types for targeted analysis
   - Filtering by specific services

4. **Integration with LogLead Pipeline**: It follows the LogLead architecture pattern, producing standardized dataframes that work seamlessly with downstream enhancers and anomaly detectors.

#### Key Functionality

The `LO2Loader` class provides the following functionality:

##### 1. **Initialization Parameters**
```python
LO2Loader(
    filename,           # Path to the data directory
    n_runs=53,         # Number of runs to process
    errors_per_run=1,  # Number of errors per run
    dup_errors=True,   # Allow duplicate errors across runs
    single_error_type=None,  # Focus on specific error type
    single_service=""  # Filter by specific service
)
```

##### 2. **Log File Loading** (`load()` method)
- Traverses the LO2 directory structure
- Processes log files from multiple runs and test cases
- Filters logs by service type if specified
- Supports both "correct" (normal) and error test cases
- Creates event-level dataframe with columns:
  - `m_message`: The actual log message
  - `run`: The run identifier
  - `test_case`: Test case name (e.g., "correct" or specific error type)
  - `service`: Service name
  - `seq_id`: Sequence identifier (combination of run, test_case, and service)
  - `normal`: Boolean indicating if the log is from a correct run
  - `m_timestamp`: Parsed timestamp from log message

##### 3. **Metrics Loading** (`load_metrics()` method)
- Optional functionality to load system metrics (CPU load, memory, etc.)
- Processes JSON metric files from the metrics directory
- Creates a separate metrics dataframe with temporal data
- Currently implemented but noted as having potential issues

##### 4. **Log File Processing** (`_process_log_file()` method)
- Reads individual log files
- Cleans and strips log lines
- Adds metadata (run, test_case, service)
- Creates sequence IDs for grouping related events

##### 5. **Timestamp Parsing** (`_parse_timestamps()` method)
- Extracts timestamps from log messages using regex
- Supports multiple timestamp formats:
  - `HH:MM:SS.mmm` (time only)
  - `YYYY-MM-DD HH:MM:SS` (full datetime)
  - `YYYY-MM-DDTHH:MM:SS` (ISO format)
- Converts timestamps to Polars DateTime objects
- Filters out rows with missing timestamps

##### 6. **Preprocessing** (`preprocess()` method)
- Creates sequence-level dataframe (`df_seq`) by grouping events
- Aggregates sequence metadata:
  - Concatenates all messages in a sequence
  - Tracks start and end times
  - Maintains normal/anomaly labels
- Adds anomaly column (inverse of normal column)

##### 7. **Sequence Creation** (`_create_df_seq()` method)
- Groups events by `seq_id`
- Aggregates key information:
  - First test_case and service values
  - Concatenated messages (joined with newlines)
  - Logical OR of normal flags
  - Min/max timestamps for duration calculation

##### 8. **Error Handling Strategies**
The loader supports different strategies for handling errors across runs:

- **Duplicate Errors Allowed** (`dup_errors=True`): Same error types can appear in multiple runs, useful for building larger datasets with specific error patterns.

- **No Duplicate Errors** (`dup_errors=False`): Each error type is used only once across all runs, ensuring broader coverage of different error scenarios.

- **Single Error Type Focus** (`single_error_type="error_name"`): Only processes one specific error type across all runs, useful for focused analysis of particular failure modes.

- **Random Error Selection** (`single_error_type="random"`): Randomly selects one error type from the first run and uses it consistently across all runs.

#### How It Was Created

The `LO2Loader` was created by studying and following the patterns established by existing loaders in LogLead:

##### 1. **Inheritance from BaseLoader**
```python
class LO2Loader(BaseLoader):
    def __init__(self, filename, df=None, df_seq=None, ...):
        super().__init__(filename, df, df_seq)
```

Like `HDFSLoader`, `BGLLoader`, and others, `LO2Loader` inherits from `BaseLoader` to leverage:
- Standard execute/preprocess lifecycle
- Mandatory column checking
- Null value detection
- Anomaly column creation

##### 2. **Pattern: Structured Loading from Existing Loaders**

**Inspired by HDFSLoader** (`loglead/loaders/hdfs.py`):
- Creating sequence IDs for grouping related events
- Separating event-level and sequence-level dataframes
- Parsing timestamps into Polars DateTime format
- Joining labels/metadata with the main dataframe

**Inspired by BGLLoader** (`loglead/loaders/bgl.py`):
- Using the `normal` column as the primary label
- Converting boolean labels to anomaly flags
- Handling timestamp parsing with multiple format support

**Inspired by NezhaLoader** (`loglead/loaders/nezha.py`):
- Processing multiple services from a microservice architecture
- Handling optional metrics loading
- Processing complex directory structures with runs and test cases

##### 3. **Pattern: Dataframe Construction**
Following the established pattern from all loaders:
```python
log_df = pl.DataFrame({
    "m_message": log_lines_cleaned
})
log_df = log_df.with_columns(
    pl.lit(run).alias("run"),
    pl.lit(test_case).alias("test_case"),
    # ... additional columns
)
```

##### 4. **Pattern: Preprocessing Pipeline**
The standard LogLead preprocessing pattern:
1. `load()` - Read raw data into event-level dataframe
2. `preprocess()` - Create sequence-level aggregations and add derived columns
3. `execute()` - (inherited from BaseLoader) Orchestrates load, preprocess, and validation

##### 5. **Implementation Specifics**

The LO2-specific implementation includes:

- **Directory Traversal Logic**: Custom logic to navigate the LO2 directory structure:
  ```
  root/
    run_1/
      correct/
        oauth2-oauth2-client.log
        oauth2-oauth2-service.log
        ...
      error_type_1/
        oauth2-oauth2-client.log
        ...
      error_type_2/
        ...
  ```

- **Service Filtering**: Uses filename pattern matching to filter by OAuth2 service type
- **Flexible Error Sampling**: Implements multiple strategies for selecting which error cases to include
- **Sequence ID Generation**: Combines run, test_case, and service into a unique identifier

## Contributor's Value-Add: Mischa Tettenborn's Implementation

### What Makes the LO2 MVP Viable

The LO2 MVP was implemented by **Mischa Tettenborn** (@mischas114), who created the complete integration layer that makes LO2 log analysis practical and usable within LogLead. His contributions transformed the theoretical capability into a working, research-ready system.

#### Core Implementation (by Mischa Tettenborn)

**1. LO2Loader Class** (`loglead/loaders/lo2.py`, 254 lines)
- Custom loader tailored to LO2's unique directory structure and OAuth2 microservice architecture
- Flexible configuration system (error sampling, service filtering, run selection)
- Seamless integration with LogLead's existing enhancer and detector ecosystem
- **Without this**: No way to load LO2 data into LogLead; the entire pipeline would be blocked at the data ingestion stage

**2. Complete Demo Script Suite** (5 scripts, ~1,200 lines total)
- `demo/run_lo2_loader.py`: CLI-based loader with extensive configuration options
- `demo/LO2_samples.py`: End-to-end pipeline demonstration (load → enhance → detect → explain)
- `demo/lo2_if_baseline.py`: Isolation Forest baseline implementation with sequence-level analysis
- `demo/lo2_phase_f_explainability.py`: Advanced explainability workflow with SHAP and nearest-neighbor analysis
- `demo/lo2_feature_test.py`: Feature matrix validation and sanity checks
- **Without these**: Users would have no clear starting point; research workflows would need to be built from scratch; critical features like result persistence, metrics tracking, and XAI integration would be missing

**3. Comprehensive Documentation Layer** (5 docs, ~600 lines)
- `docs/LO2_architektur_detail.md`: Architecture specifications and component interactions
- `docs/LO2_e2e_pipeline.md`: Complete end-to-end workflow documentation
- `docs/LO2_minimal_IF_XAI_workflow.md`: Isolation Forest and explainability best practices
- `docs/LO2_prototype_pipeline.md`: Prototype specifications with component responsibilities
- `docs/LO2_enhanced_exports.md`: Data persistence and artifact management
- **Without these**: No understanding of design decisions; no guidance on workflow setup; no reproducibility framework

**4. Tooling for Result Analysis** (`tools/lo2_result_scan.py`, 265 lines)
- Automated result artifact discovery and summarization
- Integration with `summary-result.md` for tracking experiments
- Support for dry-run and custom output paths
- **Without this**: Manual result aggregation would be error-prone; experiment tracking would require custom scripting; no systematic way to compare runs

#### Functional Impact of These Contributions

**What Changes Without Mischa's Work:**

1. **Data Loading**: Impossible to ingest LO2 logs without the custom loader
   - LO2's directory structure (runs → test cases → services) is incompatible with RawLoader
   - Error sampling strategies wouldn't exist
   - Service-level filtering would need manual pre-processing

2. **Workflow Clarity**: No clear path from data to results
   - Demo scripts provide the "glue code" that connects loader → enhancers → detectors → explainers
   - Without them: researchers would spend days figuring out correct API calls, parameter settings, and data transformations

3. **Experimental Flexibility**: Limited ability to configure analyses
   - CLI parameters in `run_lo2_loader.py` enable quick iteration (service filtering, error sampling, run limits)
   - Without them: code modifications required for each experiment variation

4. **Explainability Integration**: No systematic approach to understanding model decisions
   - `lo2_phase_f_explainability.py` implements the complete XAI workflow (IF → NN mapping → SHAP → false positive analysis)
   - Without it: explainability would be ad-hoc; no standard for comparing error patterns

5. **Result Management**: No systematic experiment tracking
   - `lo2_result_scan.py` automates artifact collection and summary generation
   - Without it: manual file collection; no version control for experiment outcomes

#### Why These Contributions Are Critical for the MVP

An MVP requires more than just a loader class—it needs a **complete workflow** that demonstrates value:

- ✅ **Data ingestion**: LO2Loader handles the complex directory structure
- ✅ **Feature engineering**: Demo scripts show how to apply enhancers (words, trigrams, Drain, lengths)
- ✅ **Model training**: Examples for both supervised and unsupervised approaches
- ✅ **Evaluation**: Metrics computation, ROC curves, precision@k
- ✅ **Explainability**: SHAP values, nearest neighbors, false positive analysis
- ✅ **Reproducibility**: Configuration persistence, result tracking, artifact management

**Without Mischa's demo scripts and tooling**, users would have:
- A loader class with no usage examples
- No understanding of how to configure experiments
- No explainability workflow
- No systematic way to track results
- No best practices for LO2-specific challenges (e.g., handling client service hotspots, calibrating contamination thresholds)

His work transforms a "technically possible" integration into a **research-ready platform** where users can immediately begin analyzing OAuth2 logs, iterating on models, and understanding anomaly patterns.

## Usage Examples

### Basic Usage
```python
from loglead.loaders import LO2Loader

# Load LO2 data with default settings
loader = LO2Loader(filename="/path/to/lo2/data")
df_events = loader.execute()

# Access event-level data
print(f"Events: {len(loader.df)}")
print(loader.df.head())

# Access sequence-level data
print(f"Sequences: {len(loader.df_seq)}")
print(loader.df_seq.head())
```

### Advanced Usage with Service Filtering
```python
# Focus on client service only
loader = LO2Loader(
    filename="/path/to/lo2/data",
    n_runs=100,
    single_service="client"
)
loader.execute()
```

### Error-Focused Analysis
```python
# Analyze a specific error type across all runs
loader = LO2Loader(
    filename="/path/to/lo2/data",
    single_error_type="timeout_error",
    n_runs=50
)
loader.execute()
```

### Using in Complete Pipeline
```python
from loglead.loaders import LO2Loader
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
from loglead import AnomalyDetector

# Load data
loader = LO2Loader(filename="/path/to/lo2/data", n_runs=100)
df_events = loader.execute()

# Enhance events
enhancer = EventLogEnhancer(df_events)
df_events = enhancer.normalize()
df_events = enhancer.words()
df_events = enhancer.parse_drain()

# Enhance sequences
seq_enhancer = SequenceEnhancer(df=df_events, df_seq=loader.df_seq)
df_seqs = seq_enhancer.seq_len()
df_seqs = seq_enhancer.tokens(token="e_words")

# Anomaly detection
detector = AnomalyDetector()
detector.item_list_col = "e_words"
detector.test_train_split(df_events, test_frac=0.9)
detector.prepare_train_test_data()
detector.train_IsolationForest()
predictions = detector.predict()
```

## Integration with LogLead Architecture

The `LO2Loader` fits into the LogLead architecture as follows:

```
┌─────────────────┐
│   LO2Loader     │
│   (lo2.py)      │
└────────┬────────┘
         │ produces df & df_seq
         ▼
┌─────────────────┐
│   Enhancers     │
│  - EventLog     │
│  - Sequence     │
└────────┬────────┘
         │ adds features
         ▼
┌─────────────────┐
│ AnomalyDetector │
│  - Supervised   │
│  - Unsupervised │
└─────────────────┘
```

### Output Schema

The `LO2Loader` produces dataframes with the following schemas:

**Event-level dataframe (df):**
- `m_message` (String): Raw log message
- `m_timestamp` (Datetime): Parsed timestamp
- `run` (String): Run identifier
- `test_case` (String): Test case name
- `service` (String): Service name
- `seq_id` (String): Sequence identifier
- `normal` (Boolean): True if from correct run
- `anomaly` (Boolean): True if anomalous (inverse of normal)

**Sequence-level dataframe (df_seq):**
- `seq_id` (String): Sequence identifier
- `m_message` (String): Concatenated log messages
- `normal` (Boolean): True if sequence is normal
- `anomaly` (Boolean): True if sequence is anomalous
- `start_time` (Datetime): First timestamp in sequence
- `end_time` (Datetime): Last timestamp in sequence

**Metrics dataframe (metrics_df, optional):**
- `run` (String): Run identifier
- `test_case` (String): Test case name
- `metric_name` (String): Name of the metric
- `timestamp` (Datetime): Metric timestamp
- `value` (Float): Metric value

## Demo Scripts

Several demo scripts demonstrate the `LO2Loader` in action:

### 1. `demo/run_lo2_loader.py`
Basic loader demonstration with CLI arguments for configuration. Shows how to:
- Load LO2 data with various parameters
- Save to Parquet format
- Inspect event and sequence dataframes

### 2. `demo/LO2_samples.py`
Complete pipeline demonstration showing:
- Loading LO2 data
- Applying enhancers (normalization, tokenization, parsing)
- Training anomaly detectors
- Generating explainability plots

### 3. `demo/lo2_phase_f_explainability.py`
Advanced explainability analysis for LO2 anomaly detection

## Design Decisions

### 1. **Flexible Error Sampling**
The loader provides multiple error sampling strategies to support different research and analysis needs:
- High error repetition for focused studies
- Diverse error coverage for comprehensive testing
- Single error type for baseline comparisons

### 2. **Service-Level Filtering**
OAuth2 systems have multiple services that may need independent analysis. The loader allows focusing on specific services while maintaining the ability to analyze the complete system.

### 3. **Timestamp Handling**
LO2 logs often lack date information, only providing time. The loader:
- Extracts whatever timestamp information is available
- Defaults to 1900-01-01 when no date is present
- Supports multiple timestamp formats for flexibility

### 4. **Metrics as Optional**
Metrics loading is optional and separate from log loading because:
- Not all analysis requires metrics
- Metrics files may not always be present
- Keeps the primary use case simple and fast

## Future Enhancements

Potential improvements noted in the code:

1. **Metrics Integration**: The metrics loader is implemented but noted as potentially having issues. Further testing and refinement may be needed.

2. **Additional Timestamp Formats**: More timestamp parsing patterns could be added as needed.

3. **Performance Optimization**: For very large datasets, batch processing or lazy loading could improve memory efficiency.

4. **Service Discovery**: Automatic detection of available services instead of requiring manual specification.

## References

- **Base Loader Pattern**: `loglead/loaders/base.py`
- **Similar Implementations**: 
  - `loglead/loaders/hdfs.py` (sequence-based logging)
  - `loglead/loaders/nezha.py` (microservice architecture)
  - `loglead/loaders/bgl.py` (timestamp parsing)
- **Demo Scripts**: `demo/run_lo2_loader.py`, `demo/LO2_samples.py`, `demo/lo2_phase_f_explainability.py`
- **Architecture Documentation**: `docs/LO2_architektur_detail.md`, `docs/LO2_e2e_pipeline.md`

## Conclusion

The `LO2Loader` class successfully extends LogLead to support OAuth2 microservice log analysis. By following established patterns from existing loaders and adding LO2-specific functionality, it provides a robust foundation for anomaly detection research on distributed authentication systems. The loader integrates seamlessly with LogLead's enhancer and anomaly detection components, enabling comprehensive log analysis workflows.
