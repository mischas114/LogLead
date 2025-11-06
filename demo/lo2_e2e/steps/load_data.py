"""Data loading step for pipeline."""

from pathlib import Path
from typing import Any, Dict
import polars as pl

from ..core.registry import register_step


@register_step("load_data")
def run(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Load data from parquet files into context.
    
    Args:
        context: Pipeline context dictionary
        **kwargs: Step parameters
            - sequences_path: Path to sequences parquet file
            - events_path: Optional path to events parquet file
            
    Returns:
        Updated context with 'df_sequences' and optionally 'df_events'
    """
    sequences_path = kwargs.get("sequences_path")
    events_path = kwargs.get("events_path")
    
    if not sequences_path:
        raise ValueError("load_data step requires 'sequences_path' parameter")
    
    seq_path = Path(sequences_path)
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    
    print(f"[load_data] Loading sequences from {sequences_path}")
    df_sequences = pl.read_parquet(seq_path)
    context["df_sequences"] = df_sequences
    print(f"[load_data] Loaded {len(df_sequences)} sequences")
    
    if events_path:
        evt_path = Path(events_path)
        if evt_path.exists():
            print(f"[load_data] Loading events from {events_path}")
            df_events = pl.read_parquet(evt_path)
            context["df_events"] = df_events
            print(f"[load_data] Loaded {len(df_events)} events")
        else:
            print(f"[load_data] Warning: Events file not found: {events_path}")
    
    return context
