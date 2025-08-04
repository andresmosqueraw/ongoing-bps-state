# src/input_handler.py

import pandas as pd
import re
from pathlib import Path
import json
from ongoing_process_state.utils import read_bpmn_model
from pix_framework.io.event_log import EventLogIDs


def ensure_fractional_seconds(ts: str) -> str:
    """
    Ensures the timestamp string has fractional seconds (.000) if none exist.
    
    Examples:
        "2012-01-09T16:45:33"    -> "2012-01-09T16:45:33.000"
        "2012-01-09T16:45:33.45" -> "2012-01-09T16:45:33.45" (unchanged)
        "2012-01-09T16:45:33Z"   -> "2012-01-09T16:45:33.000Z"
        "2012-01-09T16:45:33+02:00" -> "2012-01-09T16:45:33.000+02:00"
    
    Note: We do NOT forcibly add trailing "Z" if it's missing. We only
          add ".000" before the offset or the 'Z' (if it exists), or else
          just append ".000" at the end of the time portion.
    """
    if pd.isnull(ts):
        return ts  # preserve NaNs / None

    # Insert 'T' if it's missing
    if 'T' not in ts:
        # Replace the first space or just ensure 'T' before HH:MM
        ts = ts.replace(' ', 'T', 1)

    # If there's already a decimal, do nothing
    if '.' in ts:
        return ts

    # Otherwise, insert .000 before any offset or 'Z'
    # We'll split out the main time portion vs. offset
    # Using a regex that captures:
    #   group(1): the part up to HH:MM:SS
    #   group(2): the rest (could be empty, or 'Z', or '+01:00', etc.)
    match = re.match(r'^(.*T\d{2}:\d{2}:\d{2})(.*)$', ts)
    if match:
        main_time = match.group(1)     # e.g. "2012-01-09T16:45:33"
        remainder = match.group(2)     # e.g. "" or "Z" or "+02:00"
        ts = main_time + '.000' + remainder
    else:
        # If for some reason it doesn't match the T pattern, just append .000
        # Example: "2012-01-09 16:45:33" (with a space)
        # or something else not strictly matching. We'll just do a naive approach:
        ts += '.000'

    return ts


class InputHandler:
    """Handles the input files and parameter parsing."""
    def __init__(self, args):
        self.event_log_path = Path(args.event_log)
        self.bpmn_model_path = Path(args.bpmn_model)
        self.bpmn_parameters_str = args.bpmn_parameters
        self.start_time = args.start_time
        self.column_mapping_str = args.column_mapping
        self.column_mapping = self.parse_column_mapping()
        self.event_log_df = self.read_event_log() 
        self.event_log_ids = self.get_event_log_ids()
    
    def parse_column_mapping(self):
        """Parses column mapping for both backend-sanitized or raw frontend-style mappings."""

        if isinstance(self.column_mapping_str, str):
            mapping = json.loads(self.column_mapping_str)
        elif isinstance(self.column_mapping_str, dict):
            mapping = self.column_mapping_str
        else:
            raise TypeError("column_mapping must be a str or dict")

        # If the keys are already standard names, assume backend-sanitized mapping
        standard_keys = {'CaseId', 'Activity', 'Resource', 'StartTime', 'EndTime', 'enabled_time'}
        if standard_keys.issubset(set(mapping.keys())):
            return mapping  # Already sanitized

        # Otherwise, assume raw frontend-style and sanitize
        sanitized = {
            "CaseId": mapping.get("case", "CaseId"),
            "Activity": mapping.get("activity", "Activity"),
            "Resource": mapping.get("resource", "Resource"),
            "StartTime": mapping.get("start", "StartTime"),
            "EndTime": mapping.get("end", "EndTime"),
        }

        # Optional: extract enablement
        enablement = mapping.get("enablement")
        if enablement and enablement != "__DISCOVER__":
            sanitized["enabled_time"] = enablement
        elif isinstance(mapping.get("attributes"), dict) and "enable_time" in mapping["attributes"]:
            sanitized["enabled_time"] = mapping["attributes"]["enable_time"]
        else:
            sanitized["enabled_time"] = "enabled_time"

        return sanitized

    
    def get_event_log_ids(self):
        """Returns the EventLogIDs instance with actual column names after mapping."""
        return EventLogIDs(
            case='CaseId',
            activity='Activity',
            resource='Resource',
            start_time='StartTime',
            end_time='EndTime',
            enabled_time='enabled_time'
        )

    def parse_bpmn_parameters(self):
        """
        Interpret self.bpmn_parameters_str as a filename.
        We'll open it and parse the JSON content from that file.
        """
        with open(self.bpmn_parameters_str, 'r') as f:
            return json.load(f)

    def read_event_log(self):
        """Reads the event log CSV file into a DataFrame."""
        df = pd.read_csv(self.event_log_path)

        # Invert column mapping: actual column → standard name
        rename_map = {v: k for k, v in self.column_mapping.items()}
        df = df.rename(columns=rename_map)

        # Validate required standard columns
        required_columns = ['CaseId', 'Resource', 'Activity', 'StartTime', 'EndTime']
        for col in required_columns:
            if col not in df.columns:
                print("Parsed column mapping:", self.column_mapping)
                print("DataFrame columns after renaming:", df.columns.tolist())
                raise ValueError(f"Missing required column: {col}")

        # Convert StartTime
        df['StartTime'] = df['StartTime'].astype(str).apply(ensure_fractional_seconds)
        df['StartTime'] = pd.to_datetime(df['StartTime'], utc=False)

        # Convert EndTime
        df['EndTime'] = df['EndTime'].astype(str).apply(ensure_fractional_seconds)
        missing_end = df['EndTime'].isna()
        if missing_end.any():
            print(f"There are {missing_end.sum()} rows with missing/invalid EndTime.")
            print(df[missing_end].head())
        df['EndTime'] = pd.to_datetime(df['EndTime'], utc=False)

        return df

    def read_bpmn_model(self):
        """Reads the BPMN model file."""
        bpmn_model = read_bpmn_model(self.bpmn_model_path)
        return bpmn_model
