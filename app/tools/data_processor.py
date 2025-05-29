"""
Data Processing Tools for MCP Server

This module provides tools for processing and formatting tabular data
to be optimally consumed by LLMs. It uses dynamic column profiling
to understand data structure and apply appropriate transformations.

Usage:
    from app.tools.data_processor import register_tools
    register_tools(mcp)
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from fastmcp import Context
from app.utils.column_profiler import ColumnProfiler, ColumnMetadata # Import the profiler

# Configure logging
logger = logging.getLogger("mcp_server.tools.data_processor")

# Helper function to ensure DataFrame
def _ensure_dataframe(data: Union[List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy() # Return a copy to avoid modifying original
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return pd.DataFrame(data)
    else:
        raise ValueError("Data must be a DataFrame or a list of dictionaries")

def filter_rows(
    df: pd.DataFrame,
    metadata_map: Dict[str, ColumnMetadata], # Unused in this version, but kept for potential future use
    limit: Optional[int] = None,
    sort_by: Optional[str] = None, # Original column name
    ascending: bool = False,
    filter_conditions: Optional[Dict[str, Any]] = None # Keys are original column names
) -> pd.DataFrame:
    """Filter and sort rows based on specified criteria."""
    filtered_df = df.copy()

    if filter_conditions:
        for original_col_name, value in filter_conditions.items():
            if original_col_name in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[original_col_name].isin(value)]
                else:
                    # Attempt type conversion for comparison if column is numeric in metadata
                    meta = metadata_map.get(original_col_name)
                    if meta and meta.semantic_type in ["Currency", "Integer", "Float", "Percentage", "Year"]:
                        try:
                            # Convert filter value to column's numeric type if possible
                            col_dtype = pd.to_numeric(df[original_col_name], errors='coerce').dtype
                            typed_value = pd.Series([value]).astype(col_dtype).iloc[0]
                            filtered_df = filtered_df[pd.to_numeric(filtered_df[original_col_name], errors='coerce') == typed_value]
                        except Exception: # Fallback to string comparison if conversion fails
                            filtered_df = filtered_df[filtered_df[original_col_name].astype(str) == str(value)]
                    else: # Default to string comparison for non-numeric or unknown types
                         filtered_df = filtered_df[filtered_df[original_col_name].astype(str) == str(value)]
            else:
                logger.warning(f"Filter column '{original_col_name}' not found in DataFrame.")


    if sort_by and sort_by in filtered_df.columns:
        meta = metadata_map.get(sort_by)
        if meta and meta.semantic_type in ["Currency", "Integer", "Float", "Percentage", "Year", "Date", "Timestamp"]:
            # For numeric-like or date types, attempt direct sort or numeric conversion
            try:
                # Attempt to convert to numeric, coercing errors for mixed types
                numeric_series = pd.to_numeric(filtered_df[sort_by], errors='coerce')
                if not numeric_series.isnull().all(): # If at least one value is numeric
                     # Create a temporary column for sorting to handle NaNs correctly with numeric sort
                    temp_sort_col = f"__sort_{sort_by}"
                    filtered_df[temp_sort_col] = numeric_series
                    # Put NaNs last when ascending, first when descending if they are present
                    na_position = 'last' if ascending else 'first'
                    filtered_df = filtered_df.sort_values(by=temp_sort_col, ascending=ascending, na_position=na_position)
                    filtered_df = filtered_df.drop(columns=[temp_sort_col])
                else: # If all are NaN after coercion, sort as object
                    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
            except Exception: # Fallback to object type sorting if robust numeric conversion fails
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        else: # Default to object type sorting for other types
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending, na_position='last')


    if limit is not None and limit > 0: # Allow limit=0 to mean no limit, or handle as error if preferred
        filtered_df = filtered_df.head(limit)
    
    return filtered_df

def select_columns(
    df: pd.DataFrame,
    metadata_map: Dict[str, ColumnMetadata], # Unused in this version
    columns: Optional[List[str]] = None, # List of original column names
    exclude_columns: Optional[List[str]] = None # List of original column names
) -> pd.DataFrame:
    """Select or exclude specific columns from the DataFrame."""
    if columns:
        cols_to_keep = [col for col in columns if col in df.columns]
        if not cols_to_keep and columns: # If user specified columns but none matched
            logger.warning(f"None of the specified columns to keep were found: {columns}. Returning all columns.")
            selected_df = df.copy()
        else:
            selected_df = df[cols_to_keep].copy()
    else:
        selected_df = df.copy()

    if exclude_columns:
        cols_to_drop = [col for col in exclude_columns if col in selected_df.columns]
        selected_df = selected_df.drop(columns=cols_to_drop)
    
    return selected_df

def detect_anomalies(df: pd.DataFrame, metadata_map: Dict[str, ColumnMetadata]) -> List[str]:
    """Detect and highlight anomalies in the data."""
    anomaly_notes = []
    
    for original_col_name, meta in metadata_map.items():
        if original_col_name not in df.columns: continue

        series = df[original_col_name]
        
        # Missing values (already in meta.null_count)
        if meta.null_count > 0:
            anomaly_notes.append(f"Column '{meta.cleaned_name}' has {meta.null_count} missing values ({meta.null_ratio:.1%}).")

        if meta.semantic_type == "Currency" or "revenue" in meta.original_name.lower() or "amount" in meta.original_name.lower() :
            if pd.api.types.is_numeric_dtype(series.dtype): # Check if it's actually numeric after profiling
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if not numeric_series.empty:
                    neg_count = (numeric_series < 0).sum()
                    if neg_count > 0:
                        anomaly_notes.append(f"Column '{meta.cleaned_name}' (Currency) has {neg_count} negative values.")
                    zero_count = (numeric_series == 0).sum()
                    if zero_count > 0:
                         anomaly_notes.append(f"Column '{meta.cleaned_name}' (Currency) has {zero_count} zero values.")
        
        # Outlier detection using IQR for numeric types
        if meta.semantic_type in ["Integer", "Float", "Currency", "Percentage", "Year"] and meta.count > 0:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 1: # IQR requires at least 2 points
                Q1 = numeric_series.quantile(0.25)
                Q3 = numeric_series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0: # Avoid division by zero or issues with constant series
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
                    if not outliers.empty:
                        anomaly_notes.append(f"Column '{meta.cleaned_name}' has {len(outliers)} potential outliers (values outside {lower_bound:.2f} - {upper_bound:.2f}).")
    return anomaly_notes

def add_summary_row(df: pd.DataFrame, metadata_map: Dict[str, ColumnMetadata]) -> pd.DataFrame:
    """Add a summary row with totals, averages, or other aggregations."""
    if df.empty:
        return df

    summary_row_dict = {}
    profiler = ColumnProfiler() # For cleaning the 'TOTAL' label if needed

    # First column for "TOTAL" label
    first_col_original_name = df.columns[0]
    first_col_meta = metadata_map.get(first_col_original_name)
    summary_label_cleaned, _ = profiler._clean_column_name("TOTAL")
    summary_row_dict[first_col_original_name] = summary_label_cleaned


    for original_col_name in df.columns[1:]: # Skip first column
        if original_col_name not in metadata_map:
            summary_row_dict[original_col_name] = "" # Placeholder for unknown columns
            continue

        meta = metadata_map[original_col_name]
        series = pd.to_numeric(df[original_col_name], errors='coerce') # Attempt numeric conversion

        if not series.dropna().empty: # If there are any numeric values after coercion
            if meta.semantic_type in ["Currency", "Integer", "Float"] or \
               any(term in meta.original_name.lower() for term in ["count", "qty", "items"]):
                summary_row_dict[original_col_name] = series.sum()
            elif meta.semantic_type == "Percentage" or \
                 any(term in meta.original_name.lower() for term in ["rate", "ratio", "avg", "mean"]):
                summary_row_dict[original_col_name] = series.mean()
            else: # Default for other numeric types if not specified
                summary_row_dict[original_col_name] = "" # Or series.sum() or ""
        else:
            summary_row_dict[original_col_name] = "" # Placeholder for non-numeric or all-NaN columns

    # Create a DataFrame from the summary_row_dict to ensure proper alignment with df.columns
    summary_df_row = pd.DataFrame([summary_row_dict], columns=df.columns)
    
    # Concatenate with original DataFrame
    # If df has a MultiIndex, this might need adjustment, but assuming single index for now
    summary_df = pd.concat([df, summary_df_row], ignore_index=True)
    return summary_df


def add_ranking_indicators(df: pd.DataFrame, metadata_map: Dict[str, ColumnMetadata], original_indicator_column_name: str) -> pd.DataFrame:
    """Add ranking indicators (↑, ↓, →) to highlight performance."""
    if original_indicator_column_name not in df.columns or original_indicator_column_name not in metadata_map:
        logger.warning(f"Indicator column '{original_indicator_column_name}' not found or no metadata. Skipping indicators.")
        return df

    ranked_df = df.copy()
    meta = metadata_map[original_indicator_column_name]
    
    # New column name for the indicator
    indicator_col_name_raw = f"{original_indicator_column_name}_Trend" # Raw name, will be cleaned later
    
    series_to_rank = pd.to_numeric(df[original_indicator_column_name], errors='coerce')

    if not series_to_rank.dropna().empty:
        median_value = series_to_rank.median()
        if pd.notna(median_value): # Ensure median is not NaN
            ranked_df[indicator_col_name_raw] = series_to_rank.apply(
                lambda x: "↑" if pd.notna(x) and x > median_value else \
                          ("↓" if pd.notna(x) and x < median_value else \
                           ("→" if pd.notna(x) and x == median_value else "")) # Handle NaN x values
            )
        else: # If median is NaN (e.g. all NaNs in series)
            ranked_df[indicator_col_name_raw] = ""
    else: # If series is all NaNs or empty after coercion
        ranked_df[indicator_col_name_raw] = ""
        
    return ranked_df

def format_cell_values(df: pd.DataFrame, metadata_map: Dict[str, ColumnMetadata]) -> pd.DataFrame:
    """Format cell values into strings based on their semantic type and formatting suggestions."""
    formatted_df = df.copy().astype(object) # Convert to object to store formatted strings

    for original_col_name in df.columns:
        if original_col_name not in metadata_map: # Likely a newly added column (e.g. Trend, or TOTAL label)
            # Apply basic string conversion for these, or specific logic if name is known
            if original_col_name.endswith("_Trend"): # Handle trend indicators
                 pass # Already strings (↑, ↓, →)
            elif df[original_col_name].iloc[-1] == "TOTAL" and df.columns.get_loc(original_col_name) == 0 : # Summary row label
                 pass # Already string 'TOTAL'
            else: # Default for other new columns or if metadata is missing
                formatted_df[original_col_name] = df[original_col_name].astype(str)
            continue

        meta = metadata_map[original_col_name]
        suggestion = meta.formatting_suggestion
        
        # Handle the summary row label explicitly if it's in a data column
        # This check is more robust if the summary row's first cell is not 'TOTAL'
        # but a value that should not be formatted.
        # However, our add_summary_row puts 'TOTAL' in the first original column.
        # So, for other columns in the summary row, they contain aggregated numbers.

        for i, val in enumerate(df[original_col_name]):
            if pd.isna(val):
                formatted_df.loc[i, original_col_name] = "N/A"
                continue
            
            # Check if this is the summary row and if the current column is NOT the first one (label column)
            # This assumes the summary row is the last row.
            is_summary_value = (i == len(df) - 1) and (df.iloc[-1, 0] == "TOTAL") and (df.columns.get_loc(original_col_name) > 0)

            current_suggestion = suggestion
            # For summary row values that are sums/means of currencies, ensure they are also formatted as currency
            if is_summary_value and meta.semantic_type == "Currency" and not (suggestion and suggestion.startswith("currency")):
                current_suggestion = f"currency:{meta.units or 'USD'}:,.2f"


            if current_suggestion:
                try:
                    if current_suggestion.startswith("currency"):
                        _, unit, fmt = current_suggestion.split(":")
                        prefix = ""
                        if unit == "USD": prefix = "$"
                        elif unit == "EUR": prefix = "€"
                        elif unit == "GBP": prefix = "£"
                        # Basic numeric check before formatting
                        if isinstance(val, (int, float, np.number)):
                            formatted_df.loc[i, original_col_name] = f"{prefix}{val:{fmt.replace('f', '').strip()}}" if fmt else f"{prefix}{val}"
                        else: # If not numeric, keep as is or N/A
                            formatted_df.loc[i, original_col_name] = str(val) if pd.notna(val) else "N/A"
                    elif current_suggestion.startswith("percentage"):
                        _, fmt = current_suggestion.split(":")
                        if isinstance(val, (int, float, np.number)):
                            formatted_df.loc[i, original_col_name] = f"{val:{fmt}}%"
                        else:
                            formatted_df.loc[i, original_col_name] = str(val) if pd.notna(val) else "N/A"
                    elif current_suggestion.startswith("date"):
                        _, fmt = current_suggestion.split(":")
                        formatted_df.loc[i, original_col_name] = pd.to_datetime(val).strftime(fmt)
                    elif current_suggestion.startswith("datetime"):
                        _, fmt = current_suggestion.split(":")
                        formatted_df.loc[i, original_col_name] = pd.to_datetime(val).strftime(fmt)
                    elif current_suggestion.startswith("integer"):
                        _, fmt = current_suggestion.split(":")
                        if isinstance(val, (int, float, np.number)): # float for cases like mean of integers
                            formatted_df.loc[i, original_col_name] = f"{int(round(val)):{fmt}}"
                        else:
                            formatted_df.loc[i, original_col_name] = str(val) if pd.notna(val) else "N/A"
                    elif current_suggestion.startswith("float"):
                        _, fmt = current_suggestion.split(":")
                        if isinstance(val, (int, float, np.number)):
                            formatted_df.loc[i, original_col_name] = f"{val:{fmt}}"
                        else:
                            formatted_df.loc[i, original_col_name] = str(val) if pd.notna(val) else "N/A"
                    else: # Fallback if suggestion format is unknown
                        formatted_df.loc[i, original_col_name] = str(val)
                except Exception as e:
                    logger.warning(f"Could not apply formatting suggestion '{current_suggestion}' to value '{val}' in column '{original_col_name}': {e}")
                    formatted_df.loc[i, original_col_name] = str(val) # Fallback to string
            else: # No suggestion, just convert to string
                formatted_df.loc[i, original_col_name] = str(val)
                
    return formatted_df


def generate_column_descriptions_from_metadata(
    metadata_map: Dict[str, ColumnMetadata],
    displayed_original_columns: List[str] # Original names of columns present in the final table
) -> str:
    """Generate human-readable descriptions for columns using metadata."""
    descriptions = []
    for original_col_name in displayed_original_columns:
        if original_col_name in metadata_map:
            meta = metadata_map[original_col_name]
            # Use profile_summary as it's concise and informative
            descriptions.append(f"- **{meta.cleaned_name}{f' ({meta.units})' if meta.units else ''}**: {meta.profile_summary}")
    
    if descriptions:
        return "### Column Descriptions & Profiling Summary\n" + "\n".join(descriptions)
    return ""

def format_as_markdown(
    df_final_display: pd.DataFrame, # DataFrame with final display names and formatted string values
    metadata_map: Dict[str, ColumnMetadata],
    original_columns_in_final_df: List[str], # Original names of data columns that are in df_final_display
    title: Optional[str] = None,
    include_descriptions: bool = True,
    anomaly_notes: Optional[List[str]] = None
) -> str:
    """Format DataFrame as a Markdown table."""
    result_parts = []
    if title:
        result_parts.extend([f"## {title}", ""])
    
    if include_descriptions:
        descriptions_md = generate_column_descriptions_from_metadata(metadata_map, original_columns_in_final_df)
        if descriptions_md:
            result_parts.extend([descriptions_md, ""])
            
    if df_final_display.empty:
        result_parts.append("No data to display.")
    else:
        table_md = df_final_display.to_markdown(index=False)
        result_parts.append(table_md)
    
    if anomaly_notes: # anomaly_notes are already formatted
        result_parts.extend(["", "### Data Quality Notes"])
        result_parts.extend([f"- {note}" for note in anomaly_notes])
        
    return "\n".join(result_parts)

def format_as_bullets(
    df_final_display: pd.DataFrame, # DataFrame with final display names and formatted string values
    title: Optional[str] = None
) -> str:
    """Format DataFrame as bullet points."""
    result_parts = []
    if title:
        result_parts.extend([f"## {title}", ""])

    if df_final_display.empty:
        result_parts.append("No data to display.")
        return "\n".join(result_parts)

    key_column_display_name = df_final_display.columns[0]
    value_column_display_names = df_final_display.columns[1:]

    for _, row in df_final_display.iterrows():
        key_val = row[key_column_display_name]
        bullet = f"- **{key_val}**:"
        
        values_str = []
        for val_col_disp_name in value_column_display_names:
            values_str.append(f"{val_col_disp_name}: {row[val_col_disp_name]}")
        
        if values_str:
            bullet += " " + ", ".join(values_str)
        result_parts.append(bullet)
        
    return "\n".join(result_parts)

def format_as_json(
    df_final_display: pd.DataFrame, # DataFrame with final display names and formatted string values
    metadata_map: Dict[str, ColumnMetadata],
    original_columns_in_final_df: List[str],
    include_profiler_metadata: bool = True
) -> str:
    """Format DataFrame as JSON, optionally including profiler metadata."""
    records = df_final_display.to_dict(orient="records")
    output_data = {"data": records}

    if include_profiler_metadata:
        output_data["column_profiles"] = {
            metadata_map[og_col_name].cleaned_name: metadata_map[og_col_name].dict(exclude_none=True)
            for og_col_name in original_columns_in_final_df if og_col_name in metadata_map
        }
    return json.dumps(output_data, indent=2, default=str) # default=str for non-serializable types

# --- Main Tools ---
def register_tools(mcp):
    """Register data processing tools with the MCP server"""
    
    profiler = ColumnProfiler()

    @mcp.tool()
    async def format_table_for_llm(
        ctx: Context,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        limit: Optional[int] = 10,
        sort_by: Optional[str] = None, # Original column name
        ascending: bool = False,
        currency_override: Optional[str] = None, # E.g., "USD", "CAD"
        columns_to_include: Optional[List[str]] = None, # Original column names
        columns_to_exclude: Optional[List[str]] = None, # Original column names
        output_format: str = "markdown", # "markdown", "bullets", "json"
        add_summary_row_flag: bool = True,
        add_ranking_indicators_flag: bool = True,
        indicator_column_original_name: Optional[str] = None, # Original column name
        include_anomaly_detection_notes: bool = True,
        include_column_profile_summaries: bool = True,
    ) -> str:
        """
        Dynamically profiles, cleans, formats, and annotates tabular data for optimal LLM consumption.
        Uses ColumnProfiler for intelligent, schema-agnostic processing.

        Args:
            ctx: The MCP context.
            data: Raw tabular data as a list of dictionaries.
            title: Optional title for the output.
            limit: Maximum number of rows to include (0 or None for all).
            sort_by: Original column name to sort by.
            ascending: Sort order.
            currency_override: Override inferred currency (e.g., "USD").
            columns_to_include: List of original column names to specifically include.
            columns_to_exclude: List of original column names to specifically exclude.
            output_format: Desired output format ("markdown", "bullets", "json").
            add_summary_row_flag: Whether to add a summary row.
            add_ranking_indicators_flag: Whether to add trend indicators.
            indicator_column_original_name: Original column name to base trend indicators on (defaults to sort_by).
            include_anomaly_detection_notes: Whether to include data quality notes.
            include_column_profile_summaries: Whether to include column profile summaries in markdown.
            
        Returns:
            Formatted data as a string in the specified format.
        """
        try:
            df_original = _ensure_dataframe(data)
            if df_original.empty:
                return "No data available to format."

            metadata_map = profiler.profile_dataframe(df_original)

            # 1. Filtering and Sorting (operates on original column names)
            df_processed = filter_rows(df_original, metadata_map, limit, sort_by, ascending)

            # 2. Column Selection (operates on original column names)
            df_processed = select_columns(df_processed, metadata_map, columns_to_include, columns_to_exclude)
            
            if df_processed.empty:
                return "No data remaining after filtering or column selection."

            # Store original column names that are still in df_processed for later use
            # (e.g. for generating descriptions for only the displayed columns)
            original_columns_in_scope = [col for col in df_processed.columns if col in metadata_map]


            # 3. Anomaly Detection (produces notes, doesn't change df_processed structure yet)
            anomaly_notes = []
            if include_anomaly_detection_notes:
                anomaly_notes = detect_anomalies(df_processed, metadata_map)

            # 4. Add Summary Row (adds a row, columns are still original names)
            if add_summary_row_flag:
                # Check if there's at least one numeric-like column to summarize
                can_summarize = any(
                    meta.semantic_type in ["Currency", "Integer", "Float", "Percentage"]
                    for col_name, meta in metadata_map.items() if col_name in df_processed.columns
                )
                if can_summarize:
                    df_processed = add_summary_row(df_processed, metadata_map)
                else:
                    logger.info("No numeric-like columns to summarize, skipping summary row.")


            # 5. Add Ranking Indicators (adds a new column with a raw name)
            if add_ranking_indicators_flag:
                target_indicator_col = indicator_column_original_name or sort_by
                if target_indicator_col and target_indicator_col in df_processed.columns:
                    df_processed = add_ranking_indicators(df_processed, metadata_map, target_indicator_col)
            
            # 6. Format Cell Values (modifies cell values to strings, column names still original/raw)
            # Pass the currency_override to format_cell_values if needed, or let it use metadata.
            # For now, format_cell_values uses metadata_map[col].units which gets it from profiler.
            # If currency_override is provided, the profiler should ideally use it, or we pass it down.
            # Let's assume ColumnProfiler can take a currency_override or format_cell_values can.
            # For now, the profiler's unit inference is primary.
            df_formatted_cells = format_cell_values(df_processed, metadata_map)

            # 7. Generate Final Display Column Names and create df_final_display
            df_final_display = pd.DataFrame()
            final_column_names_ordered = [] # To maintain order

            for raw_col_name in df_formatted_cells.columns: # Iterate in current order
                display_name = ""
                if raw_col_name in metadata_map: # It's an original column
                    meta = metadata_map[raw_col_name]
                    display_name = meta.cleaned_name
                    # Use currency_override if provided and type is Currency, else use meta.units
                    units_to_display = meta.units
                    if currency_override and meta.semantic_type == "Currency":
                        units_to_display = currency_override
                    if units_to_display:
                         display_name += f" ({units_to_display})"
                else: # It's a newly added column (e.g., "TOTAL" label, Trend indicator)
                    cleaned_added_col, _ = profiler._clean_column_name(raw_col_name)
                    display_name = cleaned_added_col
                
                df_final_display[display_name] = df_formatted_cells[raw_col_name]
                final_column_names_ordered.append(display_name)
            
            # Ensure column order is preserved
            df_final_display = df_final_display[final_column_names_ordered]


            # 8. Format output
            if output_format.lower() == "bullets":
                return format_as_bullets(df_final_display, title=title)
            elif output_format.lower() == "json":
                return format_as_json(df_final_display, metadata_map, original_columns_in_scope, include_profiler_metadata=True)
            else: # Default to markdown
                return format_as_markdown(
                    df_final_display,
                    metadata_map,
                    original_columns_in_scope, # Pass original names of *data* columns that made it to final display
                    title=title,
                    include_descriptions=include_column_profile_summaries,
                    anomaly_notes=anomaly_notes if include_anomaly_detection_notes else None
                )

        except Exception as e:
            logger.error(f"Error in format_table_for_llm: {e}", exc_info=True)
            return f"Error formatting table: {str(e)}"

    @mcp.tool()
    async def clean_and_prepare_table(
        ctx: Context,
        data: List[Dict[str, Any]],
        query: Optional[str] = None, # User query to guide cleaning
        title_prefix: str = "Cleaned Data"
    ) -> str:
        """
        Automatically cleans, profiles, and prepares tabular data based on content and an optional user query.
        This tool aims for smart defaults for LLM consumption. Output is Markdown.
        """
        try:
            df_original = _ensure_dataframe(data)
            if df_original.empty:
                return "No data available to clean and prepare."

            metadata_map = profiler.profile_dataframe(df_original)
            
            # --- Auto-determine parameters based on query and metadata ---
            limit = 10 # Default limit
            sort_by_original_name: Optional[str] = None
            ascending_sort = False
            columns_to_include_original_names: Optional[List[str]] = None # Auto-select later if still None
            
            final_title = title_prefix
            if query:
                final_title = f"{title_prefix} (for query: '{query[:50]}{'...' if len(query)>50 else ''}')"
                query_lower = query.lower()

                # Limit detection
                limit_match = re.search(r"(?:top|first|last|bottom)\s*(\d+)", query_lower)
                if limit_match:
                    limit = int(limit_match.group(1))
                    logger.info(f"Query implies limit: {limit}")

                # Sort detection (simple keyword-based for now)
                # Prioritize columns mentioned in query that are also numeric-like
                potential_sort_cols = [
                    name for name, meta in metadata_map.items() 
                    if meta.semantic_type in ["Currency", "Integer", "Float", "Percentage", "Year"] and 
                       (name.lower() in query_lower or meta.cleaned_name.lower() in query_lower)
                ]
                if not potential_sort_cols: # Fallback to any numeric-like column
                    potential_sort_cols = [name for name, meta in metadata_map.items() if meta.semantic_type in ["Currency", "Integer", "Float", "Percentage", "Year"]]

                if potential_sort_cols:
                    # Heuristic: if query mentions "top", "highest", "most", "best", "largest" -> descending
                    if any(term in query_lower for term in ["top", "highest", "most", "best", "largest", "biggest"]):
                        sort_by_original_name = potential_sort_cols[0] # Pick first potential
                        ascending_sort = False
                    # Heuristic: if query mentions "bottom", "lowest", "least", "worst", "smallest" -> ascending
                    elif any(term in query_lower for term in ["bottom", "lowest", "least", "worst", "smallest"]):
                        sort_by_original_name = potential_sort_cols[0]
                        ascending_sort = True
                    elif len(potential_sort_cols) == 1: # If only one numeric col mentioned/available, use it
                        sort_by_original_name = potential_sort_cols[0]
                
                if sort_by_original_name:
                     logger.info(f"Query implies sort by: {sort_by_original_name}, ascending: {ascending_sort}")
                
                # Column selection based on query (simple version)
                # If query mentions specific column names (cleaned or original)
                mentioned_cols = []
                for og_name, meta in metadata_map.items():
                    if og_name.lower() in query_lower or meta.cleaned_name.lower() in query_lower:
                        mentioned_cols.append(og_name)
                if mentioned_cols:
                    columns_to_include_original_names = mentioned_cols
                    logger.info(f"Query implies including columns: {columns_to_include_original_names}")


            # Call the main formatter with auto-detected and default parameters
            return await format_table_for_llm(
                ctx,
                data=data, # Pass original list of dicts
                title=final_title,
                limit=limit,
                sort_by=sort_by_original_name,
                ascending=ascending_sort,
                currency_override=None, # Let profiler infer or use default
                columns_to_include=columns_to_include_original_names, # Can be None
                columns_to_exclude=None,
                output_format="markdown", # Default for this tool
                add_summary_row_flag=True,
                add_ranking_indicators_flag=True,
                indicator_column_original_name=sort_by_original_name, # Use sort_by as indicator
                include_anomaly_detection_notes=True,
                include_column_profile_summaries=True
            )

        except Exception as e:
            logger.error(f"Error in clean_and_prepare_table: {e}", exc_info=True)
            return f"Error cleaning and preparing table: {str(e)}"

    logger.info("Registered data processing tools using ColumnProfiler: format_table_for_llm, clean_and_prepare_table")

