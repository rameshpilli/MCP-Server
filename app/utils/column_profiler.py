#!/usr/bin/env python3
"""
Dynamic Column Profiler

This module provides a ColumnProfiler class that uses pandas and heuristics
to automatically extract metadata from DataFrame columns. This metadata includes
cleaned names, inferred semantic types, units, descriptions, basic statistics,
and formatting suggestions.

The goal is to replace hardcoded mappings and enable dynamic data understanding
for tools that process tabular data, especially for LLM consumption.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from pydantic import BaseModel, Field
import logging
import json # For example usage

logger = logging.getLogger(__name__)

# --- Pydantic Model for Column Metadata ---
class ColumnMetadata(BaseModel):
    original_name: str = Field(description="Original column name in the DataFrame.")
    cleaned_name: str = Field(description="Cleaned, human-readable column name.")
    semantic_type: str = Field(description="Inferred semantic type of the column (e.g., Currency, Percentage, ID, Category).")
    pandas_dtype: str = Field(description="Pandas data type of the column.")
    units: Optional[str] = Field(None, description="Inferred units for the column (e.g., USD, %, items).")
    description: str = Field(description="Auto-generated description of the column's content and characteristics.")
    formatting_suggestion: Optional[str] = Field(None, description="Suggested formatting string or rule (e.g., '%.2f', 'currency', 'date:%Y-%m-%d').")
    
    # Basic Statistics
    count: int = Field(description="Number of non-null values.")
    null_count: int = Field(description="Number of null values.")
    null_ratio: float = Field(description="Ratio of null values to total values.")
    unique_count: int = Field(description="Number of unique values.")
    unique_ratio: float = Field(description="Ratio of unique values to non-null values.")
    
    # Numeric Specific Stats (Optional)
    min_value: Optional[Union[float, int, str]] = Field(None, description="Minimum value (for numeric or date types).")
    max_value: Optional[Union[float, int, str]] = Field(None, description="Maximum value (for numeric or date types).")
    mean_value: Optional[float] = Field(None, description="Mean value (for numeric types).")
    median_value: Optional[float] = Field(None, description="Median value (for numeric types).")
    std_dev: Optional[float] = Field(None, description="Standard deviation (for numeric types).")
    
    # String Specific Stats (Optional)
    min_length: Optional[int] = Field(None, description="Minimum string length.")
    max_length: Optional[int] = Field(None, description="Maximum string length.")
    avg_length: Optional[float] = Field(None, description="Average string length.")
    
    # Categorical Specific Stats (Optional)
    top_categories: Optional[Dict[str, int]] = Field(None, description="Top N most frequent categories and their counts.")
    
    profile_summary: str = Field(description="A concise textual summary of the column profile.")

class ColumnProfiler:
    """
    Profiles DataFrame columns to extract metadata dynamically.
    """
    COMMON_CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹']
    COMMON_CURRENCY_CODES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR']
    COMMON_ID_PATTERNS = [r'id$', r'_id$', r'key$', r'code$', r'number$', r'no$']
    COMMON_DATE_KEYWORDS = ['date', 'time', 'day', 'month', 'year', 'dt', 'ts']
    COMMON_TEXT_KEYWORDS = ['text', 'desc', 'comment', 'note', 'summary', 'message', 'content', 'name', 'title']

    def _clean_column_name(self, col_name: str) -> Tuple[str, Optional[str]]:
        """
        Cleans a column name to be human-readable and extracts potential units.
        Example: "ytd_rev_usd" -> ("Revenue YTD", "USD")
        """
        name = str(col_name)
        unit_extracted = None

        # Check for common currency codes at the end
        for code in self.COMMON_CURRENCY_CODES:
            if name.lower().endswith(f"_{code.lower()}"):
                unit_extracted = code.upper()
                name = name[:-len(code)-1]
                break
            elif name.lower().endswith(code.lower()): # e.g. revenueUSD
                 unit_extracted = code.upper()
                 name = name[:-len(code)]
                 break
        
        # Replace underscores/hyphens with spaces, then title case
        name = re.sub(r'[_\-]+', ' ', name)
        name = name.title()
        
        # Handle common abbreviations
        name = name.replace("Ytd", "YTD").replace("Yoy", "YoY").replace("Id", "ID")
        name = name.replace("Prev ", "Previous ")
        name = name.replace("Pct", "Percent")

        # Remove trailing " In Xyz" if unit was extracted from there
        if unit_extracted:
            name = re.sub(rf"\s*In\s*{unit_extracted}$", "", name, flags=re.IGNORECASE).strip()

        # Further unit detection if not found yet (e.g. "Revenue (USD)")
        match = re.search(r'\((.*?)\)$', name)
        if match:
            potential_unit = match.group(1).upper()
            if potential_unit in self.COMMON_CURRENCY_CODES or potential_unit == '%':
                if not unit_extracted: # Prioritize suffix extraction
                    unit_extracted = potential_unit
                name = name[:match.start()].strip() # Remove unit from name

        cleaned_name = name.strip()
        return cleaned_name, unit_extracted

    def _infer_semantic_type_and_units(self, series: pd.Series, col_name_original: str, cleaned_name: str) -> Tuple[str, Optional[str]]:
        """
        Infers a more specific semantic type and units for a column.
        """
        s_no_na = series.dropna()
        dtype = series.dtype
        n_unique = s_no_na.nunique()
        n_total = len(series)
        unique_ratio = n_unique / n_total if n_total > 0 else 0.0
        col_name_lower = col_name_original.lower()
        
        units = None

        # 0. Constant
        if n_unique == 1:
            return "Constant", None

        # 1. Boolean
        if dtype == np.bool_ or (n_unique == 2 and s_no_na.isin([0, 1, True, False, 'Yes', 'No', 'Y', 'N']).all()):
            return "Boolean", None

        # 2. IDs
        is_id_like_name = any(re.search(pattern, col_name_lower) for pattern in self.COMMON_ID_PATTERNS)
        if is_id_like_name:
            if (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_string_dtype(dtype)) and unique_ratio > 0.9:
                return "Identifier", None
        
        # 3. Numeric types
        if pd.api.types.is_numeric_dtype(dtype):
            # Check for year
            if "year" in col_name_lower and pd.api.types.is_integer_dtype(dtype):
                if s_no_na.empty or s_no_na.between(1900, 2100).all(): # Handle empty series for year check
                    return "Year", None
            
            # Check for percentage
            if "%" in col_name_lower or "percent" in col_name_lower or "rate" in col_name_lower or "ratio" in col_name_lower:
                 # Check if values are in typical percentage ranges, handling empty series
                if s_no_na.empty or \
                   (s_no_na.between(0, 1).all() and s_no_na.max() <=1) or \
                   (s_no_na.between(0, 100).all() and s_no_na.max() <=100):
                    return "Percentage", "%"
            
            # Check for currency (more robustly)
            currency_keywords = ["price", "amount", "cost", "revenue", "salary", "value", "balance", "fee", "charge", "payment"]
            if any(keyword in col_name_lower for keyword in currency_keywords):
                # Try to infer currency code from name or values
                for code in self.COMMON_CURRENCY_CODES:
                    if code.lower() in col_name_lower:
                        units = code
                        break
                if not units and not s_no_na.empty and s_no_na.astype(str).str.contains('|'.join(re.escape(s) for s in self.COMMON_CURRENCY_SYMBOLS)).any():
                     # If symbols are present, try to determine which one. Default to USD if ambiguous.
                    sample_val_str_series = s_no_na.astype(str)
                    symbol_containing_series = sample_val_str_series[sample_val_str_series.str.contains('|'.join(re.escape(s) for s in self.COMMON_CURRENCY_SYMBOLS))]
                    if not symbol_containing_series.empty:
                        sample_val_str = symbol_containing_series.iloc[0]
                        if '$' in sample_val_str: units = 'USD'
                        elif '€' in sample_val_str: units = 'EUR'
                        elif '£' in sample_val_str: units = 'GBP'
                        # ... add more symbol to code mappings
                        else: units = 'USD' # Fallback
                return "Currency", units

            if pd.api.types.is_integer_dtype(dtype):
                return "Integer", None
            return "Float", None

        # 4. Datetime types
        if pd.api.types.is_datetime64_any_dtype(dtype):
            if s_no_na.empty or s_no_na.dt.normalize().equals(s_no_na): # All times are 00:00:00 or empty
                return "Date", None
            return "Timestamp", None
        
        # 5. String/Object types
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            if s_no_na.empty: # If all are NaN after dropna, treat as unknown or categorical based on name
                return "Categorical" if n_unique < 10 else "String", None

            # Try to convert to datetime
            try:
                s_dates = pd.to_datetime(s_no_na, errors='raise')
                if s_dates.dt.normalize().equals(s_dates):
                    return "Date", None
                return "Timestamp", None
            except (ValueError, TypeError, OverflowError):
                pass # Not datetime

            # Check for URL
            if s_no_na.astype(str).str.match(r'^https?://[^\s/$.?#].[^\s]*$', case=False).any():
                return "URL", None

            # Check for Email
            if s_no_na.astype(str).str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').any():
                return "Email", None

            # Categorical vs Free Text
            # Heuristic: if unique ratio is low OR number of unique values is small relative to total, it's categorical
            if unique_ratio < 0.5 or n_unique < max(15, 0.05 * n_total if n_total > 0 else 15) : # Adjust thresholds as needed
                return "Categorical", None
            
            # If high unique ratio and longer strings, likely free text
            avg_len = s_no_na.astype(str).str.len().mean() if not s_no_na.empty else 0
            if avg_len > 20: # Arbitrary threshold for "longer strings"
                 return "Free_Text", None
            
            # Fallback for strings
            return "String", None
            
        return "Unknown", None


    def _get_basic_stats(self, series: pd.Series, semantic_type: str) -> Dict[str, Any]:
        """Calculates basic statistics for a column."""
        s_no_na = series.dropna()
        stats = {
            "count": int(s_no_na.count()),
            "null_count": int(series.isnull().sum()),
            "null_ratio": series.isnull().mean() if len(series) > 0 else 0.0,
            "unique_count": s_no_na.nunique(),
            "unique_ratio": s_no_na.nunique() / len(s_no_na) if len(s_no_na) > 0 else 0.0,
        }

        if pd.api.types.is_numeric_dtype(series.dtype) and stats["count"] > 0:
            stats["min_value"] = float(s_no_na.min())
            stats["max_value"] = float(s_no_na.max())
            stats["mean_value"] = float(s_no_na.mean())
            stats["median_value"] = float(s_no_na.median())
            stats["std_dev"] = float(s_no_na.std())
        elif pd.api.types.is_datetime64_any_dtype(series.dtype) and stats["count"] > 0:
            stats["min_value"] = str(s_no_na.min())
            stats["max_value"] = str(s_no_na.max())
        elif (pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype)) and stats["count"] > 0:
            str_lengths = s_no_na.astype(str).str.len()
            stats["min_length"] = int(str_lengths.min())
            stats["max_length"] = int(str_lengths.max())
            stats["avg_length"] = float(str_lengths.mean())
        
        if semantic_type == "Categorical" and stats["count"] > 0:
            top_n = min(5, stats["unique_count"])
            stats["top_categories"] = dict(s_no_na.value_counts().nlargest(top_n))
            # Convert keys to string if they are not (e.g. int categories)
            stats["top_categories"] = {str(k): v for k, v in stats["top_categories"].items()}


        return stats

    def _generate_description(self, col_name: str, semantic_type: str, stats: Dict[str, Any], units: Optional[str]) -> str:
        """Generates a human-readable description for the column."""
        desc_parts = [f"Column '{col_name}'"]
        desc_parts.append(f"is of type '{semantic_type}'")
        if units:
            desc_parts.append(f"with units in '{units}'")
        
        desc_parts.append(f"containing {stats['count']} non-null values ({stats['null_ratio']:.1%} null).")
        desc_parts.append(f"It has {stats['unique_count']} unique values.")

        if stats.get("min_value") is not None and stats.get("max_value") is not None:
            desc_parts.append(f"Values range from {stats['min_value']} to {stats['max_value']}.")
        if stats.get("mean_value") is not None:
            desc_parts.append(f"The average value is {stats['mean_value']:.2f}.")
        if stats.get("top_categories"):
            top_cat_str = ", ".join([f"'{k}' ({v} times)" for k,v in stats["top_categories"].items()])
            desc_parts.append(f"Top categories include: {top_cat_str}.")
        
        return " ".join(desc_parts)

    def _suggest_formatting_rules(self, semantic_type: str, units: Optional[str]) -> Optional[str]:
        """Suggests formatting rules based on semantic type and units."""
        if semantic_type == "Currency":
            return f"currency:{units or 'USD'}:,." # e.g. currency:USD:,.2f for $1,234.56
        if semantic_type == "Percentage":
            return "percentage:.1f" # e.g. 12.3%
        if semantic_type == "Date":
            return "date:%Y-%m-%d"
        if semantic_type == "Timestamp":
            return "datetime:%Y-%m-%d %H:%M:%S"
        if semantic_type == "Integer":
            return "integer:," # e.g. 1,234
        if semantic_type == "Float":
            return "float:,.2f" # e.g. 1,234.56
        return None

    def _generate_profile_summary(self, metadata: ColumnMetadata) -> str:
        """Generates a concise textual summary of the column profile."""
        summary = f"{metadata.cleaned_name} ({metadata.semantic_type}"
        if metadata.units:
            summary += f", {metadata.units}"
        summary += "): "
        
        if metadata.semantic_type == "Constant":
            val = metadata.min_value # For constant, min=max=first_value
            summary += f"All values are '{val}'."
            return summary

        summary += f"{metadata.count} values ({metadata.null_ratio:.0%} null), {metadata.unique_count} unique. "
        
        if metadata.min_value is not None and metadata.max_value is not None:
            summary += f"Range: [{str(metadata.min_value)} .. {str(metadata.max_value)}]. " # Ensure string conversion for display
        
        if metadata.semantic_type == "Categorical" and metadata.top_categories:
            top_cats = list(metadata.top_categories.keys())[:2]
            summary += f"Top: {', '.join(top_cats)}."
        elif metadata.mean_value is not None:
            summary += f"Avg: {metadata.mean_value:.2f}."
            
        return summary.strip()

    def profile_column(self, series: pd.Series, col_name_original: str) -> ColumnMetadata:
        """Profiles a single DataFrame column."""
        logger.debug(f"Profiling column: {col_name_original}")
        
        cleaned_name, inferred_unit_from_name = self._clean_column_name(col_name_original)
        semantic_type, inferred_unit_from_type = self._infer_semantic_type_and_units(series, col_name_original, cleaned_name)
        
        # Prioritize unit from type inference, fallback to name inference
        final_units = inferred_unit_from_type if inferred_unit_from_type is not None else inferred_unit_from_name

        stats = self._get_basic_stats(series, semantic_type)
        description = self._generate_description(cleaned_name, semantic_type, stats, final_units)
        formatting_suggestion = self._suggest_formatting_rules(semantic_type, final_units)

        metadata_dict = {
            "original_name": col_name_original,
            "cleaned_name": cleaned_name,
            "semantic_type": semantic_type,
            "pandas_dtype": str(series.dtype),
            "units": final_units,
            "description": description,
            "formatting_suggestion": formatting_suggestion,
            **stats
        }

        column_meta = ColumnMetadata(**metadata_dict)
        column_meta.profile_summary = self._generate_profile_summary(column_meta)
        
        return column_meta

    def profile_dataframe(self, df: pd.DataFrame) -> Dict[str, ColumnMetadata]:
        """
        Profiles all columns in a DataFrame.

        Args:
            df: The pandas DataFrame to profile.

        Returns:
            A dictionary where keys are original column names and values are
            ColumnMetadata objects.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        logger.info(f"Starting DataFrame profiling for {len(df.columns)} columns.")
        all_metadata: Dict[str, ColumnMetadata] = {}
        for col_name in df.columns:
            try:
                all_metadata[col_name] = self.profile_column(df[col_name], str(col_name)) # Ensure col_name is string
            except Exception as e:
                logger.error(f"Error profiling column '{col_name}': {e}", exc_info=True)
                # Create a basic error metadata entry
                series_for_error = df[col_name] if col_name in df else pd.Series(dtype='object') # Handle if col_name itself is problematic
                all_metadata[col_name] = ColumnMetadata(
                    original_name=str(col_name),
                    cleaned_name=self._clean_column_name(str(col_name))[0],
                    semantic_type="Error",
                    pandas_dtype=str(series_for_error.dtype),
                    description=f"Error during profiling: {e}",
                    count=len(series_for_error) - series_for_error.isnull().sum(),
                    null_count=series_for_error.isnull().sum(),
                    null_ratio=series_for_error.isnull().mean() if len(series_for_error) > 0 else 0.0,
                    unique_count=0, 
                    unique_ratio=0.0,
                    profile_summary=f"{self._clean_column_name(str(col_name))[0]} (Error): Profiling failed."
                )
        logger.info("DataFrame profiling complete.")
        return all_metadata

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("=== Column Profiler Example ===\n")

    # Sample DataFrame
    data = {
        'client_id_val': [101, 102, 103, 104, 105, 101],
        'ClientName': ['ACME Corp', 'Beta Inc', 'Gamma LLC', 'ACME Corp', None, 'Delta Co'],
        'revenue_ytd_usd': [120000.50, 85000.75, 210000.00, 120000.50, 5000.00, 75000.00],
        'YoY_Growth_pct': [0.15, -0.05, 0.22, 0.15, 0.01, 0.10],
        'region_code': ['USA', 'CAN', 'USA', 'USA', 'EUR', 'CAN'],
        'last_interaction_date': ['2023-01-15', '2023-02-20', '2022-12-05', '2023-01-15', '2023-03-01', '2023-02-28'],
        'status_flag': [1, 0, 1, 1, 0, 1],
        'notes_on_client': ['Good client', 'Needs attention', 'High potential', 'Good client', 'New', 'Medium priority'],
        'website_url': ['http://acme.com', 'https://beta.inc', 'http://gamma.llc', 'http://acme.com', 'N/A', 'https://delta.co'],
        'founding_year': [1990, 2005, 2010, 1990, 2022, 2000],
        'employee_count': [500, 150, 300, 500, 20, 200],
        'is_active': [True, False, True, True, False, True],
        'empty_col': [None, None, None, None, None, None],
        'constant_col': ['X', 'X', 'X', 'X', 'X', 'X']
    }
    sample_df = pd.DataFrame(data)
    sample_df['last_interaction_date'] = pd.to_datetime(sample_df['last_interaction_date'])
    
    profiler = ColumnProfiler()
    
    print("Profiling sample DataFrame...\n")
    df_metadata = profiler.profile_dataframe(sample_df)
    
    for col_original_name, meta in df_metadata.items():
        print(f"--- Metadata for Column: {meta.original_name} -> {meta.cleaned_name} ---")
        # For Pydantic v2: print(meta.model_dump_json(indent=2))
        # For Pydantic v1 (or if model_dump_json is not available):
        print(json.dumps(meta.dict(), indent=2, default=str)) # Added default=str for non-serializable types like numpy.int64
        print(f"\n  Profile Summary: {meta.profile_summary}\n")

    print("\n--- Testing with a purely numeric DataFrame ---")
    numeric_data = {
        'colA_val': np.random.rand(10) * 100,
        'colB_items': np.random.randint(0, 10, 10),
        'colC_id': range(10)
    }
    numeric_df = pd.DataFrame(numeric_data)
    numeric_metadata = profiler.profile_dataframe(numeric_df)
    for col_original_name, meta in numeric_metadata.items():
        print(f"--- Metadata for Column: {meta.original_name} -> {meta.cleaned_name} ---")
        print(json.dumps(meta.dict(), indent=2, default=str))
        print(f"\n  Profile Summary: {meta.profile_summary}\n")
