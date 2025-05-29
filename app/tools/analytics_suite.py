"""
Analytics Suite for MCP Server

This module provides advanced visualization and data analysis tools for the MCP server.
It includes tools for summarizing, visualizing, comparing, forecasting, and analyzing
tabular data, as well as generating reports and exporting results.

Usage:
    from app.tools.analytics_suite import register_tools
    register_tools(mcp)
"""

import os
import base64
import logging
import json
import io
import re
import tempfile
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from fastmcp import Context

# Configure logging
logger = logging.getLogger("mcp_server.tools.analytics_suite")

# Ensure output directory exists
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Set default style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
sns.set_context("talk")

# Helper functions
def ensure_dataframe(data: Union[List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    """Convert data to DataFrame if it's not already"""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return pd.DataFrame(data)
    else:
        raise ValueError("Data must be a DataFrame or a list of dictionaries")

def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns in a DataFrame"""
    return df.select_dtypes(include=['number']).columns.tolist()

def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Detect categorical columns in a DataFrame"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect datetime columns in a DataFrame"""
    datetime_cols = []
    for col in df.columns:
        # Check if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        # Check if column has date-like string values
        elif pd.api.types.is_string_dtype(df[col]):
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except:
                pass
    return datetime_cols

def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert detected datetime columns to datetime type"""
    datetime_cols = detect_datetime_columns(df)
    df_copy = df.copy()
    for col in datetime_cols:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col])
        except:
            logger.warning(f"Failed to convert column {col} to datetime")
    return df_copy

def detect_time_series_column(df: pd.DataFrame) -> Optional[str]:
    """Detect a suitable time series column (datetime with regular intervals)"""
    datetime_cols = detect_datetime_columns(df)
    for col in datetime_cols:
        try:
            # Convert to datetime if not already
            dates = pd.to_datetime(df[col])
            # Check if sorted
            if dates.is_monotonic_increasing:
                return col
            # If not sorted, check if it would be regular when sorted
            sorted_dates = dates.sort_values()
            diffs = sorted_dates.diff().dropna()
            # Check if most differences are the same (allowing some irregularity)
            most_common_diff = diffs.value_counts().idxmax()
            if (diffs == most_common_diff).mean() > 0.7:  # 70% of intervals are the same
                return col
        except:
            continue
    return None

def save_figure_to_file(fig: plt.Figure, format: str = 'png') -> str:
    """
    Save a matplotlib figure to a file and return the file path
    
    Args:
        fig: Matplotlib figure
        format: File format (png, pdf, svg, jpg)
        
    Returns:
        Path to the saved file
    """
    # Create a unique filename
    filename = f"chart_{uuid.uuid4().hex}.{format}"
    filepath = output_dir / filename
    
    # Save the figure
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return str(filepath)

def figure_to_base64(fig: plt.Figure, format: str = 'png') -> str:
    """
    Convert a matplotlib figure to a base64-encoded string
    
    Args:
        fig: Matplotlib figure
        format: Image format (png, pdf, svg, jpg)
        
    Returns:
        Base64-encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def dataframe_to_file(df: pd.DataFrame, format: str = 'csv') -> str:
    """
    Save a DataFrame to a file and return the file path
    
    Args:
        df: Pandas DataFrame
        format: File format (csv, xlsx, json)
        
    Returns:
        Path to the saved file
    """
    # Create a unique filename
    filename = f"data_{uuid.uuid4().hex}.{format}"
    filepath = output_dir / filename
    
    # Save the DataFrame
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'xlsx':
        df.to_excel(filepath, index=False)
    elif format == 'json':
        df.to_json(filepath, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(filepath)

def extract_column_from_query(query: str, df: pd.DataFrame) -> List[str]:
    """
    Extract column names from a query string
    
    Args:
        query: Natural language query
        df: DataFrame with column names to match
        
    Returns:
        List of matched column names
    """
    matched_columns = []
    
    # Normalize column names for better matching
    normalized_columns = {col.lower().replace('_', ' '): col for col in df.columns}
    
    # Try to match column names in the query
    query_lower = query.lower()
    for norm_col, orig_col in normalized_columns.items():
        if norm_col in query_lower or orig_col.lower() in query_lower:
            matched_columns.append(orig_col)
    
    return matched_columns

def extract_groups_from_query(query: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[List]]:
    """
    Extract grouping column and values from a query
    
    Args:
        query: Natural language query
        df: DataFrame with column names to match
        
    Returns:
        Tuple of (grouping_column, group_values)
    """
    # Common grouping terms
    grouping_terms = ['region', 'country', 'category', 'type', 'segment', 'group', 'list', 'status']
    
    # Normalize column names
    normalized_columns = {col.lower().replace('_', ' '): col for col in df.columns}
    
    # Find potential grouping columns
    grouping_col = None
    for term in grouping_terms:
        for norm_col, orig_col in normalized_columns.items():
            if term in norm_col:
                grouping_col = orig_col
                break
        if grouping_col:
            break
    
    # If no grouping column found, try to find any categorical column
    if not grouping_col:
        cat_cols = detect_categorical_columns(df)
        if cat_cols:
            grouping_col = cat_cols[0]
    
    # If grouping column found, get unique values
    group_values = None
    if grouping_col and grouping_col in df.columns:
        group_values = df[grouping_col].unique().tolist()
    
    return grouping_col, group_values

def extract_numeric_columns_from_query(query: str, df: pd.DataFrame) -> List[str]:
    """
    Extract numeric column names from a query string
    
    Args:
        query: Natural language query
        df: DataFrame with column names to match
        
    Returns:
        List of matched numeric column names
    """
    # Get all numeric columns
    numeric_cols = detect_numeric_columns(df)
    
    # Match columns mentioned in the query
    matched_columns = extract_column_from_query(query, df[numeric_cols])
    
    # If no matches, use common metric columns or return the first numeric column
    if not matched_columns:
        # Try to find common metric columns
        metric_terms = ['revenue', 'sales', 'profit', 'income', 'value', 'amount', 'price']
        for col in numeric_cols:
            col_lower = col.lower()
            if any(term in col_lower for term in metric_terms):
                matched_columns.append(col)
        
        # If still no matches, use the first numeric column
        if not matched_columns and numeric_cols:
            matched_columns.append(numeric_cols[0])
    
    return matched_columns

def parse_filter_conditions(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parse filter conditions from a natural language query
    
    Args:
        query: Natural language query
        df: DataFrame with column names to match
        
    Returns:
        Dictionary of filter conditions
    """
    filter_conditions = {}
    
    # Common comparison patterns
    patterns = [
        # Greater than
        r'([\w\s]+)\s+(?:greater than|more than|above|over|>)\s+([\d,.]+)',
        # Less than
        r'([\w\s]+)\s+(?:less than|below|under|<)\s+([\d,.]+)',
        # Equal to
        r'([\w\s]+)\s+(?:equal to|equals|is|==|=)\s+([\w\d,.]+)',
        # In region/category
        r'(?:in|for|from)\s+([\w\s]+)\s+([\w\s,]+)'
    ]
    
    # Normalize column names for better matching
    normalized_columns = {col.lower().replace('_', ' ').strip(): col for col in df.columns}
    
    # Apply each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, query.lower())
        for match in matches:
            col_name, value = match.groups()
            col_name = col_name.strip()
            value = value.strip()
            
            # Try to match column name
            matched_col = None
            for norm_col, orig_col in normalized_columns.items():
                if col_name == norm_col or col_name in norm_col:
                    matched_col = orig_col
                    break
            
            if matched_col:
                # Convert value based on column type
                if pd.api.types.is_numeric_dtype(df[matched_col]):
                    try:
                        # Remove commas and convert to number
                        value = float(value.replace(',', ''))
                    except:
                        continue
                
                filter_conditions[matched_col] = value
    
    return filter_conditions

def register_tools(mcp):
    """Register analytics tools with the MCP server"""
    
    @mcp.tool()
    async def summarize_table(
        ctx: Context,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None
    ) -> str:
        """
        Generate a statistical summary of tabular data.
        
        This tool calculates basic statistics (count, mean, median, min, max, etc.)
        for numeric columns in the data.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            columns: Specific columns to summarize (if None, summarize all numeric columns)
            
        Returns:
            Markdown-formatted summary statistics
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available to summarize"
            
            # Select columns to summarize
            if columns:
                # Filter to specified columns that exist
                cols_to_summarize = [col for col in columns if col in df.columns]
                if not cols_to_summarize:
                    return f"None of the specified columns {columns} found in the data"
                df = df[cols_to_summarize]
            
            # Get numeric columns
            numeric_cols = detect_numeric_columns(df)
            if not numeric_cols:
                return "No numeric columns found to summarize"
            
            # Calculate summary statistics
            summary = df[numeric_cols].describe().T
            
            # Add additional statistics
            summary['median'] = df[numeric_cols].median()
            summary['sum'] = df[numeric_cols].sum()
            summary['variance'] = df[numeric_cols].var()
            
            # Reorder columns for better readability
            summary = summary[['count', 'mean', 'median', 'min', 'max', 'sum', 'std', 'variance', '25%', '50%', '75%']]
            
            # Format the summary
            result = "## Summary Statistics\n\n"
            result += summary.to_markdown(floatfmt=".2f")
            
            # Add count of non-numeric columns
            if len(df.columns) > len(numeric_cols):
                non_numeric = len(df.columns) - len(numeric_cols)
                result += f"\n\n*Note: {non_numeric} non-numeric columns were excluded from this summary.*"
            
            return result
            
        except Exception as e:
            logger.error(f"Error summarizing table: {e}")
            return f"Error summarizing table: {str(e)}"
    
    @mcp.tool()
    async def visualize_data(
        ctx: Context,
        data: List[Dict[str, Any]],
        chart_type: str = "bar",
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        include_file: bool = False,
        query: Optional[str] = None
    ) -> str:
        """
        Create a visualization from tabular data.
        
        This tool generates various chart types (bar, line, pie, scatter, etc.)
        from the provided data.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            chart_type: Type of chart to create (bar, line, pie, scatter, heatmap, box, hist)
            x_column: Column to use for x-axis
            y_column: Column to use for y-axis
            group_by: Column to use for grouping/coloring
            title: Chart title
            include_file: Whether to save the chart as a file and include the path
            query: Natural language query to help determine columns
            
        Returns:
            Markdown with embedded chart image and optional file path
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available to visualize"
            
            # Determine columns from query if not specified
            if query and (not x_column or not y_column):
                # Extract columns from query
                matched_columns = extract_column_from_query(query, df)
                
                # Get numeric columns for y-axis
                numeric_cols = extract_numeric_columns_from_query(query, df)
                
                # Set x and y columns based on extracted information
                if not x_column and matched_columns:
                    # Prefer non-numeric column for x-axis if available
                    non_numeric = [col for col in matched_columns if col not in numeric_cols]
                    if non_numeric:
                        x_column = non_numeric[0]
                    else:
                        x_column = matched_columns[0]
                
                if not y_column and numeric_cols:
                    y_column = numeric_cols[0]
            
            # If still no columns specified, make educated guesses
            if not x_column:
                # Try to find a categorical column
                cat_cols = detect_categorical_columns(df)
                if cat_cols:
                    x_column = cat_cols[0]
                else:
                    # Use the first column
                    x_column = df.columns[0]
            
            if not y_column:
                # Try to find a numeric column
                num_cols = detect_numeric_columns(df)
                if num_cols:
                    y_column = num_cols[0]
                else:
                    # Use the second column or the first if only one column
                    y_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            # Validate columns exist
            if x_column not in df.columns:
                return f"Column '{x_column}' not found in data"
            if y_column not in df.columns:
                return f"Column '{y_column}' not found in data"
            if group_by and group_by not in df.columns:
                return f"Group by column '{group_by}' not found in data"
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Normalize chart type
            chart_type = chart_type.lower()
            
            # Generate chart based on type
            if chart_type == "bar":
                if group_by:
                    # Grouped bar chart
                    pivot_df = df.pivot_table(index=x_column, columns=group_by, values=y_column, aggfunc='mean')
                    pivot_df.plot(kind='bar', ax=plt.gca())
                else:
                    # Simple bar chart
                    sns.barplot(x=x_column, y=y_column, data=df)
                
                plt.xticks(rotation=45, ha='right')
                
            elif chart_type == "line":
                if group_by:
                    # Multiple lines
                    for group, group_df in df.groupby(group_by):
                        plt.plot(group_df[x_column], group_df[y_column], label=str(group))
                    plt.legend()
                else:
                    # Single line
                    plt.plot(df[x_column], df[y_column])
                
                # Check if x is datetime and format accordingly
                if pd.api.types.is_datetime64_any_dtype(df[x_column]):
                    plt.gcf().autofmt_xdate()
                else:
                    plt.xticks(rotation=45, ha='right')
                
            elif chart_type == "pie":
                # Aggregate data if needed
                if len(df) > 10:  # Limit pie chart segments
                    value_counts = df[x_column].value_counts().nlargest(9)
                    # Add "Other" category for remaining values
                    if len(value_counts) < len(df[x_column].unique()):
                        other_count = df[x_column].value_counts().sum() - value_counts.sum()
                        value_counts["Other"] = other_count
                    
                    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
                else:
                    plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%')
                
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
            elif chart_type == "scatter":
                if group_by:
                    # Colored scatter plot
                    for group, group_df in df.groupby(group_by):
                        plt.scatter(group_df[x_column], group_df[y_column], label=str(group), alpha=0.7)
                    plt.legend()
                else:
                    # Simple scatter plot
                    plt.scatter(df[x_column], df[y_column], alpha=0.7)
                
            elif chart_type == "heatmap":
                # Create correlation matrix if both columns are numeric
                if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
                    corr_matrix = df[[x_column, y_column]].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                else:
                    # Create a cross-tabulation for categorical data
                    cross_tab = pd.crosstab(df[x_column], df[y_column])
                    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu')
                
            elif chart_type == "box":
                if group_by:
                    # Grouped box plot
                    sns.boxplot(x=group_by, y=y_column, data=df)
                else:
                    # Simple box plot
                    sns.boxplot(y=y_column, data=df)
                
            elif chart_type == "hist":
                # Histogram
                sns.histplot(df[y_column], kde=True)
                
            else:
                return f"Unsupported chart type: {chart_type}"
            
            # Set title and labels
            if title:
                plt.title(title)
            else:
                plt.title(f"{y_column} by {x_column}")
                
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert figure to base64 for embedding
            img_base64 = figure_to_base64(plt.gcf())
            
            # Create result
            result = f"## {title or 'Data Visualization'}\n\n"
            result += f"![Chart](data:image/png;base64,{img_base64})\n\n"
            
            # Save to file if requested
            if include_file:
                file_path = save_figure_to_file(plt.gcf())
                result += f"*Chart saved to: {file_path}*"
            
            return result
            
        except Exception as e:
            logger.error(f"Error visualizing data: {e}")
            return f"Error visualizing data: {str(e)}"
    
    @mcp.tool()
    async def compare_groups(
        ctx: Context,
        data: List[Dict[str, Any]],
        group_column: Optional[str] = None,
        group_values: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        query: Optional[str] = None,
        include_visualization: bool = True
    ) -> str:
        """
        Compare metrics across different groups in the data.
        
        This tool analyzes and compares key metrics between different groups
        (e.g., regions, categories, segments).
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            group_column: Column to use for grouping
            group_values: Specific group values to compare (if None, use top 2 groups)
            metrics: Columns to compare across groups
            query: Natural language query to help determine groups and metrics
            include_visualization: Whether to include visualizations
            
        Returns:
            Markdown-formatted comparison analysis
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available for comparison"
            
            # Determine groups and metrics from query if not specified
            if query and not group_column:
                group_column, detected_groups = extract_groups_from_query(query, df)
                if group_values is None and detected_groups:
                    group_values = detected_groups[:2]  # Use top 2 groups by default
            
            if query and not metrics:
                metrics = extract_numeric_columns_from_query(query, df)
            
            # If still no group column, try to find a suitable one
            if not group_column:
                cat_cols = detect_categorical_columns(df)
                if cat_cols:
                    # Use the categorical column with the fewest unique values
                    group_column = min(cat_cols, key=lambda col: df[col].nunique())
                else:
                    return "No suitable grouping column found in the data"
            
            # Validate group column exists
            if group_column not in df.columns:
                return f"Group column '{group_column}' not found in data"
            
            # Get unique values in the group column
            unique_groups = df[group_column].unique().tolist()
            
            # If no specific groups provided, use the top 2 groups by frequency
            if not group_values or not all(g in unique_groups for g in group_values):
                value_counts = df[group_column].value_counts()
                group_values = value_counts.index[:2].tolist()
            
            # Ensure we have at least 2 groups
            if len(group_values) < 2:
                if len(unique_groups) < 2:
                    return f"Not enough unique values in '{group_column}' for comparison"
                # Add another group
                for group in unique_groups:
                    if group not in group_values:
                        group_values.append(group)
                        break
            
            # Limit to 2 groups for simplicity
            group_values = group_values[:2]
            
            # Filter data to selected groups
            filtered_df = df[df[group_column].isin(group_values)]
            
            if filtered_df.empty:
                return f"No data available for groups: {group_values}"
            
            # If no metrics specified, use all numeric columns
            if not metrics:
                metrics = detect_numeric_columns(filtered_df)
                
                # Remove the group column if it's numeric
                if group_column in metrics:
                    metrics.remove(group_column)
            
            # Validate metrics exist
            metrics = [m for m in metrics if m in filtered_df.columns]
            if not metrics:
                return "No valid metric columns found for comparison"
            
            # Calculate statistics for each group
            group_stats = []
            for group in group_values:
                group_df = filtered_df[filtered_df[group_column] == group]
                stats = {group_column: group, 'count': len(group_df)}
                
                for metric in metrics:
                    if pd.api.types.is_numeric_dtype(filtered_df[metric]):
                        stats[f"{metric}_mean"] = group_df[metric].mean()
                        stats[f"{metric}_median"] = group_df[metric].median()
                        stats[f"{metric}_sum"] = group_df[metric].sum()
                
                group_stats.append(stats)
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(group_stats)
            
            # Calculate differences and percent changes
            diff_rows = []
            for metric in metrics:
                if f"{metric}_mean" in comparison_df.columns:
                    diff_row = {group_column: f"Difference ({metric})"}
                    
                    # Absolute difference
                    diff = comparison_df.iloc[1][f"{metric}_mean"] - comparison_df.iloc[0][f"{metric}_mean"]
                    diff_row[f"{metric}_mean"] = diff
                    
                    # Percent difference
                    if comparison_df.iloc[0][f"{metric}_mean"] != 0:
                        pct_change = (diff / comparison_df.iloc[0][f"{metric}_mean"]) * 100
                        diff_row[f"{metric}_pct"] = f"{pct_change:.2f}%"
                    else:
                        diff_row[f"{metric}_pct"] = "N/A"
                    
                    diff_rows.append(diff_row)
            
            # Format the results
            result = f"## Comparison: {group_values[0]} vs {group_values[1]}\n\n"
            
            # Add summary statistics table
            result += "### Summary Statistics\n\n"
            result += comparison_df.to_markdown(index=False, floatfmt=".2f")
            
            # Add difference table
            if diff_rows:
                result += "\n\n### Differences\n\n"
                diff_df = pd.DataFrame(diff_rows)
                result += diff_df.to_markdown(index=False, floatfmt=".2f")
            
            # Add visualizations if requested
            if include_visualization and metrics:
                result += "\n\n### Visualizations\n\n"
                
                for metric in metrics[:3]:  # Limit to first 3 metrics
                    if pd.api.types.is_numeric_dtype(filtered_df[metric]):
                        # Create bar chart
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x=group_column, y=metric, data=filtered_df)
                        plt.title(f"{metric} by {group_column}")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Convert to base64
                        img_base64 = figure_to_base64(plt.gcf())
                        result += f"#### {metric} Comparison\n\n"
                        result += f"![{metric} Comparison](data:image/png;base64,{img_base64})\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing groups: {e}")
            return f"Error comparing groups: {str(e)}"
    
    @mcp.tool()
    async def analyze_trend(
        ctx: Context,
        data: List[Dict[str, Any]],
        time_column: Optional[str] = None,
        value_column: Optional[str] = None,
        group_column: Optional[str] = None,
        query: Optional[str] = None,
        period: str = "auto"
    ) -> str:
        """
        Analyze trends and patterns in time series data.
        
        This tool detects trends, seasonality, and growth patterns in time-based data.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            time_column: Column containing time/date information
            value_column: Column containing the values to analyze
            group_column: Optional column for grouping (e.g., by client, region)
            query: Natural language query to help determine columns
            period: Time period for seasonal decomposition (auto, day, week, month, quarter, year)
            
        Returns:
            Markdown-formatted trend analysis with visualizations
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available for trend analysis"
            
            # Convert datetime columns
            df = convert_datetime_columns(df)
            
            # Determine columns from query if not specified
            if query and (not time_column or not value_column):
                # Extract columns from query
                matched_columns = extract_column_from_query(query, df)
                
                # Get numeric columns for value
                numeric_cols = extract_numeric_columns_from_query(query, df)
                
                # Try to find time column
                if not time_column:
                    time_col = detect_time_series_column(df)
                    if time_col:
                        time_column = time_col
                    else:
                        # Look for date/time related columns in matched columns
                        for col in matched_columns:
                            if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day']):
                                time_column = col
                                break
                
                # Set value column based on extracted information
                if not value_column and numeric_cols:
                    value_column = numeric_cols[0]
            
            # If still no columns specified, make educated guesses
            if not time_column:
                time_column = detect_time_series_column(df)
                if not time_column:
                    # Try to find a column with date/time in the name
                    for col in df.columns:
                        if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day']):
                            time_column = col
                            break
                    
                    if not time_column and len(df.columns) > 0:
                        # Use the first column as a last resort
                        time_column = df.columns[0]
            
            if not value_column:
                # Try to find a numeric column
                num_cols = detect_numeric_columns(df)
                if num_cols:
                    # Prefer columns with value-related names
                    for col in num_cols:
                        if any(term in col.lower() for term in ['value', 'amount', 'revenue', 'sales', 'profit']):
                            value_column = col
                            break
                    
                    if not value_column:
                        # Use the first numeric column
                        value_column = num_cols[0]
                else:
                    # Use the second column or the first if only one column
                    value_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            # Validate columns exist
            if time_column not in df.columns:
                return f"Time column '{time_column}' not found in data"
            if value_column not in df.columns:
                return f"Value column '{value_column}' not found in data"
            if group_column and group_column not in df.columns:
                return f"Group column '{group_column}' not found in data"
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                try:
                    df[time_column] = pd.to_datetime(df[time_column])
                except:
                    return f"Could not convert '{time_column}' to datetime format"
            
            # Ensure value column is numeric
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                try:
                    df[value_column] = pd.to_numeric(df[value_column])
                except:
                    return f"Could not convert '{value_column}' to numeric format"
            
            # Sort by time
            df = df.sort_values(by=time_column)
            
            # Initialize result
            result = f"## Trend Analysis: {value_column} over Time\n\n"
            
            # Analyze overall trend
            if group_column:
                # Analyze trends by group
                groups = df[group_column].unique()
                
                # Create trend visualization
                plt.figure(figsize=(12, 6))
                
                for group in groups:
                    group_df = df[df[group_column] == group]
                    plt.plot(group_df[time_column], group_df[value_column], label=str(group))
                
                plt.title(f"{value_column} Trends by {group_column}")
                plt.xlabel(time_column)
                plt.ylabel(value_column)
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Convert to base64
                img_base64 = figure_to_base64(plt.gcf())
                result += f"### Trend Visualization\n\n"
                result += f"![Trend Analysis](data:image/png;base64,{img_base64})\n\n"
                
                # Calculate growth rates by group
                result += "### Growth Analysis by Group\n\n"
                
                growth_data = []
                for group in groups:
                    group_df = df[df[group_column] == group]
                    
                    if len(group_df) < 2:
                        continue
                    
                    # Calculate overall growth
                    first_value = group_df[value_column].iloc[0]
                    last_value = group_df[value_column].iloc[-1]
                    
                    if first_value != 0:
                        overall_growth = ((last_value - first_value) / first_value) * 100
                    else:
                        overall_growth = float('inf') if last_value > 0 else 0
                    
                    # Calculate average growth rate
                    pct_changes = group_df[value_column].pct_change().dropna()
                    avg_growth_rate = pct_changes.mean() * 100
                    
                    growth_data.append({
                        group_column: group,
                        'Start Value': first_value,
                        'End Value': last_value,
                        'Overall Growth (%)': overall_growth,
                        'Avg Growth Rate (%)': avg_growth_rate
                    })
                
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    result += growth_df.to_markdown(index=False, floatfmt=".2f")
                else:
                    result += "Insufficient data to calculate growth rates by group.\n\n"
            else:
                # Analyze overall trend without grouping
                
                # Create trend visualization
                plt.figure(figsize=(12, 6))
                plt.plot(df[time_column], df[value_column])
                plt.title(f"{value_column} Trend Over Time")
                plt.xlabel(time_column)
                plt.ylabel(value_column)
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Convert to base64
                img_base64 = figure_to_base64(plt.gcf())
                result += f"### Trend Visualization\n\n"
                result += f"![Trend Analysis](data:image/png;base64,{img_base64})\n\n"
                
                # Calculate overall statistics
                result += "### Growth Analysis\n\n"
                
                if len(df) >= 2:
                    # Calculate overall growth
                    first_value = df[value_column].iloc[0]
                    last_value = df[value_column].iloc[-1]
                    first_date = df[time_column].iloc[0]
                    last_date = df[time_column].iloc[-1]
                    
                    result += f"- **Time Period**: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}\n"
                    result += f"- **Starting Value**: {first_value:.2f}\n"
                    result += f"- **Ending Value**: {last_value:.2f}\n"
                    
                    if first_value != 0:
                        overall_growth = ((last_value - first_value) / first_value) * 100
                        result += f"- **Overall Growth**: {overall_growth:.2f}%\n"
                    
                    # Calculate average growth rate
                    pct_changes = df[value_column].pct_change().dropna()
                    avg_growth_rate = pct_changes.mean() * 100
                    result += f"- **Average Growth Rate**: {avg_growth_rate:.2f}% per period\n"
                    
                    # Determine if trend is increasing, decreasing, or flat
                    if overall_growth > 5:
                        trend = "increasing"
                    elif overall_growth < -5:
                        trend = "decreasing"
                    else:
                        trend = "relatively flat"
                    
                    result += f"- **Overall Trend**: {trend.capitalize()}\n\n"
                else:
                    result += "Insufficient data to calculate growth rates.\n\n"
                
                # Try to perform seasonal decomposition if enough data points
                if len(df) >= 6:  # Need at least 6 data points for decomposition
                    try:
                        # Set the index to the time column
                        ts_df = df.set_index(time_column)
                        
                        # Determine period for decomposition
                        if period == "auto":
                            # Try to infer period from data
                            time_diffs = df[time_column].diff().dropna()
                            if len(time_diffs) > 0:
                                most_common_diff = time_diffs.value_counts().idxmax()
                                
                                if most_common_diff.days <= 1:  # Daily data
                                    period_value = 7  # Weekly seasonality
                                elif most_common_diff.days <= 7:  # Weekly data
                                    period_value = 4  # Monthly seasonality
                                elif most_common_diff.days <= 31:  # Monthly data
                                    period_value = 12  # Yearly seasonality
                                else:  # Yearly or longer
                                    period_value = 4  # Quarterly seasonality
                            else:
                                period_value = 7  # Default to weekly
                        else:
                            # Map period string to numeric value
                            period_map = {
                                "day": 7,      # Weekly seasonality for daily data
                                "week": 4,      # Monthly seasonality for weekly data
                                "month": 12,    # Yearly seasonality for monthly data
                                "quarter": 4,   # Yearly seasonality for quarterly data
                                "year": 4       # 4-year cycle for yearly data
                            }
                            period_value = period_map.get(period.lower(), 7)
                        
                        # Perform decomposition if we have enough periods
                        if len(ts_df) >= period_value * 2:
                            decomposition = seasonal_decompose(
                                ts_df[value_column],
                                model='additive',
                                period=period_value
                            )
                            
                            # Plot decomposition
                            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                            
                            # Original
                            axes[0].plot(decomposition.observed)
                            axes[0].set_title('Original Time Series')
                            
                            # Trend
                            axes[1].plot(decomposition.trend)
                            axes[1].set_title('Trend Component')
                            
                            # Seasonal
                            axes[2].plot(decomposition.seasonal)
                            axes[2].set_title('Seasonal Component')
                            
                            # Residual
                            axes[3].plot(decomposition.resid)
                            axes[3].set_title('Residual Component')
                            
                            plt.tight_layout()
                            
                            # Convert to base64
                            img_base64 = figure_to_base64(plt.gcf())
                            result += f"### Seasonal Decomposition\n\n"
                            result += f"![Seasonal Decomposition](data:image/png;base64,{img_base64})\n\n"
                            
                            # Interpret seasonality
                            seasonal_strength = decomposition.seasonal.abs().mean() / decomposition.observed.std()
                            if seasonal_strength > 0.3:
                                result += f"The data shows **strong seasonality** with a period of {period_value} time units.\n\n"
                            elif seasonal_strength > 0.1:
                                result += f"The data shows **moderate seasonality** with a period of {period_value} time units.\n\n"
                            else:
                                result += f"The data shows **weak or no seasonality** with the tested period of {period_value} time units.\n\n"
                    except Exception as decomp_error:
                        logger.warning(f"Could not perform seasonal decomposition: {decomp_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return f"Error analyzing trend: {str(e)}"
    
    @mcp.tool()
    async def detect_anomalies(
        ctx: Context,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        contamination: float = 0.05,
        query: Optional[str] = None,
        include_visualization: bool = True
    ) -> str:
        """
        Detect anomalies and outliers in the data.
        
        This tool identifies unusual patterns, outliers, or sudden changes in the data.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            columns: Specific columns to check for anomalies
            time_column: Column containing time/date information for time-based analysis
            contamination: Expected proportion of outliers in the data (0.01 to 0.1)
            query: Natural language query to help determine columns
            include_visualization: Whether to include visualizations
            
        Returns:
            Markdown-formatted anomaly analysis with visualizations
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available for anomaly detection"
            
            # Convert datetime columns
            df = convert_datetime_columns(df)
            
            # Determine columns from query if not specified
            if query and not columns:
                # Extract columns from query
                matched_columns = extract_column_from_query(query, df)
                
                # Get numeric columns
                numeric_cols = extract_numeric_columns_from_query(query, df)
                
                if numeric_cols:
                    columns = numeric_cols
                
                # Try to find time column if not specified
                if not time_column:
                    time_col = detect_time_series_column(df)
                    if time_col:
                        time_column = time_col
                    else:
                        # Look for date/time related columns in matched columns
                        for col in matched_columns:
                            if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day']):
                                time_column = col
                                break
            
            # If still no columns specified, use all numeric columns
            if not columns:
                columns = detect_numeric_columns(df)
            
            # Validate columns exist and are numeric
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not columns:
                return "No valid numeric columns found for anomaly detection"
            
            # Initialize result
            result = "## Anomaly Detection Results\n\n"
            
            # Validate time column if specified
            if time_column:
                if time_column not in df.columns:
                    time_column = None
                else:
                    # Ensure time column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                        try:
                            df[time_column] = pd.to_datetime(df[time_column])
                        except:
                            time_column = None
            
            # Sort by time if time column is available
            if time_column:
                df = df.sort_values(by=time_column)
            
            # Validate contamination parameter
            contamination = max(0.01, min(0.1, contamination))
            
            # Detect anomalies for each column
            anomaly_results = []
            
            for col in columns:
                # Skip columns with too many missing values
                if df[col].isna().sum() > len(df) * 0.5:
                    continue
                
                # Create a copy of the data for this column
                col_df = df[[col]].copy()
                col_df = col_df.dropna()
                
                if len(col_df) < 10:
                    # Skip columns with too few data points
                    continue
                
                # Reshape for Isolation Forest
                X = col_df.values.reshape(-1, 1)
                
                # Train Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                
                # Fit and predict
                predictions = iso_forest.fit_predict(X)
                
                # Convert predictions to anomaly flags (-1 for anomalies, 1 for normal)
                anomalies = predictions == -1
                
                # Get anomaly indices and values
                anomaly_indices = np.where(anomalies)[0]
                anomaly_values = col_df.iloc[anomaly_indices][col].values
                
                # Calculate Z-scores for anomaly values
                mean = col_df[col].mean()
                std = col_df[col].std()
                z_scores = [(val - mean) / std if std != 0 else 0 for val in anomaly_values]
                
                # Create anomaly result
                if len(anomaly_indices) > 0:
                    # Create result entry
                    anomaly_result = {
                        "column": col,
                        "anomaly_count": len(anomaly_indices),
                        "anomaly_percent": (len(anomaly_indices) / len(col_df)) * 100,
                        "indices": anomaly_indices.tolist(),
                        "values": anomaly_values.tolist(),
                        "z_scores": z_scores
                    }
                    
                    # Add time information if available
                    if time_column:
                        anomaly_result["times"] = df.iloc[anomaly_indices][time_column].tolist()
                    
                    anomaly_results.append(anomaly_result)
            
            # Summarize results
            if not anomaly_results:
                result += "No significant anomalies detected in the data.\n\n"
            else:
                result += f"Detected anomalies in {len(anomaly_results)} columns:\n\n"
                
                # Create summary table
                summary_data = []
                for res in anomaly_results:
                    summary_data.append({
                        "Column": res["column"],
                        "Anomaly Count": res["anomaly_count"],
                        "Anomaly %": f"{res['anomaly_percent']:.2f}%",
                        "Min Z-Score": f"{min(res['z_scores']):.2f}",
                        "Max Z-Score": f"{max(res['z_scores']):.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                result += summary_df.to_markdown(index=False)
                
                # Add detailed results for each column
                result += "\n\n### Detailed Anomaly Analysis\n\n"
                
                for res in anomaly_results:
                    col = res["column"]
                    result += f"#### Anomalies in '{col}'\n\n"
                    
                    # Create table of anomalies
                    anomaly_data = []
                    for i in range(len(res["indices"])):
                        anomaly_entry = {
                            "Index": res["indices"][i],
                            "Value": res["values"][i],
                            "Z-Score": f"{res['z_scores'][i]:.2f}"
                        }
                        
                        # Add time if available
                        if "times" in res:
                            time_val = res["times"][i]
                            if pd.api.types.is_datetime64_any_dtype(time_val):
                                time_val = time_val.strftime("%Y-%m-%d %H:%M:%S")
                            anomaly_entry["Time"] = time_val
                        
                        anomaly_data.append(anomaly_entry)
                    
                    # Create and display table
                    anomaly_df = pd.DataFrame(anomaly_data)
                    result += anomaly_df.to_markdown(index=False)
                    
                    # Add visualization if requested
                    if include_visualization:
                        plt.figure(figsize=(12, 6))
                        
                        # Plot the data
                        if time_column:
                            # Time series plot
                            plt.plot(df[time_column], df[col], 'b-', label='Normal')
                            
                            # Highlight anomalies
                            anomaly_times = [res["times"][i] for i in range(len(res["indices"]))]
                            anomaly_values = res["values"]
                            plt.scatter(anomaly_times, anomaly_values, color='red', s=50, label='Anomaly')
                            
                            plt.xlabel(time_column)
                            plt.xticks(rotation=45)
                        else:
                            # Regular plot
                            plt.plot(df.index, df[col], 'b-', label='Normal')
                            
                            # Highlight anomalies
                            plt.scatter(res["indices"], res["values"], color='red', s=50, label='Anomaly')
                            
                            plt.xlabel("Index")
                        
                        plt.ylabel(col)
                        plt.title(f"Anomalies in {col}")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        # Convert to base64
                        img_base64 = figure_to_base64(plt.gcf())
                        result += f"\n\n![Anomalies in {col}](data:image/png;base64,{img_base64})\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return f"Error detecting anomalies: {str(e)}"
    
    @mcp.tool()
    async def cluster_clients(
        ctx: Context,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        n_clusters: int = 3,
        id_column: Optional[str] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Cluster similar clients based on their characteristics.
        
        This tool groups clients or entities with similar attributes using K-means clustering.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            columns: Specific columns to use for clustering
            n_clusters: Number of clusters to create
            id_column: Column containing client/entity identifiers
            query: Natural language query to help determine columns
            
        Returns:
            Markdown-formatted clustering analysis with visualizations
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available for clustering"
            
            # Determine columns from query if not specified
            if query and not columns:
                # Extract columns from query
                numeric_cols = extract_numeric_columns_from_query(query, df)
                
                if numeric_cols:
                    columns = numeric_cols
            
            # If still no columns specified, use all numeric columns
            if not columns:
                columns = detect_numeric_columns(df)
            
            # Validate columns exist and are numeric
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not columns:
                return "No valid numeric columns found for clustering"
            
            # Determine ID column if not specified
            if not id_column:
                # Look for common ID column names
                id_patterns = ['id', 'name', 'client', 'customer', 'entity']
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in id_patterns):
                        id_column = col
                        break
                
                # If still not found, use the first non-numeric column
                if not id_column:
                    non_numeric = [col for col in df.columns if col not in columns]
                    if non_numeric:
                        id_column = non_numeric[0]
            
            # If still no ID column, use index
            if not id_column or id_column not in df.columns:
                df['id'] = [f"Entity_{i}" for i in range(len(df))]
                id_column = 'id'
            
            # Prepare data for clustering
            X = df[columns].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters if not specified
            if n_clusters <= 0 or n_clusters > min(10, len(df) // 2):
                # Try different numbers of clusters
                silhouette_scores = []
                cluster_range = range(2, min(10, len(df) // 2) + 1)
                
                for k in cluster_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Skip if only one cluster has data points
                    if len(np.unique(cluster_labels)) < k:
                        silhouette_scores.append(-1)
                        continue
                    
                    # Calculate silhouette score
                    score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(score)
                
                # Find the best number of clusters
                if silhouette_scores:
                    best_k_idx = np.argmax(silhouette_scores)
                    n_clusters = cluster_range[best_k_idx]
                else:
                    n_clusters = 3  # Default
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to the DataFrame
            df['cluster'] = cluster_labels
            
            # Initialize result
            result = f"## Client Clustering Analysis\n\n"
            result += f"Clustered {len(df)} clients into {n_clusters} groups based on {len(columns)} features.\n\n"
            
            # Summarize clusters
            result += "### Cluster Summary\n\n"
            
            cluster_summary = df.groupby('cluster').size().reset_index(name='count')
            cluster_summary['percentage'] = (cluster_summary['count'] / len(df)) * 100
            
            # Add to result
            result += cluster_summary.to_markdown(index=False, floatfmt=".2f")
            
            # Calculate cluster characteristics
            result += "\n\n### Cluster Characteristics\n\n"
            
            # Calculate mean values for each feature in each cluster
            cluster_means = df.groupby('cluster')[columns].mean().reset_index()
            
            # Add to result
            result += cluster_means.to_markdown(index=False, floatfmt=".2f")
            
            # Create visualizations
            
            # 1. PCA for dimensionality reduction
            if len(columns) > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Create scatter plot
                plt.figure(figsize=(10, 8))
                
                for i in range(n_clusters):
                    plt.scatter(
                        X_pca[cluster_labels == i, 0],
                        X_pca[cluster_labels == i, 1],
                        s=50, label=f'Cluster {i}'
                    )
                
                plt.title('Client Clusters (PCA)')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Convert to base64
                img_base64 = figure_to_base64(plt.gcf())
                result += f"\n\n### Cluster Visualization (PCA)\n\n"
                result += f"![Cluster PCA](data:image/png;base64,{img_base64})\n\n"
                
                # Explained variance
                explained_variance = pca.explained_variance_ratio_
                result += f"*Note: PCA components explain {sum(explained_variance) * 100:.2f}% of the total variance.*\n\n"
            
            # 2. Feature importance visualization
            if len(columns) > 1:
                # Calculate standardized cluster centers
                centers = kmeans.cluster_centers_
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(centers, annot=True, fmt=".2f", cmap="YlGnBu",
                            xticklabels=columns, yticklabels=[f"Cluster {i}" for i in range(n_clusters)])
                plt.title('Cluster Centers (Standardized Features)')
                plt.tight_layout()
                
                # Convert to base64
                img_base64 = figure_to_base64(plt.gcf())
                result += f"### Cluster Feature Importance\n\n"
                result += f"![Cluster Features](data:image/png;base64,{img_base64})\n\n"
            
            # 3. Sample clients from each cluster
            result += "### Sample Clients from Each Cluster\n\n"
            
            for i in range(n_clusters):
                result += f"#### Cluster {i} Samples\n\n"
                
                # Get sample clients from this cluster
                cluster_df = df[df['cluster'] == i].head(5)
                
                # Select relevant columns for display
                display_cols = [id_column] + columns[:5]  # Limit to first 5 feature columns
                
                # Display sample
                result += cluster_df[display_cols].to_markdown(index=False, floatfmt=".2f")
                result += "\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error clustering clients: {e}")
            return f"Error clustering clients: {str(e)}"
    
    @mcp.tool()
    async def correlation_matrix(
        ctx: Context,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        query: Optional[str] = None,
        method: str = "pearson"
    ) -> str:
        """
        Calculate and visualize correlations between variables.
        
        This tool shows how different metrics are related to each other.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            columns: Specific columns to include in the correlation analysis
            query: Natural language query to help determine columns
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Markdown-formatted correlation analysis with visualizations
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available for correlation analysis"
            
            # Determine columns from query if not specified
            if query and not columns:
                # Extract columns from query
                numeric_cols = extract_numeric_columns_from_query(query, df)
                
                if numeric_cols:
                    columns = numeric_cols
            
            # If still no columns specified, use all numeric columns
            if not columns:
                columns = detect_numeric_columns(df)
            
            # Validate columns exist and are numeric
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not columns:
                return "No valid numeric columns found for correlation analysis"
            
            if len(columns) < 2:
                return "At least two numeric columns are required for correlation analysis"
            
            # Limit to first 10 columns for readability
            if len(columns) > 10:
                columns = columns[:10]
                
            # Validate correlation method
            valid_methods = ["pearson", "spearman", "kendall"]
            if method.lower() not in valid_methods:
                method = "pearson"
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr(method=method.lower())
            
            # Initialize result
            result = f"## Correlation Analysis ({method.capitalize()} Method)\n\n"
            
            # Create heatmap visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                        linewidths=.5, fmt='.2f')
            plt.title(f'Correlation Matrix ({method.capitalize()})')
            plt.tight_layout()
            
            # Convert to base64
            img_base64 = figure_to_base64(plt.gcf())
            result += f"![Correlation Matrix](data:image/png;base64,{img_base64})\n\n"
            
            # Add correlation table
            result += "### Correlation Table\n\n"
            result += corr_matrix.to_markdown(floatfmt=".2f")
            
            # Identify strong correlations
            strong_pos_corr = []
            strong_neg_corr = []
            
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= 0.7:
                        corr_pair = {
                            "Variable 1": columns[i],
                            "Variable 2": columns[j],
                            "Correlation": corr_value
                        }
                        
                        if corr_value >= 0.7:
                            strong_pos_corr.append(corr_pair)
                        elif corr_value <= -0.7:
                            strong_neg_corr.append(corr_pair)
            
            # Add strong correlations to result
            if strong_pos_corr or strong_neg_corr:
                result += "\n\n### Strong Correlations\n\n"
                
                if strong_pos_corr:
                    result += "#### Strong Positive Correlations\n\n"
                    pos_df = pd.DataFrame(strong_pos_corr)
                    result += pos_df.to_markdown(index=False, floatfmt=".2f")
                    result += "\n\n"
                
                if strong_neg_corr:
                    result += "#### Strong Negative Correlations\n\n"
                    neg_df = pd.DataFrame(strong_neg_corr)
                    result += neg_df.to_markdown(index=False, floatfmt=".2f")
                    result += "\n\n"
            
            # Add interpretation guide
            result += "### Interpretation Guide\n\n"
            result += "- **1.0**: Perfect positive correlation\n"
            result += "- **0.7 to 1.0**: Strong positive correlation\n"
            result += "- **0.3 to 0.7**: Moderate positive correlation\n"
            result += "- **0 to 0.3**: Weak positive correlation\n"
            result += "- **0**: No correlation\n"
            result += "- **-0.3 to 0**: Weak negative correlation\n"
            result += "- **-0.7 to -0.3**: Moderate negative correlation\n"
            result += "- **-1.0 to -0.7**: Strong negative correlation\n"
            result += "- **-1.0**: Perfect negative correlation\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return f"Error calculating correlation matrix: {str(e)}"
    
    @mcp.tool()
    async def query_table_qa(
        ctx: Context,
        data: List[Dict[str, Any]],
        query: str
    ) -> str:
        """
        Answer natural language questions about tabular data.
        
        This tool interprets questions about the data and provides answers
        by analyzing the underlying tables.
        
        Args:
            ctx: The MCP context
            data: Tabular data as a list of dictionaries
            query: Natural language question about the data
            
        Returns:
            Markdown-formatted answer to the question
        """
        try:
            # Convert to DataFrame
            df = ensure_dataframe(data)
            
            if df.empty:
                return "No data available to answer questions"
            
            # Initialize result
            result = f"## Data Query: {query}\n\n"
            
            # Process different types of questions
            query_lower = query.lower()
            
            # 1. Questions about maximum/minimum values
            max_pattern = r"(?:what|which|who).*(?:highest|maximum|max|most|largest|greatest|top).*(?:in|of|for)?\s+(.+?)(?:\?|$)"
            min_pattern = r"(?:what|which|who).*(?:lowest|minimum|min|least|smallest|bottom).*(?:in|of|for)?\s+(.+?)(?:\?|$)"
            
            max_match = re.search(max_pattern, query_lower)
            min_match = re.search(min_pattern, query_lower)
            
            if max_match or min_match:
                # Extract the column name from the query
                if max_match:
                    col_hint = max_match.group(1).strip()
                    find_max = True
                else:
                    col_hint = min_match.group(1).strip()
                    find_max = False
                
                # Try to find matching column
                matched_cols = []
                for col in df.columns:
                    if col.lower() in col_hint or col.lower().replace('_', ' ') in col_hint:
                        matched_cols.append(col)
                
                if not matched_cols:
                    # Try to find any numeric column that might match
                    numeric_cols = detect_numeric_columns(df)
                    for col in numeric_cols:
                        if any(term in col_hint for term in col.lower().split('_')):
                            matched_cols.append(col)
                
                if matched_cols:
                    target_col = matched_cols[0]
                    
                    # Find ID column to identify the entity
                    id_col = None
                    id_patterns = ['name', 'id', 'client', 'customer', 'entity']
                    for col in df.columns:
                        if any(pattern in col.lower() for pattern in id_patterns):
                            id_col = col
                            break
                    
                    if not id_col and len(df.columns) > 0:
                        id_col = df.columns[0]
                    
                    # Find max/min value
                    if find_max:
                        if pd.api.types.is_numeric_dtype(df[target_col]):
                            idx = df[target_col].idxmax()
                        else:
                            idx = df[target_col].astype(str).str.len().idxmax()
                    else:
                        if pd.api.types.is_numeric_dtype(df[target_col]):
                            idx = df[target_col].idxmin()
                        else:
                            idx = df[target_col].astype(str).str.len().idxmin()
                    
                    # Get the row
                    row = df.loc[idx]
                    
                    # Format the answer
                    if id_col:
                        entity_name = row[id_col]
                        value = row[target_col]
                        
                        if find_max:
                            result += f"The highest {target_col} is {value}, which belongs to {entity_name}.\n\n"
                        else:
                            result += f"The lowest {target_col} is {value}, which belongs to {entity_name}.\n\n"
                    else:
                        value = row[target_col]
                        
                        if find_max:
                            result += f"The highest {target_col} is {value}.\n\n"
                        else:
                            result += f"The lowest {target_col} is {value}.\n\n"
                    
                    # Add the row data
                    result += "### Detailed Information\n\n"
                    row_df = pd.DataFrame([row])
                    result += row_df.to_markdown(index=False, floatfmt=".2f")
                    
                    return result
            
            # 2. Questions about counting or how many
            count_pattern = r"(?:how many|count|number of).*(?:in|with|where)?\s+(.+?)(?:\?|$)"
            count_match = re.search(count_pattern, query_lower)
            
            if count_match:
                condition_hint = count_match.group(1).strip()
                
                # Try to parse filter conditions
                filter_conditions = parse_filter_conditions(query, df)
                
                if filter_conditions:
                    # Apply filters
                    filtered_df = df.copy()
                    for col, value in filter_conditions.items():
                        filtered_df = filtered_df[filtered_df[col] == value]
                    
                    count = len(filtered_df)
                    
                    condition_text = ", ".join([f"{col} = {value}" for col, value in filter_conditions.items()])
                    result += f"There are **{count}** records where {condition_text}.\n\n"
                    
                    if count > 0 and count <= 5:
                        result += "### Sample Records\n\n"
                        result += filtered_df.head().to_markdown(index=False, floatfmt=".2f")
                    
                    return result
                else:
                    # Try to match a column name
                    matched_cols = []
                    for col in df.columns:
                        if col.lower() in condition_hint or col.lower().replace('_', ' ') in condition_hint:
                            matched_cols.append(col)
                    
                    if matched_cols:
                        col = matched_cols[0]
                        unique_count = df[col].nunique()
                        total_count = len(df)
                        
                        result += f"There are **{unique_count}** unique values in the '{col}' column out of {total_count} total records.\n\n"
                        
                        # Show value counts for categorical columns
                        if df[col].nunique() <= 10 or not pd.api.types.is_numeric_dtype(df[col]):
                            result += "### Value Distribution\n\n"
                            value_counts = df[col].value_counts().reset_index()
                            value_counts.columns = [col, 'Count']
                            result += value_counts.to_markdown(index=False)
                        
                        return result
            
            # 3. Questions about averages or means
            avg_pattern = r"(?:what is|what's|find|calculate).*(?:average|mean|median).*(?:of|for)?\s+(.+?)(?:\?|$)"
            avg_match = re.search(avg_pattern, query_lower)
            
            if avg_match:
                col_hint = avg_match.group(1).strip()
                
                # Try to find matching column
                matched_cols = []
                for col in df.columns:
                    if col.lower() in col_hint or col.lower().replace('_', ' ') in col_hint:
                        matched_cols.append(col)
                
                if not matched_cols:
                    # Try to find any numeric column that might match
                    numeric_cols = detect_numeric_columns