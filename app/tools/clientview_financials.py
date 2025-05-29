import logging
import os
import pandas as pd
from fastmcp import Context
from typing import Optional, Dict, Any, List, Type, ClassVar
from enum import Enum, IntEnum
from pydantic import BaseModel, Field, field_validator, ConfigDict
import requests
import json
from datetime import datetime

from app.registry.tools import register_tool
from app.utils.tool_wrappers import llm_enhance_wrapper

logger = logging.getLogger('mcp_server.tools.clientview_financials')

# Base URL for ClientView endpoints. Default to local mock server.
BASE_URL = os.getenv("CLIENTVIEW_BASE_URL", "http://localhost:8001")

###################
# COLUMN METADATA REGISTRY
###################

class DataType:
    """Standard data types for column metadata"""
    STRING = "string"
    INTEGER = "integer"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    BOOLEAN = "boolean"

class Category:
    """Standard categories for column organization"""
    IDENTIFICATION = "identification"
    DEMOGRAPHICS = "demographics"
    REVENUE_CURRENT = "revenue_current"
    REVENUE_PREVIOUS = "revenue_previous"
    REVENUE_HISTORICAL = "revenue_historical"
    RANKING = "ranking"
    INTERACTIONS_CURRENT = "interactions_current"
    INTERACTIONS_PREVIOUS = "interactions_previous"
    METADATA = "metadata"

# Centralized Column Metadata Registry
COLUMN_METADATA_REGISTRY = {
    # Identification Fields
    'ClientName': {
        'display_name': 'Client Name',
        'data_type': DataType.STRING,
        'description': 'Full legal name of the client entity',
        'category': Category.IDENTIFICATION,
        'required': True,
        'sortable': True
    },
    'ClientCDRID': {
        'display_name': 'Client CDRID',
        'data_type': DataType.INTEGER,
        'description': 'Unique client identifier in the system',
        'category': Category.IDENTIFICATION,
        'required': True,
        'sortable': True
    },
    'GID': {
        'display_name': 'GID',
        'data_type': DataType.STRING,
        'description': 'Global identifier for the client',
        'category': Category.IDENTIFICATION,
        'required': False,
        'sortable': False
    },
    
    # Demographics Fields
    'RegionName': {
        'display_name': 'Region',
        'data_type': DataType.STRING,
        'description': 'Geographic region where client is located',
        'category': Category.DEMOGRAPHICS,
        'required': False,
        'sortable': True,
        'filter_values': ['CAN', 'USA', 'EUR', 'APAC', 'LATAM', 'OTHER']
    },
    'FocusList': {
        'display_name': 'Focus List',
        'data_type': DataType.STRING,
        'description': 'Strategic client lists (Focus40, FS30, Corp100)',
        'category': Category.DEMOGRAPHICS,
        'required': False,
        'sortable': True,
        'filter_values': ['Focus40', 'FS30', 'Corp100']
    },
    'HierarchyDepth': {
        'display_name': 'Hierarchy Depth',
        'data_type': DataType.INTEGER,
        'description': 'Level in the client organizational hierarchy',
        'category': Category.DEMOGRAPHICS,
        'required': False,
        'sortable': True
    },
    
    # Current Revenue Fields
    'RevenueYTD': {
        'display_name': 'Revenue YTD',
        'data_type': DataType.CURRENCY,
        'description': 'Year-to-date revenue from this client',
        'category': Category.REVENUE_CURRENT,
        'required': True,
        'sortable': True,
        'primary_metric': True
    },
    
    # Previous Revenue Fields
    'RevenuePrevYTD': {
        'display_name': 'Revenue Prev YTD',
        'data_type': DataType.CURRENCY,
        'description': 'Previous year-to-date revenue for comparison',
        'category': Category.REVENUE_PREVIOUS,
        'required': False,
        'sortable': True
    },
    'RevenuePrevYear': {
        'display_name': 'Revenue Prev Year',
        'data_type': DataType.CURRENCY,
        'description': 'Full previous year revenue',
        'category': Category.REVENUE_PREVIOUS,
        'required': False,
        'sortable': True
    },
    
    # Historical Revenue Fields (5-year trend)
    'RevenueY0': {
        'display_name': 'Revenue Y0',
        'data_type': DataType.CURRENCY,
        'description': 'Revenue for current year (Y0)',
        'category': Category.REVENUE_HISTORICAL,
        'required': False,
        'sortable': True
    },
    'RevenueY1': {
        'display_name': 'Revenue Y1',
        'data_type': DataType.CURRENCY,
        'description': 'Revenue for previous year (Y-1)',
        'category': Category.REVENUE_HISTORICAL,
        'required': False,
        'sortable': True
    },
    'RevenueY2': {
        'display_name': 'Revenue Y2',
        'data_type': DataType.CURRENCY,
        'description': 'Revenue for 2 years ago (Y-2)',
        'category': Category.REVENUE_HISTORICAL,
        'required': False,
        'sortable': True
    },
    'RevenueY3': {
        'display_name': 'Revenue Y3',
        'data_type': DataType.CURRENCY,
        'description': 'Revenue for 3 years ago (Y-3)',
        'category': Category.REVENUE_HISTORICAL,
        'required': False,
        'sortable': True
    },
    'RevenueY4': {
        'display_name': 'Revenue Y4',
        'data_type': DataType.CURRENCY,
        'description': 'Revenue for 4 years ago (Y-4)',
        'category': Category.REVENUE_HISTORICAL,
        'required': False,
        'sortable': True
    },
    
    # Ranking Fields
    'Rank': {
        'display_name': 'Current Rank',
        'data_type': DataType.INTEGER,
        'description': 'Current ranking by revenue',
        'category': Category.RANKING,
        'required': False,
        'sortable': True
    },
    'RankPrev': {
        'display_name': 'Previous Rank',
        'data_type': DataType.INTEGER,
        'description': 'Previous period ranking for comparison',
        'category': Category.RANKING,
        'required': False,
        'sortable': True
    },
    
    # Current Interaction Fields
    'InteractionYTD': {
        'display_name': 'Total Interactions YTD',
        'data_type': DataType.INTEGER,
        'description': 'Total client interactions year-to-date',
        'category': Category.INTERACTIONS_CURRENT,
        'required': False,
        'sortable': True
    },
    'InteractionCMOCYTD': {
        'display_name': 'CMOC Interactions YTD',
        'data_type': DataType.INTEGER,
        'description': 'Capital Markets Operations Committee interactions YTD',
        'category': Category.INTERACTIONS_CURRENT,
        'required': False,
        'sortable': True
    },
    'InteractionGMOCYTD': {
        'display_name': 'GMOC Interactions YTD',
        'data_type': DataType.INTEGER,
        'description': 'Global Markets Operations Committee interactions YTD',
        'category': Category.INTERACTIONS_CURRENT,
        'required': False,
        'sortable': True
    },
    
    # Previous Interaction Fields
    'InteractionPrevYTD': {
        'display_name': 'Total Interactions Prev YTD',
        'data_type': DataType.INTEGER,
        'description': 'Total client interactions previous year-to-date',
        'category': Category.INTERACTIONS_PREVIOUS,
        'required': False,
        'sortable': True
    },
    'InteractionCMOCPrevYTD': {
        'display_name': 'CMOC Interactions Prev YTD',
        'data_type': DataType.INTEGER,
        'description': 'CMOC interactions previous year-to-date',
        'category': Category.INTERACTIONS_PREVIOUS,
        'required': False,
        'sortable': True
    },
    'InteractionGMOCPrevYTD': {
        'display_name': 'GMOC Interactions Prev YTD',
        'data_type': DataType.INTEGER,
        'description': 'GMOC interactions previous year-to-date',
        'category': Category.INTERACTIONS_PREVIOUS,
        'required': False,
        'sortable': True
    },
    
    # Metadata Fields
    'TimePeriodList': {
        'display_name': 'Time Periods',
        'data_type': DataType.STRING,
        'description': 'Comma-separated list of years with data',
        'category': Category.METADATA,
        'required': False,
        'sortable': False
    },
    'TimePeriodCategory': {
        'display_name': 'Time Period Type',
        'data_type': DataType.STRING,
        'description': 'Fiscal Year (FY) or Calendar Year (CY)',
        'category': Category.METADATA,
        'required': False,
        'sortable': False,
        'filter_values': ['FY', 'CY']
    }
}

###################
# COLUMN METADATA UTILITIES
###################

class ColumnMetadataManager:
    """Centralized manager for column metadata operations"""
    
    @staticmethod
    def get_columns_by_category(category: str) -> Dict[str, str]:
        """Get all columns belonging to a specific category"""
        return {
            metadata['display_name']: api_field 
            for api_field, metadata in COLUMN_METADATA_REGISTRY.items()
            if metadata.get('category') == category
        }
    
    @staticmethod
    def get_available_categories() -> List[str]:
        """Get all available column categories"""
        categories = set()
        for metadata in COLUMN_METADATA_REGISTRY.values():
            if 'category' in metadata:
                categories.add(metadata['category'])
        return sorted(list(categories))
    
    @staticmethod
    def get_column_description(api_field: str) -> str:
        """Get description for a specific API field"""
        if api_field in COLUMN_METADATA_REGISTRY:
            return COLUMN_METADATA_REGISTRY[api_field].get('description', f'Data for {api_field}')
        return f'Data for {api_field}'
    
    @staticmethod
    def get_primary_revenue_fields() -> List[str]:
        """Get the primary revenue fields for summary calculations"""
        return [
            api_field for api_field, metadata in COLUMN_METADATA_REGISTRY.items()
            if metadata.get('category', '').startswith('revenue') and metadata.get('primary_metric', False)
        ]
    
    @staticmethod
    def get_display_columns_mapping() -> Dict[str, str]:
        """Get the complete display name to API field mapping"""
        return {
            metadata['display_name']: api_field
            for api_field, metadata in COLUMN_METADATA_REGISTRY.items()
        }
    
    @staticmethod
    def validate_column_exists(api_field: str) -> bool:
        """Validate that a column exists in the registry"""
        return api_field in COLUMN_METADATA_REGISTRY
    
    @staticmethod
    def add_column_metadata(api_field: str, metadata: Dict[str, Any]) -> None:
        """Add new column metadata to the registry"""
        required_fields = ['display_name', 'data_type', 'description', 'category']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field '{field}' in metadata for {api_field}")
        
        COLUMN_METADATA_REGISTRY[api_field] = metadata
        logger.info(f"Added column metadata for {api_field}")

###################
# Enum Definitions
###################

class CCYEnum(str, Enum):
    """Currency enumeration"""
    usd = 'USD'
    cad = 'CAD'

class TimePeriodEnum(str, Enum):
    """Time period type enumeration"""
    fy = 'FY'  # Fiscal Year
    cy = 'CY'  # Calendar Year

class TimeFilterEnum(str, Enum):
    """Time filter enumeration"""
    year = 'YR'
    quarter = 'QR'
    month = 'MT'
    day = 'DY'
    
class ReportableMaskEnum(IntEnum):
    """Reportable mask enumeration"""
    mask_val_1 = 1
    mask_val_2 = 129

class FocusListFilterEnum(str, Enum):
    """Focus list filter enumeration"""
    focus40 = 'Focus40'
    fs30 = 'FS30'
    corp100 = 'Corp100'
    
class ClientClassificationFilterEnum(str, Enum):
    """Client classification filter enumeration"""
    central_banks = 'Central Banks'
    corporations = 'Corporations'
    financial_institutions = 'Financial Institutions'
    fund_managers = 'Fund Managers'
    governments = 'Governments'
    hedge_funds = 'Hedge Funds'
    insurance = 'Insurance'
    pension_fund_managers = 'Pension Fund Managers'
    individuals = 'Individuals'
    unassigned = 'Unassigned'

class GMPrimaryFilterEnum(IntEnum):
    """GM primary filter enumeration"""
    primary = 1
    secondary = 2

class RegionEnum(str, Enum):
    """Region enumeration"""
    can = 'CAN'
    usa = 'USA'
    eur = 'EUR'
    apac = 'APAC'
    latam = 'LATAM'
    other = 'OTHER'

class MetricEnum(str, Enum):
    """Metric enumeration for client value calculation"""
    cv = 'CV'
    cv_rbccm = 'CV_RBCCM'

class RDSortingCriteriaEnum(str, Enum):
    """Sorting criteria for Risers Decliners API"""
    gainers = 'gainers'
    decliners = 'decliners'
    top = 'top'

class TopTradesNotionalOrCVEnum(str, Enum):
    """Notional or Client Value enumeration"""
    notional = 'Notional'
    client_value = 'Client Value'

class InteractionTypeEnum(str, Enum):
    """Types of client interactions"""
    cmoc = "CMOC"  # Capital Markets Operations Committee
    gmoc = "GMOC"  # Global Markets Operations Committee
    general = "General"

class InteractionStatusEnum(str, Enum):
    """Status of client interactions"""
    completed = "Completed"
    scheduled = "Scheduled"
    none = "None"

###################
# Base Model Class
###################

class BaseFinancialModel(BaseModel):
    """
    Base model for all financial API interactions.
    Provides common functionality for API execution and data display.
    Uses centralized metadata registry for consistent column handling.
    """
    # Common configuration for all derived models
    model_config = ConfigDict(use_enum_values=True, validate_default=True)
    
    # Class variables to be overridden by derived classes
    _endpoint_url: ClassVar[str] = ""  # API endpoint URL
    _display_columns: ClassVar[List[str]] = []  # List of API field names to display
    
    # Instance variable for storing API response
    _service_response: Any = None
    
    @field_validator("product_id_filter", mode="before", check_fields=False)
    def validate_product_id(cls, value):
        """Validate product ID filter to ensure it contains only comma-separated numeric values"""
        if not isinstance(value, str):
            return value
            
        if not "".join([x.strip() for x in value.split(",")]).isnumeric():
            raise ValueError("Invalid product ID filter, expected comma separated product ids.")
        return value
    
    def execute(self):
        """
        Execute the API request to the financial service.
        This method prepares the payload, makes the HTTP request, and stores the response.
        
        Returns:
            The JSON response from the API
        """
        if not self._endpoint_url:
            raise ValueError("No endpoint URL defined for this model")
            
        # Prepare payload from model values
        payload = self.model_dump()
        
        # Convert to list and add enterprise ID
        payload_list = list(payload.values())
        payload_list.insert(0, 272487729)  # Enterprise ID
        
        # Common headers for all requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Basic cXVlcnk6cXVlcnk=",
            "User-Agent": "PostmanRuntime/7.43.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        
        # Final payload structure
        payload_final = {
            "appCode": "tb20",
            "values": payload_list
        }
        
        try:
            # Execute the request with error handling
            response = requests.post(
                self._endpoint_url, 
                json=payload_final, 
                headers=headers, 
                verify=False,
                timeout=30  # Increased timeout for production
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse and store the response
            self._service_response = json.loads(response.content)
            return self._service_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            # For production, return a standard error response format
            self._service_response = {
                "status": "error",
                "message": f"API request failed: {str(e)}",
                "data": []
            }
            return self._service_response
    
    def display(self, max_rows: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Display the API results as a pandas DataFrame.
        Uses the centralized metadata registry for column mapping.

        Args:
            max_rows: Optional limit for the number of rows to display. None shows all rows.
            columns: Optional list of specific API field names to display. None uses default columns.

        Returns:
            pandas DataFrame with formatted data
        """
        # Determine which columns to display
        display_fields = columns if columns else self._display_columns
        if not display_fields:
            # Fallback to all available fields in metadata
            display_fields = list(COLUMN_METADATA_REGISTRY.keys())
        
        # Build column mapping from metadata
        columns_config = {}
        for api_field in display_fields:
            if api_field in COLUMN_METADATA_REGISTRY:
                display_name = COLUMN_METADATA_REGISTRY[api_field]['display_name']
                columns_config[display_name] = api_field
            else:
                # Fallback for fields not in metadata
                columns_config[api_field] = api_field
                logger.warning(f"Field {api_field} not found in metadata registry")

        return self._create_display_dataframe(columns_config, max_rows=max_rows)
    
    def _create_display_dataframe(
        self,
        columns_config: Dict[str, str],
        *,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Create a display DataFrame from API response data.
        Handles different response structures (dict or list).
        
        Args:
            columns_config: Mapping of output column names to API response field names
            max_rows: Optional limit for the number of rows in the returned DataFrame.
                None means all rows will be included.
            
        Returns:
            pandas DataFrame with the specified columns
        """
        if not hasattr(self, "_service_response") or self._service_response is None:
            raise RuntimeError("Query has not been executed yet")
        
        # Create empty DataFrame with the specified columns as fallback
        empty_df = pd.DataFrame({col: [] for col in columns_config.keys()})
        
        # Check if data exists in the response
        if 'data' not in self._service_response:
            logger.warning("No 'data' field in API response")
            return empty_df
        
        data = self._service_response['data']
        
        # Handle dictionary response (aggregated data)
        if isinstance(data, dict):
            return self._process_dict_response(data, columns_config)

        # Handle list response (detailed data)
        if isinstance(data, list):
            return self._process_list_response(data, columns_config, max_rows=max_rows)
        
        # Log unexpected data structure
        logger.warning(f"Unexpected data structure: {type(data)}")
        return empty_df
    
    def _process_dict_response(self, data: Dict[str, Any], columns_config: Dict[str, str]) -> pd.DataFrame:
        """Process dictionary response into a DataFrame"""
        # Create a single row with available fields
        row_data = {}
        
        for col_name, field_path in columns_config.items():
            # Handle nested fields with dot notation (e.g., "stats.total")
            if '.' in field_path:
                parts = field_path.split('.')
                value = data
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                row_data[col_name] = value if value is not None else "N/A"
            else:
                # Direct field access
                row_data[col_name] = data.get(field_path, "N/A")
                
        # Format values using metadata
        for col, val in row_data.items():
            api_field = columns_config.get(col, col)
            row_data[col] = self._format_value(val, api_field)
                
        return pd.DataFrame([row_data])
    
    def _process_list_response(
        self,
        data: List[Dict[str, Any]],
        columns_config: Dict[str, str],
        *,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Process list response into a DataFrame with enhanced formatting."""
        if not data:  # Empty list
            return pd.DataFrame({col: [] for col in columns_config.keys()})

        if max_rows is None:
            max_rows = len(data)
        else:
            max_rows = min(max_rows, len(data))
        
        # Prepare data for DataFrame
        table = {col: [] for col in columns_config.keys()}
        
        # Process each row with error handling
        for i in range(max_rows):
            if i >= len(data):
                break
                
            item = data[i]
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dictionary item at index {i}")
                continue
                
            try:
                for col_name, field_name in columns_config.items():
                    # Get field value, defaulting to "N/A" if missing
                    value = item.get(field_name, "N/A")
                    
                    # Format using metadata
                    formatted_value = self._format_value(value, field_name)
                    table[col_name].append(formatted_value)
                    
            except Exception as e:
                logger.error(f"Error processing item at index {i}: {str(e)}")
                # Skip this item on error
                
        return pd.DataFrame(table)
    
    def _format_value(self, value: Any, api_field: str) -> str:
        """Format a value based on its metadata data type"""
        if value is None or value == "N/A":
            return "N/A" if api_field not in COLUMN_METADATA_REGISTRY else "0" if COLUMN_METADATA_REGISTRY[api_field].get('data_type') == DataType.INTEGER else "N/A"
        
        if api_field not in COLUMN_METADATA_REGISTRY:
            return str(value)
        
        data_type = COLUMN_METADATA_REGISTRY[api_field]['data_type']
        
        if data_type == DataType.CURRENCY:
            if isinstance(value, (int, float)):
                return f"{value:,.2f}"
            elif isinstance(value, str):
                try:
                    num_value = float(value.replace(',', '').replace('$', ''))
                    return f"{num_value:,.2f}"
                except ValueError:
                    return str(value)
        elif data_type == DataType.INTEGER:
            if isinstance(value, (int, float)):
                return str(int(value))
            elif isinstance(value, str):
                try:
                    return str(int(float(value)))
                except ValueError:
                    return "0"
        elif data_type == DataType.PERCENTAGE:
            if isinstance(value, (int, float)):
                return f"{value:.2f}%"
        
        return str(value)

    @classmethod
    def get_columns_by_category(cls, category: str) -> Dict[str, str]:
        """Get all columns belonging to a specific category."""
        return ColumnMetadataManager.get_columns_by_category(category)
    
    @classmethod
    def get_available_categories(cls) -> List[str]:
        """Get all available column categories."""
        return ColumnMetadataManager.get_available_categories()
    
    @classmethod
    def get_column_description(cls, api_field: str) -> str:
        """Get description for a specific API field."""
        return ColumnMetadataManager.get_column_description(api_field)

###################
# Financial Models
###################

class RisersDecliners(BaseFinancialModel):
    """
    Model for retrieving top clients by revenue.
    """
    # API endpoint
    _endpoint_url: ClassVar[str] = f"{BASE_URL}/procedure/memsql__client1__getTopClients"
    
    # Display columns - API field names that should be shown by default
    _display_columns: ClassVar[List[str]] = [
        'ClientName', 'ClientCDRID', 'GID', 'RegionName', 'FocusList', 'HierarchyDepth',
        'RevenueYTD', 'RevenuePrevYTD', 'RevenuePrevYear', 'RevenueY0', 'RevenueY1',
        'RevenueY2', 'RevenueY3', 'RevenueY4', 'Rank', 'RankPrev', 'InteractionYTD',
        'InteractionCMOCYTD', 'InteractionGMOCYTD', 'InteractionPrevYTD',
        'InteractionCMOCPrevYTD', 'InteractionGMOCPrevYTD', 'TimePeriodList', 'TimePeriodCategory'
    ]

    # Model fields
    ccy_code: CCYEnum = Field(description="Currency to report in. USD or CAD.", default=CCYEnum.usd)
    time_period: TimePeriodEnum = Field(description="Time Period. Fiscal/FY or Calendar/CY.", default=TimePeriodEnum.fy)
    reportable_mask: Optional[ReportableMaskEnum] = Field(description="ReportableMask, value is 1 or 129, default is 1 (not including PTV) if not mentioned, optional.", default=ReportableMaskEnum.mask_val_1)
    product_id_filter: str = Field(description="Product ID Filter. Comma-separated list of product IDs. Each product ID is an integer, representing product from product hierarchy. 1 means all Level 1 Capital Markets Products.", default="1")
    focus_list_filter: Optional[FocusListFilterEnum] = Field(description="Focus List Filter. Focus 40, Sponsor 30, Corporate 100. null values means all. Valid values: Focus40, FS30, Corp100.", default=None)
    client_classification_filter: Optional[ClientClassificationFilterEnum] = Field(description="Client Classification Filter. null value means all industries. Valid values: Central Banks,Corporations, Financial Institutions, Fund Managers, Governments, Hedge Funds, Insurance, Pension Fund Managers, Individuals, Unassigned.", default=None)
    gm_primary_filter: Optional[GMPrimaryFilterEnum] = Field(description="GMPrimaryIDFilter 1 Primary 2 Secondary. Valid values: 1, 2.", default=None)
    region_filter: Optional[RegionEnum] = Field(description="Region Filter. null value means all regions. Valid values: CAN, USA, EUR, APAC, LATAM, OTHER.", default=None)
    sorting_criteria: RDSortingCriteriaEnum = Field(description="Sorting criteria for the top clients. Valid values: gainers, decliners, top.", default=RDSortingCriteriaEnum.top)
    metric: MetricEnum = Field(description="Metric to calculate Client Value. Either methodology standard or RBC Capital Markets. CV or CV_RBCCM.", default=MetricEnum.cv)
    time_period_year: Optional[int] = Field(description="Time period - Year. Covers from current to last 3 years.", default=None)
    client_hierarchy_depth: int = Field(description="Client Hierarchy Depth.", default=1)
    ex_limit: int = Field(description="Ex limit.", default=0)

class ClientValueByTimePeriod(BaseFinancialModel):
    """
    Model for retrieving client value by time period.
    """
    # API endpoint
    _endpoint_url: ClassVar[str] = f"{BASE_URL}/procedure/memsql__client1__getRevenueTotalByTimePeriod"
    
    # Display columns - API field names
    _display_columns: ClassVar[List[str]] = [
        'ClientName', 'ClientCDRID', 'RevenueYTD', 'RevenuePrevYTD',
        'InteractionCMOCYTD', 'InteractionGMOCYTD', 'InteractionYTD',
        'TimePeriodList', 'TimePeriodCategory'
    ]
    
    # Model fields
    ccy_code: CCYEnum = Field(description="Currency to report in. USD or CAD.", default=CCYEnum.usd)
    reportable_mask: Optional[ReportableMaskEnum] = Field(description="ReportableMask, value is 1 or 129, default is 1 (not including PTV) if not mentioned, optional.", default=ReportableMaskEnum.mask_val_1)
    time_period: TimePeriodEnum = Field(description="Time Period. Fiscal/FY or Calendar/CY.", default=TimePeriodEnum.fy)
    product_id_filter: str = Field(description="Product ID Filter. Comma-separated list of product IDs. Each product ID is an integer, representing product from product hierarchy. 1 means all Level 1 Capital Markets Products.", default="1")
    focus_list_filter: Optional[FocusListFilterEnum] = Field(description="Focus List Filter. Focus 40, Sponsor 30, Corporate 100. null values means all. Valid values: Focus40, FS30, Corp100.", default=None)
    client_classification_filter: Optional[ClientClassificationFilterEnum] = Field(description="Client Classification Filter. null value means all industries. Valid values: Central Banks,Corporations, Financial Institutions, Fund Managers, Governments, Hedge Funds, Insurance, Pension Fund Managers, Individuals, Unassigned.", default=None)
    gm_primary_filter: Optional[GMPrimaryFilterEnum] = Field(description="GMPrimaryIDFilter 1 Primary 2 Secondary. Valid values: 1, 2.", default=None)
    region_filter: Optional[RegionEnum] = Field(description="Region Filter. null value means all regions. Valid values: CAN, USA, EUR, APAC, LATAM, OTHER.", default=None)
    metric: MetricEnum = Field(description="Metric to calculate Client Value. Either methodology standard or RBC Capital Markets. CV or CV_RBCCM.", default=MetricEnum.cv)
    time_period_filter: TimeFilterEnum = Field(description="Time period Filter. Valid values: YR, QR, MT, DY.", default=TimeFilterEnum.year)
    time_period_year: int = Field(description="Time period - Year. Covers from current to last 3 years.", default=2025)
    client_hierarchy_depth: int = Field(description="Client Hierarchy Depth.", default=1)

class ClientValueByProduct(BaseFinancialModel):
    """
    Model for retrieving client value by product.
    """
    # API endpoint
    _endpoint_url: ClassVar[str] = f"{BASE_URL}/procedure/memsql__client1__getClientValueRevenueByProduct"
    
    # Display columns - API field names (Note: ProductName, ProductID etc. are not in our main registry)
    _display_columns: ClassVar[List[str]] = [
        'ProductName', 'RevenueYTD', 'RevenuePrevYTD', 'ProductID',
        'ProductHierarchyDepth', 'ParentProductID', 'TimePeriodList'
    ]
    
    # Model fields
    ccy_code: CCYEnum = Field(description="Currency to report in. USD or CAD.", default=CCYEnum.usd)
    product_id_filter: str = Field(description="Product ID Filter. Comma-separated list of product IDs. Each product ID is an integer, representing product from product hierarchy. 1 means all Level 1 Capital Markets Products.", default="1")
    focus_list_filter: Optional[FocusListFilterEnum] = Field(description="Focus List Filter. Focus 40, Sponsor 30, Corporate 100. null values means all. Valid values: Focus40, FS30, Corp100.", default=None)
    client_classification_filter: Optional[ClientClassificationFilterEnum] = Field(description="Client Classification Filter. null value means all industries. Valid values: Central Banks,Corporations, Financial Institutions, Fund Managers, Governments, Hedge Funds, Insurance, Pension Fund Managers, Individuals, Unassigned.", default=None)
    gm_primary_filter: Optional[GMPrimaryFilterEnum] = Field(description="GMPrimaryIDFilter 1 Primary 2 Secondary. Valid values: 1, 2.", default=None)
    region_filter: Optional[RegionEnum] = Field(description="Region Filter. null value means all regions. Valid values: CAN, USA, EUR, APAC, LATAM, OTHER.", default=None)
    time_period: TimePeriodEnum = Field(description="Time Period. Fiscal/FY or Calendar/CY.", default=TimePeriodEnum.fy)
    reportable_mask: Optional[ReportableMaskEnum] = Field(description="ReportableMask, value is 1 or 129, default is 1 (not including PTV) if not mentioned, optional.", default=ReportableMaskEnum.mask_val_1)
    metric: MetricEnum = Field(description="Metric to calculate Client Value. Either methodology standard or RBC Capital Markets. CV or CV_RBCCM.", default=MetricEnum.cv)
    time_period_year: Optional[int] = Field(description="Time period - Year. Covers from current to last 3 years.", default=2025)
    client_cdrid: Optional[int] = Field(description="Unique Client Identifier.", default=None)
    client_hierarchy_depth: int = Field(description="Client Hierarchy Depth.", default=1)

###################
# Helper Functions
###################

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a markdown table string.
    
    Args:
        df: The DataFrame to convert
        
    Returns:
        Markdown table representation of the DataFrame
    """
    if df.empty:
        return "No data available"
    
    # Use pandas to_markdown for clean conversion
    try:
        return df.to_markdown(index=False)
    except AttributeError:
        # Fallback if to_markdown is not available
        # Create header
        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        
        # Create rows
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(val) for val in row) + " |")
        
        # Combine
        return "\n".join([header, separator] + rows)

###################
# MCP Tool Registration
###################

def register_tools(mcp):
    """Register ClientView financial tools with the MCP server"""
    
    
    @register_tool(
        name="get_top_clients",
        description="Retrieve and analyze top-performing clients by revenue. Essential for CRM client relationship management and revenue analysis. Use this to identify key accounts and revenue trends.",
        namespace="crm",
        input_schema={
            "sorting": {"type": "string", "description": "Sorting criteria: 'top' (highest revenue), 'gainers' (biggest growth), or 'decliners' (biggest decline)", "default": "top"},
            "currency": {"type": "string", "description": "Currency for revenue reporting: 'USD' or 'CAD'", "default": "USD"},
            "region": {"type": "string", "description": "Geographic region filter: 'USA', 'CAN', 'EUR', 'APAC', 'LATAM', 'OTHER'", "default": None},
            "focus_list": {"type": "string", "description": "Strategic client list filter: 'Focus40' (top 40 clients), 'FS30' (sponsor 30), 'Corp100' (corporate 100)", "default": None}
        }
    )
    @llm_enhance_wrapper(
        instruction="Present this client revenue data in a clear, organized format. Highlight notable trends and explain the significance of the top clients in terms of their importance to the business. Include insights that would be valuable for CRM strategy and client relationship management.",
        system_prompt="You are an expert CRM analyst specializing in client revenue analysis and relationship management."
    )
    @mcp.tool()
    async def get_top_clients(
        ctx: Context,
        sorting: str = "top",
        currency: str = "USD",
        region: str | None = None,
        focus_list: str | None = None,
    ) -> str:
        """
        Get top clients by revenue.
        
        Args:
            ctx: The MCP context
            sorting: Sorting criteria ('top', 'gainers', or 'decliners')
            currency: Currency ('USD' or 'CAD')
            region: Region filter ('USA', 'CAN', 'EUR', 'APAC', 'LATAM', 'OTHER')
            focus_list: Focus list filter ('Focus40', 'FS30', 'Corp100')
            
        Returns:
            Top clients data as a formatted markdown table
        """
        try:
            logger.info(f"Getting top clients with sorting={sorting}, currency={currency}, region={region}, focus_list={focus_list}")
            
            # Map parameters to enum values
            sorting_enum = RDSortingCriteriaEnum.top
            if sorting.lower() == "gainers":
                sorting_enum = RDSortingCriteriaEnum.gainers
            elif sorting.lower() == "decliners":
                sorting_enum = RDSortingCriteriaEnum.decliners
                
            currency_enum = CCYEnum.usd if currency.upper() == "USD" else CCYEnum.cad
            
            region_enum = None
            if region:
                try:
                    region_enum = RegionEnum(region.upper())
                except ValueError:
                    logger.warning(f"Invalid region value: {region}")
            
            focus_list_enum = None
            if focus_list:
                try:
                    focus_list_enum = FocusListFilterEnum(focus_list)
                except ValueError:
                    logger.warning(f"Invalid focus list value: {focus_list}")
            
            # Create and execute query
            rd = RisersDecliners(
                ccy_code=currency_enum,
                sorting_criteria=sorting_enum,
                region_filter=region_enum,
                focus_list_filter=focus_list_enum
            )
            
            result = rd.execute()
            
            # Debug: Log raw API response structure
            if result.get('status') == 'success':
                data = result.get('data', [])
                if data and isinstance(data, list) and len(data) > 0:
                    sample_item = data[0]
                    revenue_fields = {k: v for k, v in sample_item.items() if 'revenue' in k.lower()}
                    logger.debug(f"Available revenue fields in API response: {revenue_fields}")
                    logger.debug(f"Using RevenueYTD field: {sample_item.get('RevenueYTD', 'NOT FOUND')}")
            
            # Check API call status
            if result.get('status') != 'success':
                return f"Error retrieving top clients: {result.get('message', 'Unknown error')}"
                
            # Get display data
            df = rd.display()
            
            # Format the output
            region_text = f" in {region}" if region else ""
            focus_text = f" in {focus_list}" if focus_list else ""
            
            result = f"## Top Clients by Revenue{region_text}{focus_text} ({currency})\n\n"
            result += dataframe_to_markdown(df)
            
            # Add summary information
            if not df.empty:
                try:
                    # Debug: Log the raw data before processing
                    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                    logger.debug(f"First few rows:\n{df.head()}")
                    
                    # Determine which revenue field to use for summary based on query context
                    # Check if user is asking for previous year data
                    query_context = str(ctx).lower() if hasattr(ctx, 'context') else ""
                    
                    revenue_field = 'Revenue YTD'  # Default
                    if 'Revenue Prev Year' in df.columns and ('previous year' in query_context or 'prev year' in query_context or 'last year' in query_context):
                        revenue_field = 'Revenue Prev Year'
                        logger.info(f"Using {revenue_field} for summary based on query context")
                    elif 'Revenue Prev YTD' in df.columns and ('previous ytd' in query_context or 'prev ytd' in query_context):
                        revenue_field = 'Revenue Prev YTD'
                        logger.info(f"Using {revenue_field} for summary based on query context")
                    
                    # For revenue columns, convert to float first
                    revenue_col = df[revenue_field]
                    logger.debug(f"Revenue column ({revenue_field}) values: {revenue_col.tolist()[:5]}")
                    
                    # Handle both formatted strings and raw numbers
                    total_revenue = 0
                    for rev in revenue_col:
                        if isinstance(rev, str):
                            # Remove commas and convert to float
                            rev_clean = rev.replace(',', '').replace('$', '')
                            total_revenue += float(rev_clean)
                        elif isinstance(rev, (int, float)):
                            total_revenue += float(rev)
                    
                    logger.debug(f"Calculated total revenue: {total_revenue}")
                    
                    # Create more descriptive summary
                    field_description = {
                        'Revenue YTD': 'YTD',
                        'Revenue Prev YTD': 'Previous YTD', 
                        'Revenue Prev Year': 'Previous Year'
                    }.get(revenue_field, revenue_field)
                    
                    result += f"\n\nTotal {field_description} Revenue (top {len(df)} clients): {total_revenue:,.2f} {currency}"
                except Exception as e:
                    logger.error(f"Error calculating total revenue: {e}")
                    # Add debug info about the raw API response
                    if hasattr(rd, '_service_response') and rd._service_response:
                        sample_data = rd._service_response.get('data', [])
                        if sample_data:
                            logger.debug(f"Sample raw API data: {sample_data[0] if isinstance(sample_data, list) else sample_data}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting top clients: {e}")
            return f"Error retrieving top clients data: {str(e)}"

    
    @register_tool(
        name="get_client_value_by_product",
        description="Analyze client revenue distribution across product lines. Critical for CRM product strategy and client relationship management. Use this to understand product preferences and opportunities for cross-selling.",
        namespace="crm",
        input_schema={
            "client_cdrid": {"type": "integer", "description": "Unique client identifier (CDRID) for targeted analysis"},
            "currency": {"type": "string", "description": "Currency for revenue reporting: 'USD' or 'CAD'", "default": "USD"}
        }
    )
    @llm_enhance_wrapper(
        instruction="Present this client product revenue data in a clear, organized format. Highlight the most significant product categories and explain their importance to the client's overall revenue. Include insights that would help with CRM product strategy and relationship management.",
        system_prompt="You are an expert CRM analyst specializing in product revenue analysis and client relationship management."
    )
    @mcp.tool()
    async def get_client_value_by_product(ctx: Context, client_cdrid: int, currency: str = "USD") -> str:
        """
        Get client value breakdown by product for a specific client.
        
        Args:
            ctx: The MCP context
            client_cdrid: Client CDRID (unique identifier)
            currency: Currency ('USD' or 'CAD')
            
        Returns:
            Client value by product as a formatted markdown table
        """
        try:
            logger.info(f"Getting client value by product for client_cdrid={client_cdrid}, currency={currency}")
            
            currency_enum = CCYEnum.usd if currency.upper() == "USD" else CCYEnum.cad
            
            # Create and execute query
            cvbp = ClientValueByProduct(
                ccy_code=currency_enum,
                client_cdrid=client_cdrid
            )
            
            result = cvbp.execute()
            
            # Check API call status
            if result.get('status') != 'success':
                return f"Error retrieving client value by product: {result.get('message', 'Unknown error')}"
                
            # Get display data
            df = cvbp.display()
            
            # Format the output
            output = f"## Client Value by Product for CDRID {client_cdrid} ({currency})\n\n"
            
            if df.empty:
                output += "No product data found for this client."
            else:
                output += dataframe_to_markdown(df)
                
                # Add summary information
                try:
                    revenue_col = df['Revenue YTD']
                    total_revenue = sum(float(rev.replace(',', '')) for rev in revenue_col)
                    output += f"\n\nTotal Revenue: {total_revenue:,.2f} {currency}"
                except Exception as e:
                    logger.error(f"Error calculating total revenue: {e}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error getting client value by product: {e}")
            return f"Error retrieving client value by product: {str(e)}"

    
    @register_tool(
        name="get_client_value_by_time",
        description="Track and analyze client revenue trends over time. Essential for CRM performance tracking and relationship management. Use this to identify growth patterns and seasonal trends.",
        namespace="crm",
        input_schema={
            "time_filter": {"type": "string", "description": "Time period granularity: 'YR' (yearly), 'QR' (quarterly), 'MT' (monthly), 'DY' (daily)", "default": "YR"},
            "currency": {"type": "string", "description": "Currency for revenue reporting: 'USD' or 'CAD'", "default": "USD"},
            "time_period": {"type": "string", "description": "Fiscal period type: 'FY' (fiscal year) or 'CY' (calendar year)", "default": "FY"},
            "time_period_year": {"type": "integer", "description": "Analysis year (e.g., 2025)", "default": 2025}
        }
    )
    @llm_enhance_wrapper(
        instruction="Present this time-based revenue data in a clear, organized format. Analyze trends over time and highlight periods of significant growth or decline. Include insights that would be valuable for CRM strategy and client relationship management.",
        system_prompt="You are an expert CRM analyst specializing in revenue trend analysis and client relationship management."
    )
    @mcp.tool()
    async def get_client_value_by_time(
        ctx: Context, 
        time_filter: str = "YR", 
        currency: str = "USD",
        time_period: str = "FY",
        time_period_year: int = 2025
    ) -> str:
        """
        Get client value across time periods.
        
        Args:
            ctx: The MCP context
            time_filter: Time period filter ('YR', 'QR', 'MT', 'DY')
            currency: Currency ('USD' or 'CAD')
            time_period: Time period ('FY' or 'CY')
            time_period_year: Time period year
            
        Returns:
            Client value by time as a formatted markdown table
        """
        try:
            logger.info(f"Getting client value by time with time_filter={time_filter}, currency={currency}, time_period={time_period}, year={time_period_year}")
            
            # Map parameters to enum values
            currency_enum = CCYEnum.usd if currency.upper() == "USD" else CCYEnum.cad
            
            time_filter_enum = TimeFilterEnum.year
            if time_filter.upper() == "QR":
                time_filter_enum = TimeFilterEnum.quarter
            elif time_filter.upper() == "MT":
                time_filter_enum = TimeFilterEnum.month
            elif time_filter.upper() == "DY":
                time_filter_enum = TimeFilterEnum.day
                
            time_period_enum = TimePeriodEnum.fy if time_period.upper() == "FY" else TimePeriodEnum.cy
            
            # Create and execute query
            cvbt = ClientValueByTimePeriod(
                ccy_code=currency_enum,
                time_period_filter=time_filter_enum,
                time_period=time_period_enum,
                time_period_year=time_period_year
            )
            
            result = cvbt.execute()
            
            # Check API call status
            if result.get('status') != 'success':
                return f"Error retrieving client value by time: {result.get('message', 'Unknown error')}"
                
            # Format period information
            time_period_desc = {
                'YR': 'Year',
                'QR': 'Quarter', 
                'MT': 'Month',
                'DY': 'Day'
            }.get(time_filter.upper(), time_filter)
            
            # Format the output
            output = f"## Client Value by {time_period_desc} ({time_period} {time_period_year}, {currency})\n\n"
            # Get display data
            df = cvbt.display()
            
            if df.empty:
                output += "No time-based revenue data found."
            else:
                output += dataframe_to_markdown(df)
                
                # Add summary information
                try:
                    revenue_col = df['Revenue YTD']
                    total_revenue = sum(float(rev.replace(',', '')) for rev in revenue_col)
                    output += f"\n\nTotal Revenue: {total_revenue:,.2f} {currency}"
                except Exception as e:
                    logger.error(f"Error calculating total revenue: {e}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error getting client value by time: {e}")
            return f"Error retrieving client value by time: {str(e)}"

    @register_tool(
        name="list_available_data_fields",
        description="List all available data fields and their descriptions for client financial data. Use this to understand what information is available before querying.",
        namespace="crm",
        input_schema={
            "category_filter": {"type": "string", "description": "Optional category to filter by: 'identification', 'demographics', 'revenue_current', 'revenue_previous', 'revenue_historical', 'ranking', 'interactions_current', 'interactions_previous', 'metadata'", "default": None}
        }
    )
    @mcp.tool()
    async def list_available_data_fields(ctx: Context, category_filter: str = None) -> str:
        """
        List all available data fields and their descriptions.
        
        Args:
            ctx: The MCP context
            category_filter: Optional category to filter by
            
        Returns:
            List of available fields with descriptions
        """
        try:
            logger.info(f"Listing available data fields with category_filter={category_filter}")
            
            # Get metadata from the centralized registry
            metadata = COLUMN_METADATA_REGISTRY
            
            if not metadata:
                return "No metadata available"
            
            # Filter by category if specified
            if category_filter:
                filtered_metadata = {
                    field: meta for field, meta in metadata.items() 
                    if meta.get('category') == category_filter
                }
                if not filtered_metadata:
                    available_categories = ColumnMetadataManager.get_available_categories()
                    return f"No fields found for category '{category_filter}'. Available categories: {', '.join(available_categories)}"
                metadata = filtered_metadata
            
            # Build the output
            result = "## Available Data Fields"
            if category_filter:
                result += f" - Category: {category_filter.title()}"
            result += "\n\n"
            
            # Group by category
            categories = {}
            for field, meta in metadata.items():
                category = meta.get('category', 'uncategorized')
                if category not in categories:
                    categories[category] = []
                categories[category].append((field, meta))
            
            # Display by category
            for category, fields in sorted(categories.items()):
                result += f"### {category.replace('_', ' ').title()}\n\n"
                
                for field, meta in sorted(fields, key=lambda x: x[1].get('display_name', x[0])):
                    display_name = meta.get('display_name', field)
                    data_type = meta.get('data_type', 'unknown')
                    description = meta.get('description', 'No description available')
                    required = meta.get('required', False)
                    sortable = meta.get('sortable', False)
                    
                    result += f"- **{display_name}** (`{field}`) - *{data_type}*"
                    if required:
                        result += " **[Required]**"
                    if sortable:
                        result += " **[Sortable]**"
                    result += f"\n  {description}\n"
                    
                    # Add filter values if available
                    if 'filter_values' in meta:
                        result += f"  *Possible values: {', '.join(meta['filter_values'])}*\n"
                    result += "\n"
            
            # Add summary
            result += f"\n**Total Fields Available:** {len(metadata)}\n"
            
            if not category_filter:
                available_categories = ColumnMetadataManager.get_available_categories()
                result += f"**Available Categories:** {', '.join(available_categories)}\n"
                result += "\nTo filter by category, use the `category_filter` parameter with one of the available categories."
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing available data fields: {e}")
            return f"Error retrieving data field information: {str(e)}"

# Log registration of tools
n ,,,,,logger.info("Registere,,,1d ClientView financial tools: get_top_clients, get_client_value_by_product, get_client_value_by_time, list_available_data_fields")

###################
# METADATA UTILITIES
###################

def add_new_column_metadata(api_field: str, display_name: str, data_type: str, 
                           description: str, category: str, **kwargs) -> None:
    """
    Utility function to add new column metadata to the registry.
    
    Args:
        api_field: The API field name (e.g., 'ClientName')
        display_name: Human-readable display name (e.g., 'Client Name')
        data_type: Data type using DataType constants (e.g., DataType.STRING)
        description: Description of what this field contains
        category: Category using Category constants (e.g., Category.IDENTIFICATION)
        **kwargs: Additional metadata properties (required, sortable, filter_values, etc.)
    
    Example:
        add_new_column_metadata(
            'NewRevenueField',
            'New Revenue Field', 
            DataType.CURRENCY,
            'Revenue from new product line',
            Category.REVENUE_CURRENT,
            required=False,
            sortable=True
        )
    """
    try:
        metadata = {
            'display_name': display_name,
            'data_type': data_type,
            'description': description,
            'category': category,
            **kwargs
        }
        
        ColumnMetadataManager.add_column_metadata(api_field, metadata)
        logger.info(f"Successfully added metadata for column: {api_field}")
        
    except Exception as e:
        logger.error(f"Failed to add metadata for column {api_field}: {e}")
        raise

def validate_metadata_integrity() -> Dict[str, Any]:
    """
    Validate the integrity of the metadata registry.
    
    Returns:
        Dictionary with validation results and statistics
    """
    validation_results = {
        'total_fields': len(COLUMN_METADATA_REGISTRY),
        'categories': ColumnMetadataManager.get_available_categories(),
        'missing_required_fields': [],
        'invalid_data_types': [],
        'invalid_categories': [],
        'duplicate_display_names': []
    }
    
    valid_data_types = [DataType.STRING, DataType.INTEGER, DataType.CURRENCY, 
                       DataType.PERCENTAGE, DataType.DATE, DataType.BOOLEAN]
    valid_categories = [Category.IDENTIFICATION, Category.DEMOGRAPHICS, 
                       Category.REVENUE_CURRENT, Category.REVENUE_PREVIOUS,
                       Category.REVENUE_HISTORICAL, Category.RANKING,
                       Category.INTERACTIONS_CURRENT, Category.INTERACTIONS_PREVIOUS,
                       Category.METADATA]
    
    display_names = []
    
    for api_field, metadata in COLUMN_METADATA_REGISTRY.items():
        # Check required fields
        required_fields = ['display_name', 'data_type', 'description', 'category']
        for field in required_fields:
            if field not in metadata:
                validation_results['missing_required_fields'].append(f"{api_field}.{field}")
        
        # Check data types
        if metadata.get('data_type') not in valid_data_types:
            validation_results['invalid_data_types'].append(f"{api_field}: {metadata.get('data_type')}")
        
        # Check categories
        if metadata.get('category') not in valid_categories:
            validation_results['invalid_categories'].append(f"{api_field}: {metadata.get('category')}")
        
        # Check for duplicate display names
        display_name = metadata.get('display_name')
        if display_name in display_names:
            validation_results['duplicate_display_names'].append(display_name)
        else:
            display_names.append(display_name)
    
    validation_results['is_valid'] = (
        len(validation_results['missing_required_fields']) == 0 and
        len(validation_results['invalid_data_types']) == 0 and
        len(validation_results['invalid_categories']) == 0 and
        len(validation_results['duplicate_display_names']) == 0
    )
    
    return validation_results

# Validate metadata on module load
try:
    validation = validate_metadata_integrity()
    if validation['is_valid']:
        logger.info(f"✅ Metadata registry validation passed. {validation['total_fields']} fields across {len(validation['categories'])} categories.")
    else:
        logger.warning("⚠️ Metadata registry validation issues found:")
        for issue_type, issues in validation.items():
            if isinstance(issues, list) and issues:
                logger.warning(f"  {issue_type}: {issues}")
except Exception as e:
    logger.error(f"❌ Failed to validate metadata registry: {e}")

logger.info("🚀 ClientView Financials module loaded successfully with centralized metadata registry") 