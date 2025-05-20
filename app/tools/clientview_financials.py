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
    """
    # Common configuration for all derived models
    model_config = ConfigDict(use_enum_values=True, validate_default=True)
    
    # Class variables to be overridden by derived classes
    _endpoint_url: ClassVar[str] = ""  # API endpoint URL
    _display_columns: ClassVar[Dict[str, str]] = {}  # Column mapping for display
    
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
                timeout=10  # Add timeout for production readiness
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse and store the response
            self._service_response = json.loads(response.content)
            return self._service_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            # For production, consider returning a standard error response format
            self._service_response = {
                "status": "error",
                "message": f"API request failed: {str(e)}",
                "data": []
            }
            return self._service_response
    
    def display(self) -> pd.DataFrame:
        """
        Display the API results as a pandas DataFrame.
        Uses the column mapping defined in the derived class.
        
        Returns:
            pandas DataFrame with formatted data
        """
        if not self._display_columns:
            raise ValueError("No display columns defined for this model")
            
        return self._create_display_dataframe(self._display_columns)
    
    def _create_display_dataframe(self, columns_config: Dict[str, str]) -> pd.DataFrame:
        """
        Create a display DataFrame from API response data.
        Handles different response structures (dict or list).
        
        Args:
            columns_config: Mapping of output column names to API response field names
            
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
            return self._process_list_response(data, columns_config)
        
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
                
        # Format numeric values
        for col, val in row_data.items():
            if isinstance(val, (int, float)):
                row_data[col] = f"{val:.2f}"
                
        return pd.DataFrame([row_data])
    
    def _process_list_response(self, data: List[Dict[str, Any]], columns_config: Dict[str, str]) -> pd.DataFrame:
        """Process list response into a DataFrame with enhanced interaction data handling"""
        if not data:  # Empty list
            return pd.DataFrame({col: [] for col in columns_config.keys()})
            
        # Get max rows to display (up to 10)
        max_rows = min(10, len(data))
        
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
                    
                    # Special handling for interaction fields
                    if 'Interaction' in field_name:
                        if value is None or value == "N/A":
                            value = "0"
                        elif isinstance(value, (int, float)):
                            value = str(int(value))  # Convert to integer string
                    
                    # Format numeric values (excluding interaction counts)
                    elif isinstance(value, (int, float)) and 'Interaction' not in field_name:
                        value = f"{value:.2f}"
                        
                    table[col_name].append(value)
            except Exception as e:
                logger.error(f"Error processing item at index {i}: {str(e)}")
                # Skip this item on error
                
        return pd.DataFrame(table)

###################
# Financial Models
###################

class RisersDecliners(BaseFinancialModel):
    """
    Model for retrieving top clients by revenue.
    """
    # API endpoint
    _endpoint_url: ClassVar[str] = f"{BASE_URL}/procedure/memsql__client1__getTopClients"
    
    # Display column mapping - updated to include interaction data
    _display_columns: ClassVar[Dict[str, str]] = {
        'Client Name': 'ClientName',
        'Client CDRID': 'ClientCDRID',
        'Revenue YTD': 'RevenueYTD',
        'Region': 'RegionName',
        'Focus List': 'FocusList',
        'CMOC Interactions YTD': 'InteractionCMOCYTD',
        'GMOC Interactions YTD': 'InteractionGMOCYTD',
        'Total Interactions YTD': 'InteractionYTD',
        'CMOC Interactions Prev YTD': 'InteractionCMOCPrevYTD',
        'GMOC Interactions Prev YTD': 'InteractionGMOCPrevYTD',
        'Total Interactions Prev YTD': 'InteractionPrevYTD'
    }
    
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
    
    # Display column mapping - updated to include interaction data
    _display_columns: ClassVar[Dict[str, str]] = {
        'Client Name': 'ClientName',
        'Client CDRID': 'ClientCDRID',
        'Revenue YTD': 'RevenueYTD',
        'Revenue Prev YTD': 'RevenuePrevYTD',
        'CMOC Interactions YTD': 'InteractionCMOCYTD',
        'GMOC Interactions YTD': 'InteractionGMOCYTD',
        'Total Interactions YTD': 'InteractionYTD',
        'Time Period List': 'TimePeriodList',
        'Time Period Category': 'TimePeriodCategory'
    }
    
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
    
    # Display column mapping - updated to include interaction data
    _display_columns: ClassVar[Dict[str, str]] = {
        'Product Name': 'ProductName',
        'Revenue YTD': 'RevenueYTD',
        'Revenue Prev YTD': 'RevenuePrevYTD',
        'Product ID': 'ProductID',
        'Product Hierarchy Depth': 'ProductHierarchyDepth',
        'Parent Product ID': 'ParentProductID',
        'Time Period List': 'TimePeriodList'
    }
    
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
    async def get_top_clients(ctx: Context, **kwargs) -> str:
        currency = kwargs.get("currency", "USD")
        sorting = kwargs.get("sorting", "top")
        region = kwargs.get("region")
        focus_list = kwargs.get("focus_list")
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
                    # For revenue columns, convert to float first
                    revenue_col = df['Revenue YTD']
                    total_revenue = sum(float(rev.replace(',', '')) for rev in revenue_col)
                    result += f"\n\nTotal Revenue (top {len(df)} clients): {total_revenue:,.2f} {currency}"
                except Exception as e:
                    logger.error(f"Error calculating total revenue: {e}")
            
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

# Log registration of tools
logger.info("Registered ClientView financial tools: get_top_clients, get_client_value_by_product, get_client_value_by_time") 