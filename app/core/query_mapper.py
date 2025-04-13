from typing import Dict, Any, Optional, Tuple, List
import re
from datetime import datetime, timedelta
from .data_router import DataSourceType

class QueryTemplate:
    def __init__(
        self,
        pattern: str,
        sql_template: str,
        query_type: str,
        data_source: DataSourceType,
        examples: List[str] = None,
        parameters: List[str] = None
    ):
        self.pattern = pattern
        self.sql_template = sql_template
        self.query_type = query_type
        self.data_source = data_source
        self.examples = examples or []
        self.parameters = parameters or []

class QueryMapper:
    """Maps natural language queries to SQL queries"""
    
    def __init__(self):
        self.query_patterns = {
            # Snowflake queries
            r"(?i).*active jobs.*today.*": QueryTemplate(
                pattern="Show active jobs for today",
                sql_template="""
                    SELECT COUNT(*) as active_jobs
                    FROM jobs
                    WHERE status = 'active'
                    AND DATE(created_at) = CURRENT_DATE()
                """,
                query_type="job_count",
                data_source=DataSourceType.SNOWFLAKE
            ),
            
            # Azure Storage queries
            r"(?i).*model artifacts.*": QueryTemplate(
                pattern="Get model artifacts",
                sql_template="""
                    SELECT name, size, last_modified
                    FROM model_artifacts
                    WHERE container = '{container}'
                """,
                query_type="artifact_list",
                data_source=DataSourceType.AZURE_STORAGE
            ),
            
            # Local DB queries
            r"(?i).*api keys.*": QueryTemplate(
                pattern="List API keys",
                sql_template="""
                    SELECT key_id, owner, created_at, expires_at, is_active
                    FROM api_keys
                    WHERE is_active = true
                """,
                query_type="api_key_list",
                data_source=DataSourceType.LOCAL_DB
            )
        }

    def map_query_with_source(
        self,
        natural_query: str
    ) -> Tuple[Optional[str], str, Dict[str, Any], DataSourceType]:
        """
        Maps a natural language query to a SQL query and determines the data source
        Returns: (sql_query, query_type, parameters, data_source)
        """
        for pattern, template in self.query_patterns.items():
            match = re.match(pattern, natural_query)
            if match:
                sql_query = template.sql_template
                params = {}

                # Extract parameters if present in the query
                if template.parameters:
                    # Extract time period parameters if present
                    time_match = re.search(r"(\d+)\s+(hour|day|week)s?", natural_query.lower())
                    if time_match:
                        value = int(time_match.group(1))
                        period = time_match.group(2).upper()
                        if period == "HOUR":
                            period = "HOURS"
                        elif period == "DAY":
                            period = "DAYS"
                        elif period == "WEEK":
                            period = "WEEKS"
                        params = {"value": value, "period": period}
                        sql_query = sql_query.format(**params)

                return sql_query.strip(), template.query_type, params, template.data_source

        return None, "unknown", {}, None

    def get_source_for_query_type(self, query_type: str) -> Optional[DataSourceType]:
        """Determine the appropriate data source for a query type"""
        for template in self.query_patterns.values():
            if template.query_type == query_type:
                return template.data_source
        return None 