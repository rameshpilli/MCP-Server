from typing import Dict, Any
import time
from datetime import datetime
from .query_mapper import QueryMapper
from .query_logger import QueryLogger, QueryLog, SnowflakeConnector

class ModelQueryHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_mapper = QueryMapper()
        self.query_logger = QueryLogger(config)
        self.snowflake = SnowflakeConnector(config)

    async def handle_query(self, model_id: str, query_text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Map natural language query to SQL
            sql_query, query_type, params = self.query_mapper.map_query(query_text)
            
            if not sql_query:
                response = {
                    "status": "error",
                    "error": "Could not understand the query"
                }
                execution_time = time.time() - start_time
                
                await self._log_query(
                    model_id=model_id,
                    query_text=query_text,
                    query_type="unknown",
                    parameters={},
                    response=response,
                    execution_time=execution_time,
                    status="error",
                    error="Query mapping failed"
                )
                
                return response

            # Execute query on Snowflake
            result = await self.snowflake.execute_query(sql_query, params)
            execution_time = time.time() - start_time

            # Log the query
            await self._log_query(
                model_id=model_id,
                query_text=query_text,
                query_type=query_type,
                parameters=params,
                response=result,
                execution_time=execution_time,
                status=result["status"],
                error=result.get("error")
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_response = {
                "status": "error",
                "error": str(e)
            }

            await self._log_query(
                model_id=model_id,
                query_text=query_text,
                query_type="unknown",
                parameters={},
                response=error_response,
                execution_time=execution_time,
                status="error",
                error=str(e)
            )

            return error_response

    async def _log_query(
        self,
        model_id: str,
        query_text: str,
        query_type: str,
        parameters: Dict[str, Any],
        response: Dict[str, Any],
        execution_time: float,
        status: str,
        error: str = None
    ):
        log_entry = QueryLog(
            model_id=model_id,
            query_text=query_text,
            query_type=query_type,
            parameters=parameters,
            response=response,
            timestamp=datetime.utcnow(),
            execution_time=execution_time,
            status=status,
            error=error
        )
        
        await self.query_logger.log_query(log_entry) 