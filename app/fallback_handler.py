import logging

logger = logging.getLogger(__name__)


class FallbackHandler:
    """Simple fallback strategy for tool execution."""

    def __init__(self, sqlite_cache=None, vector_db=None, llm_client=None, use_sqlite=True, use_vector=False):
        self.sqlite_cache = sqlite_cache
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.use_sqlite = use_sqlite
        self.use_vector = use_vector

    async def handle_fallback(self, tool_name, failed_params, query):
        """Return a fallback result when a tool fails or yields no data."""
        result = await self._try_general_parameters(tool_name, failed_params)
        if self._has_data(result):
            return result

        if self.use_sqlite and self.sqlite_cache:
            try:
                cached = await self.sqlite_cache.lookup(tool_name, failed_params)
                if self._has_data(cached):
                    return cached
            except Exception as exc:
                logger.debug(f"SQLite cache lookup failed: {exc}")

        if self.use_vector and self.vector_db:
            try:
                similar = await self.vector_db.search(query)
                if self._has_data(similar):
                    return similar
            except Exception as exc:
                logger.debug(f"Vector search failed: {exc}")

        return {
            "data": [],
            "fallback_attempted": True,
            "tool": tool_name
        }

    async def _try_general_parameters(self, tool_name, params):
        """Example generalized retry logic placeholder."""
        try:
            new_params = {k: None for k in params}
            if hasattr(self.llm_client, 'retry_tool'):
                return await self.llm_client.retry_tool(tool_name, new_params)
        except Exception as exc:
            logger.debug(f"General parameter retry failed: {exc}")
        return None

    def _has_data(self, result):
        if not result:
            return False
        if isinstance(result, dict):
            return bool(result.get("data"))
        return True
