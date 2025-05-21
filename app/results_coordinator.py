import logging


logger = logging.getLogger(__name__)


class ResultsCoordinator:
    """Coordinates execution of tool plans and combines results.

    The coordinator can optionally stream step results for responsive UIs.
    """

    def __init__(self, sqlite_cache=None, vector_cache=None, fallback_handler=None):
        self.sqlite_cache = sqlite_cache
        self.vector_cache = vector_cache
        self.fallback_handler = fallback_handler

    async def process_plan_streaming(self, plan, bridge):
        """Async generator yielding each step result as it completes."""
        context = {}
        for step in plan:
            tool = step.get("tool")
            params = step.get("parameters", {})
            use_ctx = step.get("use_context", {})
            for target, source in use_ctx.items():
                if source in context:
                    params[target] = context[source]

            result = await bridge.execute_tool(tool, params)
            if self._has_data(result):
                for key in step.get("output_context", []):
                    if isinstance(result, dict) and key in result:
                        context[key] = result[key]
                yield result
            else:
                if self.fallback_handler:
                    fb = await self.fallback_handler.handle_fallback(tool, params, step.get("query", ""))
                    if fb:
                        yield fb

    async def process_plan(self, plan, bridge):
        """Execute a multi-step plan and collect all results."""
        results = []
        async for res in self.process_plan_streaming(plan, bridge):
            results.append(res)
        return results

    def _has_data(self, result):
        if result is None:
            return False
        if isinstance(result, dict):
            data = result.get("data")
            return bool(data)
        return True
