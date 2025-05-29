# app/llm_wrapper.py
"""
LLMWrapper:
    - This file defines a wrapper that lets LangChain use our internal LLM (the one behind `call_llm()`).
    - Normally, LangChain tries to connect to OpenAI — but we can't allow that in our setup.
    - So instead, we use our secured, bearer-token-based LLM endpoint through our MCP client.

Why we need this:
    - LangChain agents need a chat model that follows their expected interface.
    - This wrapper makes our internal LLM "look like" a LangChain-compatible chat model.
    - That way, agents like `initialize_agent(..., llm=LLMWrapper())` just work.

How it works:
    - Converts LangChain chat messages (system/user/assistant) into our format.
    - Sends them to `mcp_client.call_llm()` which handles token, auth, and calling the LLM.
    - Formats the result back into what LangChain expects.

Used in:
    - `LangChainBridge`, instead of `ChatOpenAI`.

"""
# from typing import List, Optional, Any, Callable
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
# from langchain_core.outputs import ChatResult, ChatGeneration
#
#
# class LLMWrapper(BaseChatModel):
#     """
#     Makes our internal LLM look like a LangChain-compatible chat model.
#     """
#
#     def __init__(self, call_llm_fn: Optional[Callable] = None):
#         super().__init__()
#         self.call_llm_fn = call_llm_fn  # optional override
#
#     async def _agenerate(
#         self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs
#     ) -> ChatResult:
#         # Lazy import to avoid circular dependency
#         if self.call_llm_fn is not None:
#             call_llm = self.call_llm_fn
#         else:
#             from app.client import mcp_client  # 🧠 Import inside the function
#             call_llm = mcp_client.call_llm
#
#         # Convert LangChain messages to OpenAI format
#         prompt = []
#         for m in messages:
#             if isinstance(m, SystemMessage):
#                 prompt.append({"role": "system", "content": m.content})
#             elif isinstance(m, HumanMessage):
#                 prompt.append({"role": "user", "content": m.content})
#             elif isinstance(m, AIMessage):
#                 prompt.append({"role": "assistant", "content": m.content})
#
#         raw_response = await call_llm(prompt)
#
#         # Extract assistant reply
#         if isinstance(raw_response, dict):
#             text = raw_response["choices"][0]["message"]["content"]
#         else:
#             text = str(raw_response)
#
#         return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
#
#     async def ainvoke(self, input: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
#         result = await self._agenerate(input, stop=stop, **kwargs)
#         return result.generations[0].message
#
#     def _generate(self, *args, **kwargs):
#         raise NotImplementedError("Synchronous mode is not supported.")
#
#     def invoke(self, *args, **kwargs):
#         raise NotImplementedError("Synchronous mode is not supported.")
#
#     @property
#     def _llm_type(self) -> str:
#         return "custom-internal-llm"


# from typing import List, Optional, Any, Callable
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
# from langchain_core.outputs import ChatResult, ChatGeneration
# from pydantic import Field
#
#
# class LLMWrapper(BaseChatModel):
#     """
#     Makes our internal LLM look like a LangChain-compatible chat model.
#     """
#
#     call_llm_fn: Optional[Callable] = Field(default=None, exclude=True)
#
#     def __init__(self, call_llm_fn: Optional[Callable] = None, **kwargs):
#         super().__init__(**kwargs)
#         self.call_llm_fn = call_llm_fn
#
#     async def _agenerate(
#         self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs
#     ) -> ChatResult:
#         if self.call_llm_fn is not None:
#             call_llm = self.call_llm_fn
#         else:
#             from app.client import mcp_client  # fallback import
#             call_llm = mcp_client.call_llm
#
#         prompt = []
#         for m in messages:
#             if isinstance(m, SystemMessage):
#                 prompt.append({"role": "system", "content": m.content})
#             elif isinstance(m, HumanMessage):
#                 prompt.append({"role": "user", "content": m.content})
#             elif isinstance(m, AIMessage):
#                 prompt.append({"role": "assistant", "content": m.content})
#
#         raw_response = await call_llm(prompt)
#
#         if isinstance(raw_response, dict):
#             text = raw_response["choices"][0]["message"]["content"]
#         else:
#             text = str(raw_response)
#
#         return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
#
#     async def ainvoke(
#         self, input: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
#     ) -> BaseMessage:
#         result = await self._agenerate(input, stop=stop, **kwargs)
#         return result.generations[0].message
#
#     def _generate(self, *args, **kwargs):
#         raise NotImplementedError("Synchronous mode is not supported.")
#
#     def invoke(self, *args, **kwargs):
#         raise NotImplementedError("Synchronous mode is not supported.")
#
#     @property
#     def _llm_type(self) -> str:
#         return "custom-internal-llm"

# app/llm_wrapper.py

from typing import List, Optional, Any, Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
import asyncio


class LLMWrapper(BaseChatModel):
    """
    Makes our internal LLM look like a LangChain-compatible chat model.
    Supports both async and sync LangChain execution paths.
    """

    call_llm_fn: Optional[Callable] = Field(default=None, exclude=True)

    def __init__(self, call_llm_fn: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.call_llm_fn = call_llm_fn

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        # Fallback to global MCP client
        if self.call_llm_fn is not None:
            call_llm = self.call_llm_fn
        else:
            from app.client import mcp_client
            call_llm = mcp_client.call_llm

        # Format messages to OpenAI-style prompt
        prompt = []
        for m in messages:
            if isinstance(m, SystemMessage):
                prompt.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                prompt.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                prompt.append({"role": "assistant", "content": m.content})

        if not prompt:
            raise ValueError("LLMWrapper: Cannot send empty prompt to LLM.")

        raw_response = await call_llm(prompt)

        if isinstance(raw_response, dict):
            text = raw_response["choices"][0]["message"]["content"]
        else:
            text = str(raw_response)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def ainvoke(
        self, input: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> BaseMessage:
        result = await self._agenerate(input, stop=stop, **kwargs)
        return result.generations[0].message

    def generate(
        self, messages: List[List[BaseMessage]], **kwargs
    ) -> ChatResult:
        """
        LangChain sometimes calls sync generate(). We wrap _agenerate to support this.
        """
        return asyncio.run(self._agenerate(messages[0], **kwargs))

    def _generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._generate(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom-internal-llm"
