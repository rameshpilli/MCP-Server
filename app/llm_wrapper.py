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
import logging


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
        logger = logging.getLogger(__name__)
        logger.debug(f"LLMWrapper received messages: {messages}")
        logger.debug(f"Message types: {[type(m).__name__ for m in messages]}")
        
        # Import message classes for comparison
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        # Fallback to global MCP client
        if self.call_llm_fn is not None:
            call_llm = self.call_llm_fn
        else:
            from app.client import mcp_client
            call_llm = mcp_client.call_llm

        # Format messages to OpenAI-style prompt
        prompt = []
        for i, m in enumerate(messages):
            # CRITICAL FIX: Handle tuple/malformed message objects
            try:
                # First check if it's a tuple (common error case)
                if isinstance(m, tuple):
                    logger.error(f"Message {i} is a tuple: {m}. This indicates a data corruption issue.")
                    # Try to extract content from tuple if possible
                    if len(m) >= 2 and isinstance(m[1], str):
                        content = m[1]
                        role = "user"  # Default fallback
                        logger.warning(f"Extracted content from tuple: {content[:100]}...")
                        prompt.append({"role": role, "content": content})
                        continue
                    else:
                        logger.error(f"Cannot extract content from malformed tuple: {m}")
                        continue
                
                # Check if it has content attribute before accessing it
                if not hasattr(m, 'content'):
                    logger.error(f"Message {i} has no 'content' attribute. Type: {type(m)}, Value: {repr(m)}")
                    continue
                
                content = m.content
                if content is None:
                    logger.warning(f"Message {i} has None content, skipping")
                    continue
                
                logger.debug(f"Processing message {i}: type={type(m)}, content_preview={str(content)[:100]}...")
                
                if isinstance(m, SystemMessage):
                    prompt.append({"role": "system", "content": content})
                    logger.debug(f"Added system message to prompt")
                elif isinstance(m, HumanMessage):
                    prompt.append({"role": "user", "content": content})
                    logger.debug(f"Added user message to prompt")
                elif isinstance(m, AIMessage):
                    prompt.append({"role": "assistant", "content": content})
                    logger.debug(f"Added assistant message to prompt")
                else:
                    logger.warning(f"Unknown message type: {type(m)}")
                    # Determine role based on type name
                    role = "user"  # Default fallback
                    type_name = str(type(m)).lower()
                    if "system" in type_name:
                        role = "system"
                    elif "ai" in type_name or "assistant" in type_name:
                        role = "assistant"
                    
                    prompt.append({"role": role, "content": content})
                    logger.debug(f"Added {role} message to prompt as fallback")
                    
            except Exception as e:
                logger.error(f"Error processing message {i}: {e}")
                logger.error(f"Message type: {type(m)}, repr: {repr(m)}")
                # Skip this message and continue
                continue

        logger.debug(f"Converted prompt: {prompt}")
        logger.debug(f"Prompt length: {len(prompt)}")
        
        if not prompt:
            logger.error(f"Empty prompt after conversion. Original messages: {messages}")
            logger.error(f"Message types: {[type(m) for m in messages]}")
            logger.error(f"First message repr: {repr(messages[0]) if messages else 'No messages'}")
            # Return a default response instead of raising an error
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="I'm sorry, I couldn't process that request due to a formatting issue."))])

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
    
    def bind(self, **kwargs):
        """Handle LangChain's bind method for function calling."""
        logger = logging.getLogger(__name__)
        logger.debug(f"LLMWrapper.bind called with kwargs: {kwargs}")
        # For now, just return self since we handle function calling differently
        return self
