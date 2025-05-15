"""
Prompt Templates

This module defines prompt templates that can be used by tools and the main server.
"""

import logging
from app.registry.prompts import registry, PromptTemplate

# Get logger
logger = logging.getLogger('mcp_server.prompts.templates')

# Register summarization prompt
summarization_prompt = PromptTemplate(
    name="document_summarization",
    description="Prompt for summarizing a document",
    template="""
Please provide a concise summary of the following document:

DOCUMENT: {document_content}

The summary should:
1. Be no more than 3-5 sentences
2. Capture the main points and key information
3. Highlight any important details or conclusions
4. Be written in a clear, professional tone

SUMMARY:
""",
    variables=["document_content"]
)
registry.register(summarization_prompt)

# Register sentiment analysis prompt
sentiment_analysis_prompt = PromptTemplate(
    name="sentiment_analysis",
    description="Prompt for analyzing sentiment of text",
    template="""
Analyze the sentiment of the following text:

TEXT: {text}

Provide a sentiment score from -1.0 (very negative) to 1.0 (very positive), and briefly explain your rating.

SENTIMENT ANALYSIS:
""",
    variables=["text"]
)
registry.register(sentiment_analysis_prompt)

# Register entity extraction prompt
entity_extraction_prompt = PromptTemplate(
    name="entity_extraction",
    description="Prompt for extracting entities from text",
    template="""
Extract entities from the following text:

TEXT: {text}

Please identify:
1. People (individuals mentioned by name)
2. Organizations (companies, agencies, institutions)
3. Locations (cities, countries, places)
4. Dates and times
5. Products or services

ENTITIES:
""",
    variables=["text"]
)
registry.register(entity_extraction_prompt)

# Log registered prompt templates
logger.info(f"Registered prompt templates: {', '.join(registry.list_prompts().keys())}") 