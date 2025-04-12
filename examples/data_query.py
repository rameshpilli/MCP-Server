import asyncio
import os
from app.core.llm_assistant import LLMAssistant

async def main():
    # Initialize the LLM Assistant
    assistant = LLMAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Example questions
    questions = [
        "How many open jobs do we have today?",
        "What departments have the most job openings?",
        "Show me the sales trends for the last week",
        "How many employees joined in the last month?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = await assistant.answer_question(question)
        print(f"Answer: {answer}")
        print("-" * 80)
    
    # Analyze trends
    print("\nAnalyzing job posting trends:")
    trends = await assistant.analyze_trends("jobs_data", timeframe="1w")
    print(trends)

if __name__ == "__main__":
    asyncio.run(main()) 