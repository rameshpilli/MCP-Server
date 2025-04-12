from typing import Dict, Any, List
import openai
from .data_layer import DataLayer

class LLMAssistant:
    def __init__(self, openai_api_key: str):
        self.data_layer = DataLayer()
        openai.api_key = openai_api_key
        
        self.system_prompt = """You are an AI assistant with access to company data through the MCP server.
        Available datasets:
        1. jobs_data: Daily statistics about job postings
           - Fields: job_id, posting_date, department, status
           - Updated: Daily
        
        2. employee_data: HR employee records
           - Fields: employee_id, name, department, role, join_date
        
        3. sales_data: Sales transactions
           - Fields: transaction_id, date, amount, product, customer
           - Updated: Daily
        
        When asked about data:
        1. Identify which dataset(s) to use
        2. Use appropriate filters and aggregations
        3. Present the information clearly
        4. Include relevant trends or patterns
        """
    
    async def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get relevant data context based on the query."""
        context = {}
        
        # Determine which datasets to include based on keywords
        if any(word in query.lower() for word in ['job', 'position', 'opening', 'vacancy']):
            context['jobs'] = await self.data_layer.get_jobs_data()
        
        if any(word in query.lower() for word in ['employee', 'staff', 'team']):
            context['employees'] = await self.data_layer.get_employee_data()
        
        if any(word in query.lower() for word in ['sale', 'revenue', 'transaction']):
            context['sales'] = await self.data_layer.get_sales_data()
        
        return context
    
    def format_context(self, context: Dict[str, Any]) -> str:
        """Format the context data for the LLM prompt."""
        formatted = []
        
        for dataset_name, data in context.items():
            formatted.append(f"\n{dataset_name.upper()} DATA:")
            formatted.append(f"Total records: {data['total_rows']}")
            formatted.append(f"Columns: {', '.join(data['columns'])}")
            formatted.append("\nSample data:")
            
            # Add first 5 records as examples
            for record in data['data'][:5]:
                formatted.append(str(record))
        
        return "\n".join(formatted)
    
    async def answer_question(self, question: str) -> str:
        """Answer a question using available data."""
        # Get relevant data context
        context = await self.get_context_for_query(question)
        formatted_context = self.format_context(context)
        
        # Create the prompt
        prompt = f"""Context:
{formatted_context}

Question: {question}

Please analyze the data and provide a clear, concise answer. Include relevant statistics and trends if applicable."""
        
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    async def analyze_trends(self, dataset: str, timeframe: str = "1w") -> str:
        """Analyze trends in a specific dataset."""
        data = await self.data_layer.get_data(dataset)
        
        prompt = f"""Analyze the following data and identify key trends:
        
Dataset: {dataset}
Timeframe: {timeframe}
Data: {data['data'][:100]}  # Limited to 100 records for context

Please provide:
1. Key metrics and their changes
2. Notable patterns or trends
3. Any anomalies or interesting findings
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content 