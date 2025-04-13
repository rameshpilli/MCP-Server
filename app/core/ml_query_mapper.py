from typing import Dict, Any, Optional, Tuple, List
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from .query_mapper import QueryMapper
import re

@dataclass
class QueryTemplate:
    pattern: str
    sql_template: str
    query_type: str
    examples: List[str]
    parameters: List[str]

class MLQueryMapper(QueryMapper):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.templates = self._init_templates()
        self.template_embeddings = self._compute_template_embeddings()
        
    def _init_templates(self) -> List[QueryTemplate]:
        return [
            QueryTemplate(
                pattern="Show active jobs for today",
                sql_template="""
                    SELECT COUNT(*) as active_jobs
                    FROM jobs
                    WHERE status = 'active'
                    AND DATE(created_at) = CURRENT_DATE()
                """,
                query_type="job_count",
                examples=[
                    "How many active jobs are there today?",
                    "Show me today's active jobs",
                    "Count of active jobs for today",
                    "Current active jobs"
                ],
                parameters=[]
            ),
            QueryTemplate(
                pattern="Show failed jobs for time period",
                sql_template="""
                    SELECT COUNT(*) as failed_jobs
                    FROM jobs
                    WHERE status = 'failed'
                    AND created_at >= DATEADD({period}, -{value}, CURRENT_TIMESTAMP())
                """,
                query_type="job_count",
                examples=[
                    "How many jobs failed in the last 2 hours?",
                    "Show failed jobs from past 3 days",
                    "Count of failed jobs in last week",
                    "Failed job count for past 5 days"
                ],
                parameters=["period", "value"]
            ),
            QueryTemplate(
                pattern="Calculate success rate for time period",
                sql_template="""
                    SELECT 
                        ROUND(100.0 * 
                            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) /
                            COUNT(*), 2) as success_rate
                    FROM jobs
                    WHERE created_at >= DATEADD({period}, -{value}, CURRENT_TIMESTAMP())
                """,
                query_type="success_rate",
                examples=[
                    "What's the success rate for the last 24 hours?",
                    "Show job success rate for past week",
                    "Calculate success percentage for last 3 days",
                    "Job success rate in last 48 hours"
                ],
                parameters=["period", "value"]
            )
        ]

    def _compute_template_embeddings(self) -> Dict[str, torch.Tensor]:
        embeddings = {}
        for template in self.templates:
            # Compute embeddings for all examples
            example_embeddings = self.model.encode(template.examples)
            # Store the mean embedding for the template
            embeddings[template.pattern] = torch.tensor(np.mean(example_embeddings, axis=0))
        return embeddings

    def map_query(self, natural_query: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
        # Get query embedding
        query_embedding = torch.tensor(self.model.encode([natural_query])[0])
        
        # Find best matching template
        best_score = -1
        best_template = None
        
        for template in self.templates:
            template_embedding = self.template_embeddings[template.pattern]
            score = cosine_similarity(
                query_embedding.reshape(1, -1),
                template_embedding.reshape(1, -1)
            )[0][0]
            
            if score > best_score:
                best_score = score
                best_template = template

        # If no good match found, fall back to regex patterns
        if best_score < 0.7:
            return super().map_query(natural_query)

        # Extract parameters if needed
        params = {}
        if best_template.parameters:
            # Extract time period using regex
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

        sql_query = best_template.sql_template
        if params:
            sql_query = sql_query.format(**params)

        return sql_query.strip(), best_template.query_type, params 