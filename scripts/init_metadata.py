import asyncio
import os
from datetime import datetime
from app.core.metadata_manager import MetadataManager

async def init_metadata():
    # Initialize metadata manager with SQLite for development
    # Later can be changed to PostgreSQL or Azure
    metadata_manager = MetadataManager(
        db_url="sqlite:///data/metadata.db"
    )

    # Register Snowflake data source for jobs data
    snowflake_source = await metadata_manager.register_data_source({
        "id": "snowflake_jobs",
        "name": "Snowflake Jobs Database",
        "type": "snowflake",
        "description": "Production jobs tracking database in Snowflake",
        "config": {
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "your_account"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "compute_wh"),
            "database": "jobs_db",
            "schema": "public",
            "role": "reporter"
        }
    })

    # Register jobs tables in Snowflake
    await metadata_manager.register_table({
        "id": "jobs_active",
        "data_source_id": "snowflake_jobs",
        "name": "active_jobs",
        "schema": "public",
        "topic": "jobs",
        "description": "Active jobs tracking table",
        "columns": {
            "job_id": {"type": "string", "description": "Unique job identifier"},
            "status": {"type": "string", "description": "Current job status"},
            "created_at": {"type": "timestamp", "description": "Job creation time"},
            "updated_at": {"type": "timestamp", "description": "Last update time"},
            "type": {"type": "string", "description": "Job type"},
            "priority": {"type": "integer", "description": "Job priority"}
        },
        "sample_queries": [
            "Show me active jobs from today",
            "How many high priority jobs are running?",
            "List failed jobs from last 24 hours"
        ]
    })

    await metadata_manager.register_table({
        "id": "jobs_history",
        "data_source_id": "snowflake_jobs",
        "name": "jobs_history",
        "schema": "public",
        "topic": "jobs",
        "description": "Historical jobs data",
        "columns": {
            "job_id": {"type": "string", "description": "Unique job identifier"},
            "status": {"type": "string", "description": "Final job status"},
            "created_at": {"type": "timestamp", "description": "Job creation time"},
            "completed_at": {"type": "timestamp", "description": "Job completion time"},
            "duration": {"type": "float", "description": "Job duration in seconds"},
            "error": {"type": "string", "description": "Error message if failed"}
        },
        "sample_queries": [
            "What's the average job duration today?",
            "Show me job failure rate by type",
            "List longest running jobs"
        ]
    })

    # Register S3 data source for news data
    s3_source = await metadata_manager.register_data_source({
        "id": "s3_news",
        "name": "Local S3 News Storage",
        "type": "s3",
        "description": "News articles and metadata stored in local S3",
        "config": {
            "endpoint_url": os.getenv("S3_ENDPOINT", "http://localhost:9000"),
            "bucket": "news-data",
            "region": "us-east-1",
            "access_key": os.getenv("S3_ACCESS_KEY", "your_access_key"),
            "secret_key": os.getenv("S3_SECRET_KEY", "your_secret_key")
        }
    })

    # Register news data tables/collections
    await metadata_manager.register_table({
        "id": "news_articles",
        "data_source_id": "s3_news",
        "name": "articles",
        "topic": "news",
        "description": "News articles collection",
        "columns": {
            "article_id": {"type": "string", "description": "Unique article identifier"},
            "title": {"type": "string", "description": "Article title"},
            "content": {"type": "text", "description": "Article content"},
            "published_at": {"type": "timestamp", "description": "Publication date"},
            "source": {"type": "string", "description": "News source"},
            "category": {"type": "string", "description": "Article category"}
        },
        "sample_queries": [
            "Show me today's technology news",
            "Find articles about AI from last week",
            "List most recent financial news"
        ]
    })

    await metadata_manager.register_table({
        "id": "news_analytics",
        "data_source_id": "s3_news",
        "name": "news_analytics",
        "topic": "news",
        "description": "News analytics and metrics",
        "columns": {
            "article_id": {"type": "string", "description": "Article reference"},
            "sentiment_score": {"type": "float", "description": "Article sentiment score"},
            "topic_tags": {"type": "array", "description": "Extracted topic tags"},
            "read_count": {"type": "integer", "description": "Number of reads"},
            "share_count": {"type": "integer", "description": "Number of shares"}
        },
        "sample_queries": [
            "What are the most-read articles today?",
            "Show articles with positive sentiment",
            "List trending topics this week"
        ]
    })

    print("✅ Metadata initialization completed!")
    print("\nRegistered Data Sources:")
    print("1. Snowflake Jobs Database")
    print("   - Tables: active_jobs, jobs_history")
    print("2. S3 News Storage")
    print("   - Collections: articles, news_analytics")

if __name__ == "__main__":
    asyncio.run(init_metadata()) 