import asyncio
import json
from app.core.metadata_manager import MetadataManager
from tabulate import tabulate

async def view_metadata():
    metadata_manager = MetadataManager(
        db_url="sqlite:///data/metadata.db"
    )
    
    # Get all data sources
    session = metadata_manager.Session()
    try:
        # Get data sources
        data_sources = session.query(metadata_manager.DataSourceMetadata).all()
        
        print("\n=== Data Sources ===")
        sources_data = []
        for source in data_sources:
            sources_data.append([
                source.id,
                source.name,
                source.type,
                source.description
            ])
        
        print(tabulate(
            sources_data,
            headers=["ID", "Name", "Type", "Description"],
            tablefmt="grid"
        ))
        
        # Get tables for each data source
        print("\n=== Tables by Data Source ===")
        for source in data_sources:
            print(f"\n{source.name}:")
            tables_data = []
            for table in source.tables:
                tables_data.append([
                    table.name,
                    table.topic,
                    table.description,
                    len(table.sample_queries)
                ])
            
            print(tabulate(
                tables_data,
                headers=["Table Name", "Topic", "Description", "Sample Queries"],
                tablefmt="grid"
            ))
            
            # Show detailed column information
            print(f"\nColumn Details for {source.name}:")
            for table in source.tables:
                print(f"\n{table.name} Columns:")
                columns_data = []
                for col_name, col_info in table.columns.items():
                    columns_data.append([
                        col_name,
                        col_info["type"],
                        col_info["description"]
                    ])
                
                print(tabulate(
                    columns_data,
                    headers=["Column", "Type", "Description"],
                    tablefmt="simple"
                ))
                
                print("\nSample Queries:")
                for query in table.sample_queries:
                    print(f"- {query}")
                print("-" * 80)
    
    finally:
        session.close()

if __name__ == "__main__":
    asyncio.run(view_metadata()) 