import asyncio
import sys
import logging
import json
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the classes from the module
from app.tools.clientview_financials import (
    RisersDecliners, 
    ClientValueByProduct, 
    ClientValueByTimePeriod,
    CCYEnum, 
    RegionEnum,
    RDSortingCriteriaEnum
)

async def test_risers_decliners():
    """Test the RisersDecliners API"""
    logger.info("Testing RisersDecliners API...")
    
    # Create and execute a request
    rd = RisersDecliners(
        ccy_code=CCYEnum.usd,
        sorting_criteria=RDSortingCriteriaEnum.top
    )
    
    try:
        result = rd.execute()
        logger.info(f"API call status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            # Use a direct approach to access the first few rows without pandas
            data = result.get('data', [])
            
            logger.info(f"Retrieved {len(data)} rows")
            logger.info("\nSample data (first 3 rows):")
            
            for i, row in enumerate(data[:3]):
                logger.info(f"Row {i+1}: {row}")
            
            # Now try with pandas display method
            logger.info("\nUsing pandas display method:")
            df = rd.display()
            logger.info(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Column names: {df.columns.tolist()}")
        else:
            logger.error(f"API call failed: {result}")
    except Exception as e:
        logger.error(f"Error executing RisersDecliners: {e}")

async def test_client_value_by_product():
    """Test the ClientValueByProduct API"""
    logger.info("\nTesting ClientValueByProduct API...")
    
    # Using Blackrock's CDRID from the sample data
    client_cdrid = 34960  # Blackrock Inc.
    
    # Create and execute a request
    cvbp = ClientValueByProduct(
        ccy_code=CCYEnum.usd,
        client_cdrid=client_cdrid
    )
    
    try:
        result = cvbp.execute()
        logger.info(f"API call status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            # Use a direct approach to access the first few rows without pandas
            data = result.get('data', [])
            
            logger.info(f"Retrieved {len(data)} rows")
            logger.info("\nSample data (first 3 rows):")
            
            for i, row in enumerate(data[:3]):
                logger.info(f"Row {i+1}: {row}")
            
            # Now try with pandas display method
            logger.info("\nUsing pandas display method:")
            df = cvbp.display()
            logger.info(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Column names: {df.columns.tolist()}")
        else:
            logger.error(f"API call failed: {result}")
    except Exception as e:
        logger.error(f"Error executing ClientValueByProduct: {e}")

async def test_client_value_by_time():
    """Test the ClientValueByTimePeriod API"""
    logger.info("\nTesting ClientValueByTimePeriod API...")
    
    # Create and execute a request
    cvbt = ClientValueByTimePeriod(
        ccy_code=CCYEnum.usd,
        time_period_year=2025
    )
    
    try:
        result = cvbt.execute()
        logger.info(f"API call status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            # Print the raw response structure for debugging
            logger.info("\nRaw response structure:")
            logger.info(f"Response keys: {list(result.keys())}")
            
            if 'data' in result:
                data = result['data']
                logger.info(f"Data type: {type(data)}")
                logger.info(f"Data length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    logger.info("\nFirst data item:")
                    first_item = data[0]
                    logger.info(f"Type: {type(first_item)}")
                    
                    if hasattr(first_item, 'keys'):
                        logger.info(f"Keys: {list(first_item.keys())}")
                        
                        # Print first few items directly
                        logger.info("\nSample data (first 3 rows):")
                        for i, row in enumerate(data[:3]):
                            logger.info(f"Row {i+1}: {row}")
                    else:
                        logger.info(f"First item value: {first_item}")
                else:
                    logger.info("Data is empty or not a list")
            else:
                logger.info("No 'data' key in the response")
            
            # Now wrap in a try block the display method call
            try:
                logger.info("\nAttempting to use display method:")
                df = cvbt.display()
                logger.info(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")
                logger.info(f"Column names: {df.columns.tolist()}")
            except Exception as display_error:
                logger.error(f"Display method error: {display_error}")
                
                # Let's try to manually create a DataFrame from the data
                logger.info("\nAttempting manual DataFrame creation:")
                import pandas as pd
                try:
                    if isinstance(data, list) and len(data) > 0:
                        # Manually create the DataFrame
                        table = {
                            'Client Name': [],
                            'Client CDRID': [],
                            'Revenue YTD': []
                        }
                        
                        for elem in data[:5]:  # Just try first 5 items
                            if isinstance(elem, dict):
                                table['Client Name'].append(elem.get("ClientName", "N/A"))
                                table['Client CDRID'].append(elem.get("ClientCDRID", "N/A"))
                                table['Revenue YTD'].append(f"{elem.get('RevenueYTD', 0):.2f}")
                        
                        manual_df = pd.DataFrame(table)
                        logger.info(f"Manual DataFrame created with {len(manual_df)} rows")
                        if not manual_df.empty:
                            logger.info(manual_df.head().to_string())
                except Exception as manual_error:
                    logger.error(f"Manual DataFrame creation error: {manual_error}")
        else:
            logger.error(f"API call failed: {result}")
    except Exception as e:
        logger.error(f"Error executing ClientValueByTimePeriod: {e}")

async def test_interaction_data():
    """Test the interaction data handling in various APIs"""
    logger.info("\nTesting Interaction Data Handling...")
    
    # Test with RisersDecliners as it has the most interaction fields
    rd = RisersDecliners(
        ccy_code=CCYEnum.usd,
        sorting_criteria=RDSortingCriteriaEnum.top
    )
    
    try:
        result = rd.execute()
        logger.info(f"API call status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            data = result.get('data', [])
            
            if data:
                # Get the first client with interaction data
                sample_client = next(
                    (row for row in data if any(
                        row.get(field) is not None 
                        for field in ['InteractionCMOCYTD', 'InteractionGMOCYTD', 'InteractionYTD']
                    )),
                    data[0]  # Fallback to first client if none have interaction data
                )
                
                logger.info("\nSample client interaction data:")
                logger.info(f"Client: {sample_client.get('ClientName', 'N/A')} (CDRID: {sample_client.get('ClientCDRID', 'N/A')})")
                logger.info(f"CMOC Interactions YTD: {sample_client.get('InteractionCMOCYTD', 'None')}")
                logger.info(f"GMOC Interactions YTD: {sample_client.get('InteractionGMOCYTD', 'None')}")
                logger.info(f"Total Interactions YTD: {sample_client.get('InteractionYTD', 'None')}")
                logger.info(f"CMOC Interactions Prev YTD: {sample_client.get('InteractionCMOCPrevYTD', 'None')}")
                logger.info(f"GMOC Interactions Prev YTD: {sample_client.get('InteractionGMOCPrevYTD', 'None')}")
                logger.info(f"Total Interactions Prev YTD: {sample_client.get('InteractionPrevYTD', 'None')}")
                
                # Test the display method with interaction data
                logger.info("\nTesting display method with interaction data:")
                df = rd.display()
                
                # Verify interaction columns are present
                interaction_columns = [
                    'CMOC Interactions YTD',
                    'GMOC Interactions YTD',
                    'Total Interactions YTD',
                    'CMOC Interactions Prev YTD',
                    'GMOC Interactions Prev YTD',
                    'Total Interactions Prev YTD'
                ]
                
                missing_columns = [col for col in interaction_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"Missing interaction columns: {missing_columns}")
                else:
                    logger.info("All interaction columns present in display output")
                    
                # Show sample of interaction data from display
                if not df.empty:
                    logger.info("\nSample interaction data from display:")
                    sample_df = df[['Client Name'] + interaction_columns].head(3)
                    logger.info("\n" + sample_df.to_string())
            else:
                logger.warning("No data returned from API")
        else:
            logger.error(f"API call failed: {result}")
    except Exception as e:
        logger.error(f"Error testing interaction data: {e}")

async def run_all_tests():
    """Run all tests"""
    logger.info("Starting ClientView financials API tests...")
    
    await test_risers_decliners()
    await test_client_value_by_product()
    await test_client_value_by_time()
    await test_interaction_data()  # Add the new interaction data test
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 