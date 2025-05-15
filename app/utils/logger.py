import logging
import csv
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.config import config

# Set up logging - container-friendly configuration
logger = logging.getLogger("mcp_server")
logger.setLevel(getattr(logging, config.LOG_LEVEL))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Always add a StreamHandler for container environments
stream_handler = logging.StreamHandler(sys.stdout)  # Use stdout explicitly for containers
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Function to initialize S3 logging if configured
def init_s3_logging():
    """Initialize S3 logging if configured"""
    if config.S3_SERVER and config.S3_BUCKET_NAME:
        try:
            import boto3
            from botocore.client import Config
            
            # Create S3 client with custom endpoint
            s3_config = Config(
                s3={'addressing_style': 'path'},
                signature_version='s3v4',
            )
            
            s3_client = boto3.client(
                's3',
                endpoint_url=config.S3_SERVER,
                aws_access_key_id=config.S3_USER,
                aws_secret_access_key=config.S3_PASS,
                verify=config.S3_VERIFY_SSL,
                config=s3_config
            )
            
            logger.info(f"Initialized S3 logging to {config.S3_BUCKET_NAME}/{config.S3_OUTPUT_FOLDER}")
            return s3_client
            
        except ImportError:
            logger.warning("boto3 not installed. S3 logging disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize S3 logging: {str(e)}")
    
    return None

# Initialize S3 client if configured
s3_client = init_s3_logging()

# Local CSV logging - only used outside of Kubernetes
# In Kubernetes, we'll prefer logging to stdout and optionally to S3
if not config.IN_KUBERNETES:
    # Create logs directory if it doesn't exist
    logs_dir = Path(config.LOG_DIR)
    logs_dir.mkdir(exist_ok=True)
    
    # Set up file logger
    log_file = logs_dir / f"mcp_server_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # CSV logging setup
    csv_file = logs_dir / f"mcp_interactions_{datetime.now().strftime('%Y%m%d')}.csv"
    csv_headers = [
        "timestamp",
        "step",
        "message",
        "session_id",
        "tools_used",
        "context_found",
        "response_length",
        "processing_time_ms",
        "status"
    ]
    
    # Initialize CSV file with headers if it doesn't exist
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

def log_to_s3(log_data: Dict[str, Any], log_type: str = "interaction") -> bool:
    """
    Log data to S3 if configured
    
    Args:
        log_data: The data to log
        log_type: The type of log (interaction, error)
        
    Returns:
        bool: True if logged successfully, False otherwise
    """
    if not s3_client:
        return False
        
    try:
        # Add timestamp and pod info
        log_data["timestamp"] = datetime.now().isoformat()
        if config.IN_KUBERNETES:
            log_data["pod_name"] = config.POD_NAME
            log_data["namespace"] = config.NAMESPACE
        
        # Convert to JSON
        log_json = json.dumps(log_data)
        
        # Create S3 key with timestamp and pod name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        pod_suffix = f"-{config.POD_NAME}" if config.IN_KUBERNETES else ""
        
        # Construct the S3 key
        s3_key = f"{config.S3_OUTPUT_FOLDER}/{log_type}-{timestamp}{pod_suffix}.json"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=config.S3_BUCKET_NAME,
            Key=s3_key,
            Body=log_json
        )
        
        return True
    except Exception as e:
        logger.error(f"Failed to log to S3: {str(e)}")
        return False

def log_interaction(
    step: str,
    message: str,
    session_id: Optional[str] = None,
    tools_used: Optional[list] = None,
    context_found: bool = False,
    response_length: int = 0,
    processing_time_ms: float = 0,
    status: str = "success"
) -> None:
    """
    Log an interaction to both log files and S3 if configured
    
    In Kubernetes, logs primarily to stdout and optionally to S3
    In local development, logs to file, stdout, and CSV
    """
    # Format message for logging
    log_message = (
        f"Step: {step} | Message: {message[:100]}{'...' if len(message) > 100 else ''} | "
        f"Session: {session_id or 'none'} | Status: {status}"
    )
    
    # Log to standard logger (will go to stdout and file if configured)
    logger.info(log_message)
    
    # Create structured log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "message": message[:100],  # Truncate long messages
        "session_id": session_id or "none",
        "tools_used": tools_used or [],
        "context_found": context_found,
        "response_length": response_length,
        "processing_time_ms": processing_time_ms,
        "status": status
    }
    
    # In Kubernetes, also log to S3 if configured
    if config.IN_KUBERNETES:
        log_to_s3(log_data, "interaction")
    # In local development, log to CSV
    else:
        csv_file = Path(config.LOG_DIR) / f"mcp_interactions_{datetime.now().strftime('%Y%m%d')}.csv"
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log_data["timestamp"],
                    step,
                    message[:100],  # Truncate long messages
                    session_id or "none",
                    ','.join(tools_used or []),
                    str(context_found),
                    response_length,
                    processing_time_ms,
                    status
                ])
        except Exception as e:
            logger.error(f"Failed to log to CSV: {str(e)}")

def log_error(step: str, error: Exception, session_id: Optional[str] = None) -> None:
    """
    Log an error to both log file and S3/CSV
    
    Args:
        step: The processing step where the error occurred
        error: The exception that was raised
        session_id: Optional session ID for tracking
    """
    error_msg = str(error)
    
    # Log to standard logger (will go to stdout and file if configured)
    logger.error(f"Error in {step} | Error: {error_msg} | Session: {session_id or 'none'}")
    
    # Create structured log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "error": error_msg,
        "error_type": type(error).__name__,
        "session_id": session_id or "none",
        "status": "error"
    }
    
    # In Kubernetes, log to S3 if configured
    if config.IN_KUBERNETES:
        log_to_s3(log_data, "error")
    # In local development, log to CSV
    else:
        csv_file = Path(config.LOG_DIR) / f"mcp_interactions_{datetime.now().strftime('%Y%m%d')}.csv"
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log_data["timestamp"],
                    step,
                    error_msg,
                    session_id or "none",
                    "",  # tools_used
                    "False",  # context_found
                    0,  # response_length
                    0,  # processing_time_ms
                    "error"
                ])
        except Exception as e:
            logger.error(f"Failed to log error to CSV: {str(e)}")

# Helper function for periodic log flushing (used in K8s)
async def flush_logs_periodically(interval_seconds=60):
    """
    Periodically flush logs to S3 if running in Kubernetes
    This is useful for ensuring logs make it to persistent storage
    
    Args:
        interval_seconds: How often to flush logs in seconds
    """
    if not config.IN_KUBERNETES or not s3_client:
        return
        
    import asyncio
    
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            # You could implement batched log flushing here
            # if you want to buffer logs for performance
            logger.info("Periodic log flush")
        except Exception as e:
            logger.error(f"Error in periodic log flush: {str(e)}")

# Initialize logs for Kubernetes
def init_kubernetes_logging():
    """Initialize Kubernetes-specific logging if needed"""
    if config.IN_KUBERNETES:
        # Log startup info
        logger.info(f"Starting in Kubernetes environment")
        logger.info(f"Pod name: {config.POD_NAME}")
        logger.info(f"Namespace: {config.NAMESPACE}")
        
        # Additional Kubernetes-specific logging setup could go here
        # For example, adding JSON formatting for better log aggregation

# Initialize logging based on environment
if config.IN_KUBERNETES:
    init_kubernetes_logging()
else:
    # Initialize CSV file for local development
    logs_dir = Path(config.LOG_DIR)
    logs_dir.mkdir(exist_ok=True)
    
    csv_file = logs_dir / f"mcp_interactions_{datetime.now().strftime('%Y%m%d')}.csv"
    csv_headers = [
        "timestamp",
        "step",
        "message",
        "session_id",
        "tools_used",
        "context_found",
        "response_length",
        "processing_time_ms",
        "status"
    ]
    
    # Initialize CSV file with headers if it doesn't exist
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers) 