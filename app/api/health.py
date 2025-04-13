from fastapi import APIRouter, Depends
from app.core.config import Settings
from app.core.startup import StartupValidator
from app.core.dependencies import get_settings

router = APIRouter()

@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint"""
    if settings.TESTING:
        return {"status": "healthy", "mode": "test"}
        
    validator = StartupValidator(settings)
    checks = await validator.run_all_checks()
    
    status = "healthy" if all(result[1] for result in checks) else "unhealthy"
    failed_checks = [check[0] for check in checks if not check[1]]
    
    return {
        "status": status,
        "mode": "production",
        "failed_checks": failed_checks if failed_checks else None
    } 