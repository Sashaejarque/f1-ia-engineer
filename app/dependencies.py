import os
import logging
from fastapi import HTTPException, Header

logger = logging.getLogger(__name__)


async def verify_internal_secret(x_internal_secret: str = Header(...)) -> str:
    """
    Verify that the provided X-Internal-Secret header matches the configured secret.
    
    Args:
        x_internal_secret: The value from the X-Internal-Secret header
        
    Returns:
        The validated secret
        
    Raises:
        HTTPException: 403 Forbidden if the secret doesn't match
    """
    secret = os.getenv("INTERNAL_SERVICE_SECRET")
    
    
    if not secret:
        raise HTTPException(
            status_code=500,
            detail="Internal service secret not configured"
        )
    
    if x_internal_secret != secret:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials"
        )
    
    return x_internal_secret
