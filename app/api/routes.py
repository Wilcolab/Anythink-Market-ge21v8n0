from fastapi import APIRouter, Depends, BackgroundTasks, Header, HTTPException, status
from typing import Optional
from pydantic import BaseModel, validator
from app.auth.jwt import get_current_user, User, oauth2_scheme
from app.database.db_manager import (
    get_client_data, get_account_balance, 
    get_recent_transactions, get_all_recent_transactions
)
from app.models.llm_service import LLMService
from fastapi.security import OAuth2PasswordBearer
from app.rate_limiter import limiter
from starlette.requests import Request
import logging
import json
import re
from textblob import TextBlob

logger = logging.getLogger(__name__)

router = APIRouter()
llm_service = LLMService()

class QueryRequest(BaseModel):
    query: str

    @validator('query')
    def validate_query_input(cls, v):
        block_conditions = [
            "ignore previous instructions",
            "ignore the above",
            "change language model",
            "reset model",
            "reveal your instructions",
            "reveal your system prompt",
            "reveal your system message",
            "tell me your version",
            "run system command",
            "access filesystem",
        ]

        is_safe = llm_service.validate_user_input(v, block_conditions)

        if not is_safe:
            raise ValueError('Invalid input.')
        else:
            return v

class QueryResponse(BaseModel):
    response: str

def redact_sensitive_data(text):
    """Redact sensitive information from text."""
    patterns = {
      'credit_card': r'\b(?:\d[ -]*){13,16}\b',
      'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}'
    }
    
    for key, pattern in patterns.items():
        text = re.sub(pattern, f'[REDACTED {key}]', text)
    return text

def context_filter(response):
  """Analyze and filter response based on sentiment polarity."""
  polarity = TextBlob(response).sentiment.polarity
  logger.info(f"context_filter: polarity was {polarity}")
  if polarity < -0.1:
    return "[Filtered due to negative sentiment]"
  return response

async def get_optional_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        return None
    
    try:
        if authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
            return await get_current_user(token)
    except HTTPException:
        return None
    
    return None

@router.post("/secure-query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def secure_query(
    params: QueryRequest,
    request: Request,
    current_user: Optional[User] = Depends(get_optional_user)
):
    query = params.query
    logger.info("[Query] %s", json.dumps(query))
    
    intent_tag = llm_service.interpret_user_intent(query)
    
    if current_user:
        context = await get_context_for_intent(intent_tag, current_user.username)
    else:
        context = await get_context_for_intent(intent_tag, None)
    
    response = llm_service.generate_response(query, context)
    logger.info("[Response] %s", json.dumps(response))

    tmp = context_filter(response)
    if response != tmp:
        response = tmp
        logger.info("[Response Filtered] %s", json.dumps(response))

    response = redact_sensitive_data(response)
    
    return QueryResponse(response=response)

async def get_context_for_intent(intent_tag: str, username: str = None) -> str:
    if intent_tag == "account_balance":
        if not username:
            return "Account information is only available for authenticated users."
            
        accounts = await get_account_balance(username)
        if accounts:
            accounts_info = "\n".join([f"{account['account_type'].capitalize()} account {account['account_number']}: ${account['balance']:.2f}" for account in accounts])
            return f"Account Information:\n{accounts_info}"
        else:
            return "No account information available."
            
    elif intent_tag in ["transaction_history", "spending_analysis"]:
        if username:
            transactions = await get_recent_transactions(username)
            if transactions:
                trans_info = "\n".join([f"{t['transaction_date'].split()[0]} - {t['description']} - ${t['amount']:.2f} ({t['transaction_type']})" for t in transactions])
                return f"Recent Transactions for {username}:\n{trans_info}"
            else:
                return "No recent transactions found for this user."
        else:
            # For anonymous users, return all recent transactions in the system
            transactions = await get_all_recent_transactions(5)
            if transactions:
                trans_info = "\n".join([f"{t['transaction_date'].split()[0]} - {t['username']} - {t['description']} - ${t['amount']:.2f} ({t['transaction_type']})" for t in transactions])
                return f"Recent Transactions in the system:\n{trans_info}"
            else:
                return "No recent transactions found in the system."
            
    else:
        data_items = await get_client_data(intent_tag) if intent_tag else []
        return "\n\n".join([item['info'] for item in data_items]) if data_items else ""

@router.get("/users/me")
async def get_current_user_info(current_user: Optional[User] = Depends(get_optional_user)):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
