import os
from openai import OpenAI
from typing import Optional, Dict, Any
import re
import dotenv

dotenv.load_dotenv()

# use openai to ask question to get information
def ask_openai(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Optional[Dict[str, Any]]:
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Send request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract answer
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
        }
        
    except Exception as e:
        print(f"❌ fail in ask_openai: {str(e)}")
        return None



def extract_number(text: str) -> Optional[float]:
    """
    Extract numbers from text, supporting multiple formats
    
    Args:
        text: Text containing numbers
    
    Returns:
        Extracted number (float), returns None if not found
    
    Examples:
        >>> extract_number("Price is $3.45 per gallon")
        3.45
        >>> extract_number("Gas price 4.25 dollars")
        4.25
        >>> extract_number("3,456.78")
        3456.78
    """
    try:
        # Remove all whitespace characters
        text = text.strip()
        
        # Try direct conversion to float
        try:
            return float(text)
        except ValueError:
            pass
        
        # Remove currency symbols and other non-numeric characters (keep numbers, decimal points and commas)
        text = re.sub(r'[^\d.,]', '', text)
        
        # Handle different decimal point formats
        if ',' in text and '.' in text:
            # If both comma and decimal point exist, assume comma is thousand separator
            text = text.replace(',', '')
        elif ',' in text:
            # If only comma exists, assume it's decimal point
            text = text.replace(',', '.')
        
        # Extract first number
        match = re.search(r'\d+\.?\d*', text)
        if match:
            return float(match.group())
            
        return None
        
    except Exception:
        return None

def extract_price(text: str, currency: str = "USD") -> Optional[float]:
    """
    从文本中提取价格，支持多种格式
    
    Args:
        text: 包含价格的文本
        currency: 货币类型（默认USD）
    
    Returns:
        提取到的价格（浮点数），如果未找到则返回None
    
    Examples:
        >>> extract_price("价格是$3.45每加仑")
        3.45
        >>> extract_price("油价4.25美元")
        4.25
        >>> extract_price("3,456.78元")
        3456.78
    """
    try:
        # Remove all whitespace characters
        text = text.strip()
        
        # Remove currency symbols
        text = text.replace(currency, '').replace('$', '').replace('¥', '').replace('€', '')
        
        # Use extract_number to extract numbers
        return extract_number(text)
        
    except Exception:
        return None
    
