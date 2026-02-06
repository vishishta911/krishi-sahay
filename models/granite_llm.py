"""
IBM Watsonx Granite LLM client for agricultural Q&A.
Provides secure token management, chat completion, and error handling.
"""

import os
import sys
import requests
import time
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============ Configuration Constants ============
REQUEST_TIMEOUT = 30  # Seconds for API requests
REQUEST_RETRIES = 1  # Number of retries on failure
TOKEN_REFRESH_BUFFER = 300  # Refresh token 5 minutes before expiry

# API Endpoints
IAM_AUTH_URL = "https://iam.cloud.ibm.com/identity/token"
WATSONX_API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation"


def validate_env_variables() -> Dict[str, str]:
    """
    Validate all required environment variables on startup.
    
    Returns:
        Dictionary with validated environment variables
        
    Raises:
        ValueError: If any required variable is missing
    """
    required_vars = {
        'WATSONX_API_KEY': 'IBM Cloud API Key',
        'WATSONX_PROJECT_ID': 'Watsonx Project ID'
    }
    
    optional_vars = {
        'WATSONX_MODEL_ID': 'ibm/granite-3-8b-instruct',
        'WATSONX_REGION': 'us-south'
    }
    
    # Collect validated variables
    env_vars = {}
    missing_vars = []
    
    # Check required variables
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(f"  • {var_name} ({description})")
        else:
            env_vars[var_name] = value
    
    # Add optional variables with defaults
    for var_name, default_value in optional_vars.items():
        env_vars[var_name] = os.getenv(var_name, default_value)
    
    # Raise if any required variables are missing
    if missing_vars:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing_vars)
        raise ValueError(error_msg)
    
    return env_vars


class GraniteLLMClient:
    """
    Secure IBM Watsonx Granite LLM client for agricultural Q&A.
    Handles token management, prompt generation, and API communication.
    """
    
    def __init__(self):
        """
        Initialize Granite LLM client with validated credentials.
        
        Raises:
            ValueError: If required environment variables are missing
        """
        # Validate and load environment variables
        self.env_vars = validate_env_variables()
        
        self.api_key = self.env_vars['WATSONX_API_KEY']
        self.project_id = self.env_vars['WATSONX_PROJECT_ID']
        self.model_id = self.env_vars['WATSONX_MODEL_ID']
        self.region = self.env_vars['WATSONX_REGION']
        
        # Token management (private to prevent accidental logging)
        self._access_token = None
        self._token_expiry = None
        
        print(f"✓ Granite LLM client initialized")
        print(f"  Model: {self.model_id}")
        print(f"  Region: {self.region}")
    
    # ============ Token Management ============
    
    def _fetch_iam_token(self) -> str:
        """
        Fetch a new IAM access token from IBM Cloud.
        Private method - tokens are never logged or printed.
        
        Returns:
            Access token string
            
        Raises:
            ValueError: If authentication fails
        """
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        data = {
            'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
            'apikey': self.api_key,
            'response_type': 'cloud_iam'
        }
        
        try:
            response = requests.post(
                IAM_AUTH_URL,
                headers=headers,
                data=data,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            if 'access_token' not in token_data:
                raise ValueError("Invalid IAM response: missing access token")
            
            # Store token and expiry time
            self._access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            print(f"✓ Token obtained (expires in {expires_in}s)")
            return self._access_token
            
        except requests.exceptions.Timeout:
            raise ValueError(f"Token request timed out (>{REQUEST_TIMEOUT}s)")
        except requests.exceptions.HTTPError:
            raise ValueError("Authentication failed: check your API key")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Token request failed: {str(e)}")
    
    def _ensure_valid_token(self) -> str:
        """
        Ensure access token is valid, refresh if necessary.
        
        Returns:
            Valid access token
        """
        # Get fresh token if none exists
        if self._access_token is None:
            return self._fetch_iam_token()
        
        # Check if token is expiring soon
        if self._token_expiry is not None:
            time_remaining = (self._token_expiry - datetime.now()).total_seconds()
            if time_remaining < TOKEN_REFRESH_BUFFER:
                print("Token expiring soon, refreshing...")
                return self._fetch_iam_token()
        
        return self._access_token
    
    # ============ Prompt Formatting ============
    
    def _build_prompt(self, user_query: str, context_answers: Optional[List[Dict]] = None) -> str:
        """
        Build a clean, structured prompt with bullet-point context.
        
        Args:
            user_query: The user's agricultural question
            context_answers: List of context answer dicts with 'question' and 'answer' keys
            
        Returns:
            Formatted prompt string
        """
        prompt = "You are a helpful agricultural assistant for farmers.\n\n"
        
        # Add context from retrieved answers (if available)
        if context_answers and len(context_answers) > 0:
            prompt += "RELEVANT KNOWLEDGE:\n"
            for i, answer in enumerate(context_answers, 1):
                prompt += f"\n{i}. Question: {answer.get('question', 'N/A')}\n"
                prompt += f"   Answer: {answer.get('answer', 'N/A')}\n"
            prompt += "\n"
        
        # Add user's query
        prompt += f"USER QUESTION:\n{user_query}\n\n"
        prompt += "Please provide a clear, helpful answer based on the knowledge provided above."
        
        return prompt
    
    # ============ API Request ============
    
    def _call_chat_completion(self, prompt: str,
                             max_tokens: int = 256,
                             temperature: float = 0.7,
                             attempt: int = 1) -> str:
        """
        Call Granite LLM chat completion API with retry logic.
        Private method to separate API logic from high-level interface.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            attempt: Current attempt number (for retry control)
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If request fails after retries
        """
        # Get valid token
        token = self._ensure_valid_token()
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model_id': self.model_id,
            'input': prompt,
            'parameters': {
                'decoding_method': 'greedy',
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            },
            'project_id': self.project_id
        }
        
        try:
            print(f"Calling {self.model_id}...")
            
            response = requests.post(
                WATSONX_API_URL,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            if 'results' not in response_data or len(response_data['results']) == 0:
                raise ValueError("API returned empty results")
            
            generated_text = response_data['results'][0]['generated_text'].strip()
            print(f"✓ Response received ({len(generated_text)} characters)")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            error = "API request timed out"
            if attempt < REQUEST_RETRIES + 1:
                print(f"⚠️ {error}, retrying...")
                time.sleep(1)
                return self._call_chat_completion(
                    prompt, max_tokens, temperature, attempt + 1
                )
            raise ValueError(f"{error} (after {REQUEST_RETRIES} retry)")
            
        except requests.exceptions.HTTPError as e:
            error = f"API error ({response.status_code})"
            if attempt < REQUEST_RETRIES + 1:
                print(f"⚠️ {error}, retrying...")
                time.sleep(1)
                return self._call_chat_completion(
                    prompt, max_tokens, temperature, attempt + 1
                )
            raise ValueError(error)
            
        except requests.exceptions.RequestException as e:
            error = "API connection failed"
            if attempt < REQUEST_RETRIES + 1:
                print(f"⚠️ {error}, retrying...")
                time.sleep(1)
                return self._call_chat_completion(
                    prompt, max_tokens, temperature, attempt + 1
                )
            raise ValueError(error)
    
    # ============ Public Interface ============
    
    def generate_answer(self, user_query: str,
                       context_answers: Optional[List[Dict]] = None,
                       max_tokens: int = 256,
                       temperature: float = 0.7) -> Dict:
        """
        Generate an answer to a user query with optional context.
        
        Args:
            user_query: The user's agricultural question
            context_answers: List of relevant Q&A from knowledge base
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response and metadata:
            {
                'success': bool,
                'answer': str (only if success),
                'error': str (only if failed),
                'query': str,
                'model': str,
                'timestamp': str
            }
        """
        try:
            # Validate input
            if not user_query or not isinstance(user_query, str):
                return {
                    'success': False,
                    'error': 'User query must be a non-empty string',
                    'query': str(user_query),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Build prompt with context
            prompt = self._build_prompt(user_query, context_answers)
            
            # Call API
            response_text = self._call_chat_completion(prompt, max_tokens, temperature)
            
            # Clean response
            answer = ' '.join(response_text.split())
            
            return {
                'success': True,
                'answer': answer,
                'query': user_query,
                'model': self.model_id,
                'context_used': context_answers is not None and len(context_answers) > 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            # User-friendly error message
            return {
                'success': False,
                'error': f"Could not generate answer: {str(e)}",
                'query': user_query,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            # Catch unexpected errors
            return {
                'success': False,
                'error': "An unexpected error occurred while generating the answer",
                'query': user_query,
                'timestamp': datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict:
        """
        Check if client is properly configured and can authenticate.
        
        Returns:
            Dictionary with health status
        """
        try:
            self._ensure_valid_token()
            return {
                'status': 'healthy',
                'model': self.model_id,
                'authenticated': True
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'authenticated': False
            }


def create_client() -> GraniteLLMClient:
    """
    Create and initialize a Granite LLM client with validation.
    
    Returns:
        Initialized GraniteLLMClient instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        client = GraniteLLMClient()
        
        # Verify health
        health = client.health_check()
        if health['status'] == 'healthy':
            print("✓ Client health check passed")
        else:
            print(f"✗ Health check failed: {health.get('error')}")
        
        return client
        
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        raise
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        raise


def main():
    """Main entry point for testing the Granite LLM client."""
    print("=" * 70)
    print("GRANITE LLM CLIENT TEST")
    print("=" * 70)
    
    try:
        # Create client
        client = create_client()
        
        # Test with sample context and query
        print("\n" + "=" * 70)
        print("SAMPLE INFERENCE")
        print("=" * 70)
        
        sample_context = [
            {
                'question': 'What is crop rotation?',
                'answer': 'Crop rotation is the practice of growing different crops in the same area in sequential seasons. It improves soil health, reduces pest buildup, and maintains soil fertility.'
            }
        ]
        
        sample_query = "Why is crop rotation important?"
        
        print(f"\nQuery: {sample_query}")
        print(f"Context: {len(sample_context)} reference answer(s)")
        
        response = client.generate_answer(sample_query, sample_context)
        
        if response['success']:
            print(f"\n✓ Response received:")
            print(f"Answer: {response['answer']}")
        else:
            print(f"\n✗ Error: {response['error']}")
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
