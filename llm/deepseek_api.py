import os
from openai import OpenAI, APIError, RateLimitError
import time
import logging
from .base_llm import BaseLLM

logger = logging.getLogger(__name__)

class DeepSeekAPI(BaseLLM):
    """Client for interacting with the DeepSeek API (OpenAI compatible)."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        if not api_key:
            raise ValueError("DeepSeek API key is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.messages = [] # Initialize messages here as required by BaseLLM structure
        
    def get_models(self):
        """Placeholder for get_models. DeepSeek API might not have a direct equivalent easily accessible via basic client."""
        # In a real scenario, you might call an endpoint or return a fixed list
        logger.warning("get_models() not implemented for DeepSeekAPI, returning defaults.")
        return ["deepseek-chat", "deepseek-reasoner"]
        
    def add_message(self, role: str, content: str):
        """Adds a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        
    def add_messages(self, messages: list):
        """Adds multiple messages to the conversation history."""
        self.messages.extend(messages)
        
    def get_messages(self) -> list:
        """Returns the current conversation history."""
        return self.messages
        
    def reset_messages(self):
        """Clears the conversation history."""
        self.messages = []
        
    def generate_text(self, model_name: str = "deepseek-chat", temperature: float = 0.7, max_retries: int = 3, retry_delay: int = 5, **kwargs) -> str:
        """Generates text using the specified DeepSeek model."""
        
        # Filter out unsupported kwargs for deepseek-reasoner if used
        # Note: deepseek-chat might support these, but we filter based on reasoner limitations for safety
        # since the user initially asked for reasoner.
        # Specifically, temperature, top_p, presence_penalty, frequency_penalty are ignored by reasoner.
        # response_format is explicitly not supported by reasoner. 
        # We will rely on the prompt to ask for JSON with deepseek-chat.
        
        api_kwargs = {
            "model": model_name,
            "messages": self.messages,
            "stream": False, # Non-streaming for this implementation
            # Add other supported params if needed, filtering based on model if necessary
        }
        
        # Add temperature only if model is not reasoner (or maybe it's ignored? Add cautiously)
        if model_name != "deepseek-reasoner":
             api_kwargs["temperature"] = temperature
             # Potentially add other compatible params like max_tokens here from kwargs
             if 'max_tokens' in kwargs:
                 api_kwargs['max_tokens'] = kwargs['max_tokens']
        else:
            # deepseek-reasoner specific params (if any) - max_tokens is mentioned in docs
             if 'max_tokens' in kwargs:
                 api_kwargs['max_tokens'] = kwargs['max_tokens'] # Max 8k for final answer
             pass # Add reasoner-specific params here if they exist

        logger.debug(f"Sending request to DeepSeek with model={model_name}")
        logger.debug(f"Messages count: {len(self.messages)}")
        if self.messages:
            logger.debug(f"First message role: {self.messages[0].get('role')}")
            
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**api_kwargs)
                
                # Handle potential variations in response structure
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    # For deepseek-reasoner, log the internal CoT if present
                    if model_name == "deepseek-reasoner" and hasattr(response.choices[0].message, 'reasoning_content'):
                         reasoning = response.choices[0].message.reasoning_content
                         logger.info(f"DeepSeek Reasoner CoT: {reasoning[:200]}...") # Log beginning of CoT
                    
                    if content:
                        return content.strip()
                    else:
                        logger.warning("DeepSeek API returned an empty message content.")
                        return "" # Return empty string for empty content
                else:
                    logger.error(f"Unexpected DeepSeek API response structure: {response}")
                    return "Error: Unexpected API response structure."
                    
            except RateLimitError as e:
                logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            except APIError as e:
                logger.error(f"DeepSeek API error: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                
        logger.error(f"Failed to get response from DeepSeek API after {max_retries} attempts.")
        return "Error: Max retries exceeded."

    def prompt_template(self, role: str, prompt: str):
        """Formats a prompt into the message structure expected by the API."""
        return {"role": role, "content": prompt} 