import requests
from typing import Optional, List
import logging, os, json, traceback

from gptcache import Cache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import manager_factory
from gptcache.embedding import Huggingface
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_community.cache import GPTCache
from langchain_core.language_models.llms import LLM
from langchain.globals import set_llm_cache
from transformers import AutoTokenizer

# Set up logging for debugging - modify to include more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def truncate_prompt(prompt: str, max_length: int = 500) -> str:
    """
    Truncate the prompt to fit within the embedding model's max token length.
    
    Args:
        prompt: The input prompt string
        max_length: Maximum number of tokens (default: 500)
    
    Returns:
        Truncated prompt string
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        encoded = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors=None,
            padding=False
        )
        token_count = len(encoded['input_ids'])
        logger.debug(f"Token count for prompt: {token_count}")
        
        if token_count > max_length:
            logger.warning(f"Token count {token_count} exceeds {max_length}, forcing truncation")
            encoded['input_ids'] = encoded['input_ids'][:max_length]
            token_count = len(encoded['input_ids'])
            logger.debug(f"Token count after truncation: {token_count}")
        
        truncated_prompt = tokenizer.decode(
            encoded['input_ids'],
            skip_special_tokens=False
        )
        logger.debug(f"Truncated prompt: {truncated_prompt[:100]}...")
        return truncated_prompt
    except Exception as e:
        logger.error(f"Error in truncate_prompt: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        # Fallback to simple string slicing if tokenizer fails
        return prompt[:1000]  # Use a larger size for string-based truncation

def init_gptcache(cache_obj: Cache, llm_string: str):
    """
    Initialize GPTCache with Huggingface embeddings, SQLite+FAISS storage, and similarity evaluation.
    
    Args:
        cache_obj: The core gptcache.Cache instance
        llm_string: A unique name per LLM model+params (avoids cache collisions)
    """
    try:
        # Ensure cache directory exists and is writable
        cache_dir = "./.cache_data"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
        if not os.access(cache_dir, os.W_OK):
            raise PermissionError(f"Cache directory {cache_dir} is not writable")


        # Initialize embedding model with error handling
        try:
            hf = Huggingface(model="sentence-transformers/all-MiniLM-L6-v2")
            # Test the embedding function with a simple string
            test_embedding = hf.to_embeddings("Test string")
            logger.debug(f"Embedding test successful, shape: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Create data manager with clear dimension specification
        data_manager = manager_factory(
            manager="sqlite,faiss",
            data_dir=cache_dir,
            vector_params={"dimension": 384},  # Dimension for all-MiniLM-L6-v2
        )
        
        def pre_embedding_with_truncation(data, **params):
            """Wrap get_prompt to truncate the input prompt."""
            try:
                prompt = get_prompt(data, **params)
                if not prompt or not isinstance(prompt, str):
                    logger.error(f"Invalid prompt type: {type(prompt)}")
                    return "" if not prompt else str(prompt)[:1000]
                
                logger.debug(f"Original prompt length: {len(prompt)}")
                truncated = truncate_prompt(prompt)
                logger.debug(f"Truncated prompt length: {len(truncated)}")
                
                if not truncated:
                    logger.error("Truncated prompt is empty")
                    return prompt[:1000]  # Fallback to simple truncation
                    
                return truncated
            except Exception as e:
                logger.error(f"Error in pre_embedding_with_truncation: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                # Return a safe default value
                return str(data)[:1000] if data else ""

        # Initialize cache object with our components
        cache_obj.init(
            pre_embedding_func=pre_embedding_with_truncation,
            embedding_func=hf.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(max_distance=0.3),
        )
        
        # Test cache initialization
        logger.info(f"Cache initialized successfully for {llm_string}")
        
    except Exception as e:
        logger.error(f"Error initializing GPTCache: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Re-raise with more context
        raise RuntimeError(f"Failed to initialize GPTCache: {str(e)}")

# Custom cache wrapper to log save attempts and properly handle errors
class SafeGPTCache(GPTCache):
    def __init__(self, init_gptcache_func):
        super().__init__(init_gptcache_func)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get(self, key):
        try:
            result = super().get(key)
            if result is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit! Total hits: {self.cache_hits}")
            else:
                self.cache_misses += 1
                logger.debug(f"Cache miss. Total misses: {self.cache_misses}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def set(self, key, value, **kwargs):
        try:
            if not key:
                logger.error("Empty key provided to cache")
                return
                
            if not value:
                logger.error("Empty value provided to cache")
                return
                
            # Try to serialize the key and value to catch any JSON serialization issues
            try:
                json.dumps({"key": key, "value": value})
            except (TypeError, OverflowError) as e:
                logger.error(f"Cache data is not JSON serializable: {str(e)}")
                return
                
            logger.debug(f"Attempting to save to cache: key digest={hash(key)}")
            super().set(key, value, **kwargs)
            logger.debug("Successfully saved to cache")
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Don't re-raise the exception - let the application continue

# Initialize global cache with error handling
try:
    cache = Cache()
    safe_cache = SafeGPTCache(init_gptcache)
    set_llm_cache(safe_cache)
    logger.info("Successfully initialized global LLM cache")
except Exception as e:
    logger.error(f"Failed to initialize global cache: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    # Continue without cache rather than crashing
    logger.warning("Continuing without cache functionality")

class LLMSettings(BaseSettings):
    openrouter_api_key: str
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    model_name: str = "mistralai/mistral-7b-instruct:free"
    max_tokens: int = 512
    temperature: float = 0.7

    model_config = SettingsConfigDict(
        env_prefix="RAG_LLM_",
        env_file=".env",
        extra="ignore",
    )

class HuggingFaceAPI(LLM):
    """A minimal LangChain LLM wrapper around the HF Inference API."""

    model_id: str
    api_key: Optional[str]
    max_new_tokens: int
    temperature: float
    top_p: float

    @property
    def _llm_type(self) -> str:
        return "huggingface_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True,
                **({"stop": stop} if stop else {}),
            },
        }

        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)

class OpenRouterAPI(LLM):
    """LangChain LLM wrapper around the OpenRouter Chat API."""

    model_name: str
    api_key: str
    api_base: str
    temperature: float
    max_tokens: int

    @property
    def _llm_type(self) -> str:
        return "openrouter_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if stop:
            data["stop"] = stop

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            if "choices" not in result or not result["choices"]:
                raise ValueError("No choices in response")
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid response format: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Invalid response format: {str(e)}")

class LLMService:
    """Wraps OpenRouterAPI with GPTCache for repeated prompts."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.llm = OpenRouterAPI(
            api_key=settings.openrouter_api_key,
            api_base=settings.openrouter_api_base,
            model_name=settings.model_name,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )

    def generate(self, prompt: str) -> str:
        """Generate text using cached LLM responses."""
        try:
            if not prompt:
                logger.error("Empty prompt provided")
                raise ValueError("Prompt cannot be empty")
                
            # Log prompt with size for debugging
            logger.debug(f"Generating response for prompt (size: {len(prompt)})")
            
            response = self.llm.invoke(prompt)
            
            if not response:
                logger.error("Empty response from LLM")
                raise ValueError("LLM returned empty response")
                
            logger.debug(f"Generated response (size: {len(response)})")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise