import os
from dotenv import load_dotenv
from httpx import Client
from openai import APIConnectionError, AuthenticationError, OpenAI, OpenAIError

# --- Configuration ---
# Load environment variables from a .env file in the same directory
load_dotenv()

# Get API URL and Key from environment variables, with defaults
API_BASE_URL = os.getenv("QWEN_API_URL", "https://wxstudio.thuarchdog.com:60089/v1")
API_KEY = os.getenv("QWEN_API_KEY", "your-api-key-here")  # A default placeholder
# --- !! DEBUGGING FLAG !! ---
# Set to True if you suspect an SSL certificate issue (e.g., self-signed certs)
ALLOW_INSECURE_CONNECTIONS = True

# The model to use for the test
MODEL_NAME = "Qwen3-8B"


class LLMClient:
    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise ValueError(
                "API key not found. Please set QWEN_API_KEY in your .env file."
            )

        # Configure the underlying HTTP client to handle SSL verification
        verify_ssl = not ALLOW_INSECURE_CONNECTIONS

        if ALLOW_INSECURE_CONNECTIONS:
            print(
                "\n⚠️ WARNING: SSL certificate verification is disabled. Using insecure connections."
            )

        # By creating our own Client with explicit Mounts and no proxies,
        # we prevent it from automatically picking up system proxy settings.
        # NOTE: The 'proxies' argument is removed for compatibility with older httpx versions.
        # We will rely on unsetting environment variables if proxy issues persist.
        http_client = Client(verify=verify_ssl, http1=True, http2=False)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            # Add a timeout to the main client as well, given the timeout issues.
            timeout=30.0,
        )
        self.call_count = 0

    def request_llm_output(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        presence_penalty: float = 1.5,
    ) -> str:
        """Sends a request to the LLM and returns the response, with detailed error handling."""
        self.call_count += 1
        try:
            print("\n--- Attempting LLM API Call ---")
            print(f"Model: {MODEL_NAME}")
            print(f"Params: temp={temperature}, max_tokens={max_tokens}")
            completion = self.client.chat.completions.create(  # type: ignore
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Use provided parameters
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
            )
            print("✅ LLM API call successful.")
            return completion.choices[0].message.content or ""
        except AuthenticationError as e:
            print(f"❌ ERROR: Authentication failed. Check your API key. Details: {e}")
        except APIConnectionError as e:
            print(
                f"❌ ERROR: Could not connect to the API server. Check the base URL and your network/VPN. Details: {e}"
            )
        except OpenAIError as e:
            print(f"❌ ERROR: An unexpected API error occurred. Details: {e}")
        except Exception as e:
            print(
                f"❌ ERROR: An unexpected error occurred during the API call. Details: {e}"
            )
        return ""

    def reset_call_count(self):
        self.call_count = 0

    def get_call_count(self):
        return self.call_count


llm_client = LLMClient(api_key=API_KEY, base_url=API_BASE_URL)


def query_llm(prompt: str) -> str:
    """A simple wrapper around the LLM client to make a request."""
    # This is a simplification. In a real scenario, you might have a more complex
    # system for determining the system vs. user prompt.
    return llm_client.request_llm_output(
        system_prompt="You are a helpful assistant.", user_prompt=prompt
    )


if __name__ == "__main__":

    try:
        reply = query_llm(
            "Hello! Please respond with a simple, one-sentence greeting to confirm you are working."
        )
        if reply:
            print("\n--- 4. LLM Response ---")
            print(f"🤖: {reply}")
    except ValueError as e:
        print(f"\n❌ CONFIGURATION ERROR: {e}")

    print("\n" + "=" * 50)
    print("Debugging finished.")
