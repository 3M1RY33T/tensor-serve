"""
Manual smoke script for OpenAI-compatible API endpoints.

Run this after:
1. Starting the server: uvicorn main:app --reload
2. Configuring an AI endpoint: curl -X POST http://localhost:8000/config/set-ai-endpoint ...
3. Loading a database: curl http://localhost:8000/load?name=...

Usage:
    python test/test_openai_api.py
"""

import json

import requests

BASE_URL = "http://localhost:8000"


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_test(name: str):
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}TEST: {name}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_info(msg: str):
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.END}")


def check_health():
    print_test("Server Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print_success("Server is running")
            print(f"  - DB loaded: {data.get('db_loaded')}")
            print(f"  - AI configured: {data.get('ai_configured')}")
            print(f"  - Active collection: {data.get('active_collection')}")
            return True
        print_error(f"Health check failed: {resp.status_code}")
        return False
    except Exception as e:
        print_error(f"Could not connect to server: {e}")
        return False


def check_v1_models():
    print_test("GET /v1/models (OpenAI-compatible model list)")
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)

        if resp.status_code == 400:
            print_error(f"Model not configured: {resp.json().get('detail')}")
            print_info("Configure with: POST /config/set-ai-endpoint")
            return False

        if resp.status_code == 200:
            data = resp.json()
            print_success("Retrieved model list in OpenAI format")
            print(f"  - Response object type: {data.get('object')}")
            print(f"  - Number of models: {len(data.get('data', []))}")

            if data.get("data"):
                model = data["data"][0]
                print(f"  - First model ID: {model.get('id')}")
                print(f"  - Owner: {model.get('owned_by')}")
                print("\nFull response:")
                print(json.dumps(data, indent=2))
            return True

        print_error(f"Endpoint returned {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def check_v1_chat_completions():
    print_test("POST /v1/chat/completions (OpenAI-compatible chat)")

    try:
        health = requests.get(f"{BASE_URL}/health").json()

        if not health.get("ai_configured"):
            print_error("AI endpoint not configured")
            print_info("Configure with: POST /config/set-ai-endpoint")
            return False

        if not health.get("db_loaded"):
            print_error("Vector database not loaded")
            print_info("Load a database with: GET /load?name=<db_name>")
            return False
    except Exception as e:
        print_error(f"Could not check prerequisites: {e}")
        return False

    try:
        models_resp = requests.get(f"{BASE_URL}/v1/models")
        if models_resp.status_code != 200:
            print_error("Could not retrieve model list")
            return False

        model_id = models_resp.json()["data"][0]["id"]
        print_info(f"Using model: {model_id}")

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
            "temperature": 0.7,
            "max_tokens": 200,
        }

        print(f"  - Sending request with {len(payload['messages'])} messages")
        print(f"  - Temperature: {payload['temperature']}")
        print(f"  - Max tokens: {payload['max_tokens']}")

        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)

        if resp.status_code == 400:
            print_error(f"Bad request: {resp.json().get('detail')}")
            return False

        if resp.status_code == 502:
            print_error(f"AI endpoint error: {resp.json().get('detail')}")
            return False

        if resp.status_code == 200:
            data = resp.json()
            print_success("Received response from OpenAI-compatible endpoint")
            print(f"  - Response ID: {data.get('id')}")
            print(f"  - Object type: {data.get('object')}")
            print(f"  - Model: {data.get('model')}")
            print(f"  - Created: {data.get('created')}")

            if data.get("choices"):
                choice = data["choices"][0]
                print(f"  - Choice 0 finish reason: {choice.get('finish_reason')}")
                print(f"  - Assistant response preview: {choice['message']['content'][:100]}...")

            if data.get("usage"):
                usage = data["usage"]
                print(f"  - Prompt tokens: {usage.get('prompt_tokens')}")
                print(f"  - Completion tokens: {usage.get('completion_tokens')}")
                print(f"  - Total tokens: {usage.get('total_tokens')}")

            print("\nFull response:")
            print(json.dumps(data, indent=2))
            return True

        print_error(f"Endpoint returned {resp.status_code}: {resp.text}")
        return False
    except requests.exceptions.Timeout:
        print_error("Request timed out (AI endpoint may be slow)")
        return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def check_v1_chat_multi_turn():
    print_test("POST /v1/chat/completions (multi-turn conversation)")

    try:
        health = requests.get(f"{BASE_URL}/health").json()
        if not health.get("ai_configured") or not health.get("db_loaded"):
            print_info("Skipping: prerequisites not met")
            return None
    except Exception:
        return None

    try:
        models_resp = requests.get(f"{BASE_URL}/v1/models")
        model_id = models_resp.json()["data"][0]["id"]

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a branch of AI..."},
                {"role": "user", "content": "Can you give me an example?"},
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        }

        print(f"  - Sending multi-turn conversation ({len(payload['messages'])} messages)")

        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            print_success("Multi-turn conversation handled correctly")
            content = data["choices"][0]["message"]["content"]
            print(f"  - Response preview: {content[:100]}...")
            return True

        print_error(f"Request returned {resp.status_code}")
        return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def main():
    print(f"\n{Colors.BLUE}╔{'=' * 58}╗{Colors.END}")
    print(f"{Colors.BLUE}║ OpenAI-Compatible API Test Suite                   ║{Colors.END}")
    print(f"{Colors.BLUE}╚{'=' * 58}╝{Colors.END}")

    print(f"\nBase URL: {BASE_URL}")
    print(f"Docs available at: {BASE_URL}/docs")

    results = {}
    results["health"] = check_health()
    if not results["health"]:
        print_error("\nServer is not reachable. Start the server with:")
        print("  uvicorn main:app --reload")
        return

    results["v1_models"] = check_v1_models()
    results["v1_chat"] = check_v1_chat_completions()
    results["v1_multi_turn"] = check_v1_chat_multi_turn()

    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"Passed:  {Colors.GREEN}{passed}{Colors.END}")
    print(f"Failed:  {Colors.RED}{failed}{Colors.END}")
    if skipped:
        print(f"Skipped: {Colors.YELLOW}{skipped}{Colors.END}")

    if failed == 0 and passed > 0:
        print(f"\n{Colors.GREEN}All tests passed! ✓{Colors.END}")
    elif failed > 0:
        print(f"\n{Colors.RED}Some tests failed.{Colors.END}")

    print()


if __name__ == "__main__":
    main()
