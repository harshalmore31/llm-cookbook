import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
# BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def run_single_swarm():

    payload = {
        "name": "Swarms with Function Calling",
        "description": "You are an AI with tools",
        "agents": [
            {
                "agent_name": "AI assistant",
                "description": "You are an helpful assistant",
                "system_prompt": "You are a helpful assistant.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
            },
        ],
        "max_loops": 2,
        "swarm_type": "ConcurrentWorkflow",
        "task": "what is weather in Boston !",
        "output_type": "dict"
    }

    print("Sending swarm request...")
    response = requests.post(f"{BASE_URL}/v1/swarm/completions", headers=headers, json=payload)
    print("Status Code:", response.status_code)
    try:
        result = response.json()
    except Exception as ex:
        print("Error parsing JSON response:", ex)
        print("Response Text:", response.text)
        result = None
    return result

# def get_logs():
#     response = requests.get(f"{BASE_URL}/v1/swarm/logs", headers=headers)
#     try:
#         return response.json()
#     except Exception as ex:
#         print("Error parsing logs JSON:", ex)
#         return None

if __name__ == "__main__":
    # Run health check
    health = run_health_check()
    print("Health Check Response:")
    print(json.dumps(health, indent=4))
    
    # Run a single swarm with RAG-enabled agents
    swarm_result = run_single_swarm()
    print("Swarm Result:")
    print(json.dumps(swarm_result, indent=4))
    
    # Retrieve and print API logs
    # logs = get_logs()
    # print("Logs:")
    # print(json.dumps(logs, indent=4))
