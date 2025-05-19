import requests

webhook_url = "https://hooks.zapier.com/hooks/catch/10321312/2rx22ly/"  # Replace with your Zapier webhook URL

# Event details
event_data = {
    "summary": "Team Meeting",
    "start": {
        "dateTime": "2024-11-23T10:00:00",
        "timeZone": "UTC"
    },
    "end": {
        "dateTime": "2024-11-23T11:00:00",
        "timeZone": "UTC"
    },
    "description": "Discuss project updates and next steps.",
    "location": "Zoom Link: https://zoom.us/j/123456789"
}

# Send POST request
response = requests.post(webhook_url, json=event_data)

if response.status_code == 200:
    print("Webhook triggered successfully!")
else:
    print(f"Failed to trigger webhook. Status code: {response.status_code}, Response: {response.text}")
