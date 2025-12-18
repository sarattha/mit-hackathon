import base64

import requests

# Create a minimal white 1x1 JPEG
pixel_base64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
pixel_bytes = base64.b64decode(pixel_base64)

with open("test_image.jpg", "wb") as f:
    f.write(pixel_bytes)

url = "http://localhost:8000/ask"

data = {
    "query": "Why is the sky blue?",
    "transcript": "Talking about atmosphere.",
    "chat_history": "[]",
    # Send as base64 data URL (preferred)
    "image_base64": f"data:image/jpeg;base64,{pixel_base64}",
}

print("Sending request to /ask...")
try:
    response = requests.post(url, data=data)
    print("Status Code:", response.status_code)
    try:
        json_resp = response.json()
        print("Response Keys:", json_resp.keys())
        print("Emotion Data:", json_resp.get("emotion"))
        print("Quality Data:", json_resp.get("quality"))
        print("Tutor Response:", json_resp.get("response")[:50] + "...")
    except Exception:
        print("Failed to parse JSON:", response.text)
except Exception as e:
    print("Request failed (Is server running?):", e)
