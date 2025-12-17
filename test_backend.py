import requests
import base64
import os

# Create a minimal white 1x1 JPEG
pixel_base64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
pixel_bytes = base64.b64decode(pixel_base64)

with open("test_image.jpg", "wb") as f:
    f.write(pixel_bytes)

url = "http://localhost:8000/ask"

files = {
    'image': ('test_image.jpg', open('test_image.jpg', 'rb'), 'image/jpeg')
}
data = {
    'query': 'Why is the sky blue?',
    'transcript': 'Talking about atmosphere.',
    'chat_history': '[]'
}

print("Sending request to /ask...")
try:
    response = requests.post(url, data=data, files=files)
    print("Status Code:", response.status_code)
    try:
        json_resp = response.json()
        print("Response Keys:", json_resp.keys())
        print("Emotion Data:", json_resp.get('emotion'))
        print("Quality Data:", json_resp.get('quality'))
        print("Tutor Response:", json_resp.get('response')[:50] + "...")
    except Exception as e:
        print("Failed to parse JSON:", response.text)
except Exception as e:
    print("Request failed (Is server running?):", e)
