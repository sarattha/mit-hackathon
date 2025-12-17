import base64
import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import Runner
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles # Optional if we had assets, but single file is easier with HTMLResponse
from pydantic import BaseModel

from agents import Runner
from my_agents import emotion_agent, quality_agent, tutor_agent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _input_text_message(text: str):
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

class AskResponse(BaseModel):
    response: str
    emotion: dict
    quality: dict


@app.post("/ask", response_model=AskResponse)
async def ask(
    query: str = Form(...),
    transcript: str = Form(""),
    chat_history: str = Form("[]"),  # JSON string
    image: UploadFile = File(...)
):
    try:
        # 1. Process Image
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        content_type = image.content_type or "image/jpeg"
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        image_url = f"data:{content_type};base64,{base64_image}"

        # 2. Emotion Classification
        emotion_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze the emotion in this image."},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ]
        
        # Use async run
        emotion_result = await Runner.run(emotion_agent, emotion_input)
        emotion_data = json.loads(emotion_result.final_output)

        # 3. Question Quality
        quality_result = await Runner.run(quality_agent, _input_text_message(query))
        quality_data = json.loads(quality_result.final_output)

        # 4. Tutor Response
        context = f"""
        Student Question: {query}
        Video Transcript: {transcript}
        Detected Emotion: {emotion_data.get('emotion')} (Confidence: {emotion_data.get('confidence')})
        Question Quality: {quality_data.get('label')} (Score: {quality_data.get('score')})
        Feature: {quality_data.get('feedback')}
        
        Chat History: {chat_history}
        """

        tutor_result = await Runner.run(tutor_agent, _input_text_message(context))
        
        return AskResponse(
            response=tutor_result.final_output,
            emotion=emotion_data,
            quality=quality_data
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return AskResponse(
            response="Sorry, I encountered an error processing your request.",
            emotion={"error": str(e)},
            quality={"error": str(e)}
        )

@app.post("/ask_with_stream")
async def ask_with_stream(
    query: str = Form(...),
    transcript: str = Form(""),
    chat_history: str = Form("[]"),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        content_type = image.content_type or "image/jpeg"
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        image_url = f"data:{content_type};base64,{base64_image}"

        emotion_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze the emotion in this image."},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ]
        emotion_result = await Runner.run(emotion_agent, emotion_input)
        emotion_data = json.loads(emotion_result.final_output)

        quality_result = await Runner.run(quality_agent, _input_text_message(query))
        quality_data = json.loads(quality_result.final_output)

        context = f"""
        Student Question: {query}
        Video Transcript: {transcript}
        Detected Emotion: {emotion_data.get('emotion')} (Confidence: {emotion_data.get('confidence')})
        Question Quality: {quality_data.get('label')} (Score: {quality_data.get('score')})
        Feature: {quality_data.get('feedback')}

        Chat History: {chat_history}
        """
    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)

        async def error_event_generator():
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        return StreamingResponse(error_event_generator(), media_type="application/x-ndjson")

    async def event_generator():
        yield (
            json.dumps({"type": "metadata", "emotion": emotion_data, "quality": quality_data})
            + "\n"
        )

        try:
            stream_result = Runner.run_streamed(tutor_agent, _input_text_message(context))
            async for event in stream_result.stream_events():
                if getattr(event, "type", None) != "raw_response_event":
                    continue

                data = event.data
                if getattr(data, "type", None) == "response.output_text.delta":
                    delta = getattr(data, "delta", "")
                    if delta:
                        yield json.dumps({"type": "token", "content": delta}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
