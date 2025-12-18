import base64
import json
import logging
import re
from typing import Optional

from agents import Runner
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from my_agents import emotion_agent, quality_agent, relevance_agent, tutor_agent

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


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.*)$", re.DOTALL)


def _image_url_from_upload_or_base64(
    *,
    image_base64: Optional[str],
) -> str:
    if image_base64:
        match = _DATA_URL_RE.match(image_base64.strip())
        if match:
            mime = match.group("mime") or "image/jpeg"
            b64 = match.group("b64")
        else:
            mime = "image/jpeg"
            b64 = image_base64.strip()

        try:
            base64.b64decode(b64, validate=False)
        except Exception as e:
            raise ValueError("Invalid base64 image data") from e

        return f"data:{mime};base64,{b64}"

    raise ValueError("Missing image: provide 'image_base64' or an 'image' upload")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f:
        return f.read()


class AskResponse(BaseModel):
    response: str
    emotion: dict
    quality: dict
    relevance: dict


@app.post("/ask", response_model=AskResponse)
async def ask(
    query: str = Form(...),
    transcript: str = Form(""),
    chat_history: str = Form("[]"),  # JSON string
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    try:
        # 1. Process Image
        if image is not None:
            # Ensure we use the async read API to avoid sync I/O in the event loop.
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_type = image.content_type or "image/jpeg"
            if not content_type.startswith("image/"):
                content_type = "image/jpeg"
            image_url = f"data:{content_type};base64,{base64_image}"
        else:
            image_url = _image_url_from_upload_or_base64(image_base64=image_base64)

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

        # 4. Lesson Relevance Guardrail (skip if transcript is missing)
        if transcript.strip():
            relevance_prompt = f"""
Student Question:
{query}

Lesson Transcript:
{transcript}

Recent Chat History (JSON string):
{chat_history}
""".strip()

            relevance_result = await Runner.run(relevance_agent, _input_text_message(relevance_prompt))
            relevance_data = json.loads(relevance_result.final_output)
        else:
            relevance_data = {
                "in_scope": True,
                "confidence": 0.0,
                "reason": "No transcript provided; skipping lesson scope check.",
                "suggested_questions": [],
            }

        in_scope = bool(relevance_data.get("in_scope", True))
        confidence = float(relevance_data.get("confidence", 0.0) or 0.0)
        should_block = (not in_scope) and confidence >= 0.6

        if should_block:
            suggestions = relevance_data.get("suggested_questions") or []
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n\nTry one of these instead:\n- " + "\n- ".join(str(s) for s in suggestions[:4])

            return AskResponse(
                response=(
                    "I can help with questions related to the current lesson. "
                    f"Your question appears out of scope: {relevance_data.get('reason', 'Out of scope.')}"
                    f"{suggestion_text}"
                ),
                emotion=emotion_data,
                quality=quality_data,
                relevance=relevance_data,
            )

        # 5. Tutor Response
        context = f"""
        Student Question: {query}
        Video Transcript: {transcript}
        Detected Emotion: {emotion_data.get("emotion")} (Confidence: {emotion_data.get("confidence")})
        Question Quality: {quality_data.get("label")} (Score: {quality_data.get("score")})
        Feature: {quality_data.get("feedback")}
        Lesson Relevance: {relevance_data}
        
        Chat History: {chat_history}
        """

        tutor_result = await Runner.run(tutor_agent, _input_text_message(context))

        return AskResponse(
            response=tutor_result.final_output,
            emotion=emotion_data,
            quality=quality_data,
            relevance=relevance_data,
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return AskResponse(
            response="Sorry, I encountered an error processing your request.",
            emotion={"error": str(e)},
            quality={"error": str(e)},
            relevance={"error": str(e)},
        )


@app.post("/ask_with_stream")
async def ask_with_stream(
    query: str = Form(...),
    transcript: str = Form(""),
    chat_history: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    try:
        if image is not None:
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_type = image.content_type or "image/jpeg"
            if not content_type.startswith("image/"):
                content_type = "image/jpeg"
            image_url = f"data:{content_type};base64,{base64_image}"
        else:
            image_url = _image_url_from_upload_or_base64(image_base64=image_base64)

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

        if transcript.strip():
            relevance_prompt = f"""
Student Question:
{query}

Lesson Transcript:
{transcript}

Recent Chat History (JSON string):
{chat_history}
""".strip()

            relevance_result = await Runner.run(relevance_agent, _input_text_message(relevance_prompt))
            relevance_data = json.loads(relevance_result.final_output)
        else:
            relevance_data = {
                "in_scope": True,
                "confidence": 0.0,
                "reason": "No transcript provided; skipping lesson scope check.",
                "suggested_questions": [],
            }

        in_scope = bool(relevance_data.get("in_scope", True))
        confidence = float(relevance_data.get("confidence", 0.0) or 0.0)
        should_block = (not in_scope) and confidence >= 0.6

        context = f"""
        Student Question: {query}
        Video Transcript: {transcript}
        Detected Emotion: {emotion_data.get("emotion")} (Confidence: {emotion_data.get("confidence")})
        Question Quality: {quality_data.get("label")} (Score: {quality_data.get("score")})
        Feature: {quality_data.get("feedback")}
        Lesson Relevance: {relevance_data}

        Chat History: {chat_history}
        """
    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)

        async def error_event_generator():
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        return StreamingResponse(error_event_generator(), media_type="application/x-ndjson")

    async def event_generator():
        yield (
            json.dumps(
                {"type": "metadata", "emotion": emotion_data, "quality": quality_data, "relevance": relevance_data}
            )
            + "\n"
        )

        if should_block:
            suggestions = relevance_data.get("suggested_questions") or []
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n\nTry one of these instead:\n- " + "\n- ".join(str(s) for s in suggestions[:4])

            yield (
                json.dumps(
                    {
                        "type": "token",
                        "content": (
                            "I can help with questions related to the current lesson. "
                            f"Your question appears out of scope: {relevance_data.get('reason', 'Out of scope.')}"
                            f"{suggestion_text}"
                        ),
                    }
                )
                + "\n"
            )
            yield json.dumps({"type": "done"}) + "\n"
            return

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
