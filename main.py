import base64
import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any

from agents import Runner
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from my_agents import (
    followup_rephrase_agent,
    image_decider_agent,
    generate_image_base64,
    quality_agent,
    relevance_agent,
    vision_tutor_agent,
)

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

_VISION_TUTOR_MAX_GUIDANCE_TURNS = int(os.getenv("VISION_TUTOR_MAX_GUIDANCE_TURNS", "3"))


def _input_text_message(text: str):
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


def _vision_input_message(text: str, image_url: str | None):
    if not image_url:
        return _input_text_message(text)

    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": image_url, "detail": "auto"},
            ],
        }
    ]


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.*)$", re.DOTALL)


def _image_url_from_upload_or_base64(
    *,
    image_base64: str | None,
) -> str | None:
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

    return None


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _safe_json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default

    if isinstance(value, (dict, list)):
        return value

    text = str(value).strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        body = (fenced.group("body") or "").strip()
        if body:
            try:
                return json.loads(body)
            except Exception:
                pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if 0 <= start_obj < end_obj:
        candidate = text[start_obj : end_obj + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if 0 <= start_arr < end_arr:
        candidate = text[start_arr : end_arr + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return default


def _parse_chat_history(chat_history: str) -> list[Any]:
    parsed = _safe_json_loads(chat_history, default=[])
    if isinstance(parsed, list):
        return parsed
    return []


def _count_assistant_turns(history: list[Any]) -> int:
    count = 0
    for item in history:
        if isinstance(item, Mapping):
            role = str(item.get("role") or item.get("sender") or "").strip().lower()
            if role == "assistant" or role == "tutor":
                count += 1
    return count


def _extract_image_data_url_from_run_result(run_result: Any) -> str | None:
    new_items = getattr(run_result, "new_items", None) or []
    for item in new_items:
        if getattr(item, "type", None) != "tool_call_item":
            continue

        raw_call = getattr(item, "raw_item", None)
        call_type = _get_field(raw_call, "type")
        if call_type != "image_generation_call":
            continue

        img_result = _get_field(raw_call, "result")
        if isinstance(img_result, str) and img_result:
            return f"data:image/png;base64,{img_result}"

    return None


def _should_rewrite_query(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False
    if normalized in {
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "no",
        "nope",
        "maybe",
        "this",
        "that",
        "this one",
        "that one",
        "i think so",
        "not sure",
        "idk",
        "i don't know",
        "same",
    }:
        return True
    if len(normalized) < 12 and "?" not in normalized:
        return True
    return False


async def _rewrite_followup_query(*, query: str, transcript: str, chat_history: str) -> str:
    if not _should_rewrite_query(query):
        return query

    prompt = f"""
Raw student input:
{query}

Lesson transcript/context:
{transcript}

Recent chat history (JSON string):
{chat_history}
""".strip()

    try:
        result = await Runner.run(followup_rephrase_agent, _input_text_message(prompt))
        data = json.loads(result.final_output)
        rewritten = (data.get("rewritten_query") or "").strip()
        if rewritten:
            return rewritten
    except Exception:
        # Fallback to original query on any rewrite error.
        pass

    return query


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f:
        return f.read()


class AskResponse(BaseModel):
    response: str
    quality: dict
    relevance: dict
    generated_image: str | None = None
    generated_image_caption: str | None = None


@app.post("/ask", response_model=AskResponse)
async def ask(
    query: str = Form(...),
    transcript: str = Form(""),
    chat_history: str = Form("[]"),  # JSON string
    image: UploadFile | None = File(None),
    image_base64: str | None = Form(None),
    generate_image: bool = Form(False),
):
    try:
        effective_query = await _rewrite_followup_query(query=query, transcript=transcript, chat_history=chat_history)
        chat_history_obj = _parse_chat_history(chat_history)
        assistant_turns_so_far = _count_assistant_turns(chat_history_obj)
        should_answer_now = assistant_turns_so_far >= _VISION_TUTOR_MAX_GUIDANCE_TURNS
        tutor_phase = "answer" if should_answer_now else "guide"

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

        # 2. Question Quality
        quality_result = await Runner.run(quality_agent, _input_text_message(effective_query))
        quality_data = _safe_json_loads(
            quality_result.final_output,
            default={"score": 0, "label": "Unknown", "feedback": "Quality model did not return JSON."},
        )

        # 3. Lesson Relevance Guardrail (skip if transcript is missing)
        if transcript.strip():
            relevance_prompt = f"""
Student Question:
{effective_query}

Lesson Transcript:
{transcript}

Recent Chat History (JSON string):
{chat_history}
""".strip()

            relevance_result = await Runner.run(relevance_agent, _input_text_message(relevance_prompt))
            relevance_data = _safe_json_loads(
                relevance_result.final_output,
                default={
                    "in_scope": True,
                    "confidence": 0.0,
                    "reason": "Relevance model did not return JSON; skipping lesson scope check.",
                    "suggested_questions": [],
                },
            )
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
                quality=quality_data,
                relevance=relevance_data,
            )

        # 4. Tutor Response (grounded by the real-time image)
        context = f"""
        Student Raw Input: {query}
        Interpreted Question: {effective_query}
        Video Transcript: {transcript}
        Question Quality: {quality_data.get("label")} (Score: {quality_data.get("score")})
        Feature: {quality_data.get("feedback")}
        Lesson Relevance: {relevance_data}

        Tutor Guidance Policy:
        - max_guidance_turns: {_VISION_TUTOR_MAX_GUIDANCE_TURNS}
        - assistant_turns_so_far: {assistant_turns_so_far}
        - should_answer_now: {"yes" if should_answer_now else "no"}
        - tutoring_phase: {tutor_phase}
        
        Chat History: {chat_history}
        """

        tutor_result = await Runner.run(vision_tutor_agent, _vision_input_message(context, image_url))

        generated_image = None
        generated_image_caption = None
        if generate_image:
            try:
                decision_prompt = f"""
Interpreted student question:
{effective_query}

Lesson transcript/context:
{transcript}

Tutor response:
{tutor_result.final_output}

Tutor phase:
{tutor_phase}

Student-provided photo available:
{"yes" if image_url else "no"}

If a photo is available and you set use_input_image=true:
- Write the prompt as an image-edit instruction that assumes the photo will be used as the base.
- Prefer overlays: labels, arrows, highlights, simple callouts.
- Keep the underlying scene/objects recognizable; avoid changing the whole image style.
""".strip()
                decision_result = await Runner.run(image_decider_agent, _input_text_message(decision_prompt))
                decision = json.loads(decision_result.final_output)
                print("decision: ", decision)
                should_generate = bool(decision.get("should_generate", False))
                confidence = float(decision.get("confidence", 0.0) or 0.0)
                use_input_image = bool(decision.get("use_input_image", False))
                prompt = (decision.get("prompt") or "").strip()
                caption = (decision.get("caption") or "").strip()

                if should_generate and confidence >= 0.6 and prompt:
                    input_image_data_url = image_url if (use_input_image and image_url) else None
                    image_b64 = await asyncio.to_thread(
                        generate_image_base64, prompt=prompt, input_image_data_url=input_image_data_url
                    )
                    generated_image = f"data:image/png;base64,{image_b64}"
                    generated_image_caption = caption or None
            except Exception:
                pass

        return AskResponse(
            response=tutor_result.final_output,
            quality=quality_data,
            relevance=relevance_data,
            generated_image=generated_image,
            generated_image_caption=generated_image_caption,
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return AskResponse(
            response="Sorry, I encountered an error processing your request.",
            quality={"error": str(e)},
            relevance={"error": str(e)},
            generated_image=None,
            generated_image_caption=None,
        )


@app.post("/ask_with_stream")
async def ask_with_stream(
    query: str = Form(...),
    transcript: str = Form(""),
    chat_history: str = Form("[]"),
    image: UploadFile | None = File(None),
    image_base64: str | None = Form(None),
    generate_image: bool = Form(False),
):
    try:
        effective_query = await _rewrite_followup_query(query=query, transcript=transcript, chat_history=chat_history)
        chat_history_obj = _parse_chat_history(chat_history)
        assistant_turns_so_far = _count_assistant_turns(chat_history_obj)
        should_answer_now = assistant_turns_so_far >= _VISION_TUTOR_MAX_GUIDANCE_TURNS
        tutor_phase = "answer" if should_answer_now else "guide"

        if image is not None:
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_type = image.content_type or "image/jpeg"
            if not content_type.startswith("image/"):
                content_type = "image/jpeg"
            image_url = f"data:{content_type};base64,{base64_image}"
        else:
            image_url = _image_url_from_upload_or_base64(image_base64=image_base64)

        quality_result = await Runner.run(quality_agent, _input_text_message(effective_query))
        quality_data = _safe_json_loads(
            quality_result.final_output,
            default={"score": 0, "label": "Unknown", "feedback": "Quality model did not return JSON."},
        )

        if transcript.strip():
            relevance_prompt = f"""
Student Question:
{effective_query}

Lesson Transcript:
{transcript}

Recent Chat History (JSON string):
{chat_history}
""".strip()

            relevance_result = await Runner.run(relevance_agent, _input_text_message(relevance_prompt))
            relevance_data = _safe_json_loads(
                relevance_result.final_output,
                default={
                    "in_scope": True,
                    "confidence": 0.0,
                    "reason": "Relevance model did not return JSON; skipping lesson scope check.",
                    "suggested_questions": [],
                },
            )
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
        Student Raw Input: {query}
        Interpreted Question: {effective_query}
        Video Transcript: {transcript}
        Question Quality: {quality_data.get("label")} (Score: {quality_data.get("score")})
        Feature: {quality_data.get("feedback")}
        Lesson Relevance: {relevance_data}

        Tutor Guidance Policy:
        - max_guidance_turns: {_VISION_TUTOR_MAX_GUIDANCE_TURNS}
        - assistant_turns_so_far: {assistant_turns_so_far}
        - should_answer_now: {"yes" if should_answer_now else "no"}
        - tutoring_phase: {tutor_phase}

        Chat History: {chat_history}
        """
    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)
        error_message = str(e) or "Unknown error"

        async def error_event_generator(msg: str = error_message):
            yield json.dumps({"type": "error", "content": msg}) + "\n"

        return StreamingResponse(error_event_generator(), media_type="application/x-ndjson")

    async def event_generator():
        yield (json.dumps({"type": "metadata", "quality": quality_data, "relevance": relevance_data}) + "\n")

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
            full_text = ""
            stream_result = Runner.run_streamed(vision_tutor_agent, _vision_input_message(context, image_url))
            async for event in stream_result.stream_events():
                if getattr(event, "type", None) != "raw_response_event":
                    continue

                data = event.data
                if getattr(data, "type", None) == "response.output_text.delta":
                    delta = getattr(data, "delta", "")
                    if delta:
                        full_text += delta
                        yield json.dumps({"type": "token", "content": delta}) + "\n"

            generated_image = None
            generated_image_caption = None
            if generate_image:
                try:
                    decision_prompt = f"""
Interpreted student question:
{effective_query}

Lesson transcript/context:
{transcript}

Tutor response:
{full_text}

Tutor phase:
{tutor_phase}

Student-provided photo available:
{"yes" if image_url else "no"}

If a photo is available and you set use_input_image=true:
- Write the prompt as an image-edit instruction that assumes the photo will be used as the base.
- Prefer overlays: labels, arrows, highlights, simple callouts.
- Keep the underlying scene/objects recognizable; avoid changing the whole image style.
""".strip()
                    decision_result = await Runner.run(image_decider_agent, _input_text_message(decision_prompt))
                    decision = json.loads(decision_result.final_output)
                    print("decision: ", decision)
                    should_generate = bool(decision.get("should_generate", False))
                    confidence = float(decision.get("confidence", 0.0) or 0.0)
                    use_input_image = bool(decision.get("use_input_image", False))
                    prompt = (decision.get("prompt") or "").strip()
                    caption = (decision.get("caption") or "").strip()

                    if should_generate and confidence >= 0.6 and prompt:
                        input_image_data_url = image_url if (use_input_image and image_url) else None
                        image_b64 = await asyncio.to_thread(
                            generate_image_base64, prompt=prompt, input_image_data_url=input_image_data_url
                        )
                        generated_image = f"data:image/png;base64,{image_b64}"
                        generated_image_caption = caption or None
                except Exception:
                    pass

            if generated_image:
                yield (
                    json.dumps(
                        {
                            "type": "image",
                            "data_url": generated_image,
                            "caption": generated_image_caption,
                        }
                    )
                    + "\n"
                )

            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
