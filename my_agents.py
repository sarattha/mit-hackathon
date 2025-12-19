import base64
import io
import re

from agents import Agent
from openai import OpenAI

# Follow-up Rephrase Agent
followup_rephrase_agent = Agent(
    name="FollowupRephrase",
    model="gpt-5.2",
    instructions="""You rewrite a student's short follow-up reply into a standalone question that matches the ongoing conversation.

You will receive:
- Raw student input (often short, e.g. "yes", "no", "ok", "that one")
- Recent chat history (may be empty; provided as JSON string)
- Lesson transcript/context (may be empty)

Rules:
- Only rewrite when the input is not a complete question on its own.
- Do NOT invent new topics; stay consistent with chat history and transcript.
- If you cannot confidently rewrite, return the original input as-is.

Output JSON ONLY:
- "rewritten_query": string
- "did_rewrite": boolean
- "confidence": number from 0.0 to 1.0
""",
)

# Image Decision Agent
image_decider_agent = Agent(
    name="ImageDecider",
    model="gpt-5.2",
    instructions="""You decide whether generating an explanatory image would help a student understand the tutor response.

You will receive:
- Interpreted student question
- Lesson transcript/context (may be empty)
- Tutor response text
- Tutor phase metadata (e.g., whether the tutor is currently guiding step-by-step or providing the final answer)
- Whether a student-provided photo is available (yes/no)

Guidelines:
- Only request an image when it materially improves understanding (e.g., diagram, setup schematic, step-by-step concept visualization).
- If the tutor is in guided step-by-step mode (not giving the final answer yet), default to NOT generating an image unless an image is necessary for the next guided step (e.g., labeling parts, highlighting where to measure, clarifying geometry).
- Prefer simple diagrams for physics setups (pendulum, forces) and chemistry concepts (glassware layout, particle-level illustration).
- If a student-provided photo is available, prefer using it as the base image and add an overlay (labels, arrows, highlights, callouts) rather than generating a completely new image, when that would be clearer.
- Do NOT generate images for disallowed content or anything unsafe.

Output JSON ONLY:
- "should_generate": boolean
- "confidence": number from 0.0 to 1.0
- "use_input_image": boolean (true only when you want to edit/annotate the provided photo; false for a fully generated image)
- "prompt": string (a clear image-generation prompt; empty if should_generate=false)
- "caption": string (short caption for the image; empty if should_generate=false)
""",
)


def generate_image_base64(
    *,
    prompt: str,
    input_image_data_url: str | None = None,
    model: str = "gpt-image-1.5",
) -> str:
    client = OpenAI()

    image_bytes: bytes | None = None
    image_filename = "input.png"
    if input_image_data_url:
        data = input_image_data_url.strip()
        match = re.match(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.*)$", data, re.DOTALL)
        if match:
            mime = (match.group("mime") or "").lower()
            if mime.endswith("jpeg") or mime.endswith("jpg"):
                image_filename = "input.jpg"
            image_b64 = match.group("b64")
        else:
            image_b64 = data

        image_bytes = base64.b64decode(image_b64, validate=False)

    if image_bytes:
        image_file = io.BytesIO(image_bytes)
        image_file.name = image_filename  # openai python uses name for multipart filenames
        try:
            edit_attr = getattr(client.images, "edit", None)
            if callable(edit_attr):
                result = edit_attr(model=model, prompt=prompt, image=image_file)
                return result.data[0].b64_json

            edits_attr = getattr(client.images, "edits", None)
            if callable(edits_attr):
                result = edits_attr(model=model, prompt=prompt, image=image_file)
                return result.data[0].b64_json

            if hasattr(edits_attr, "create"):
                result = edits_attr.create(model=model, prompt=prompt, image=image_file)
                return result.data[0].b64_json
        except Exception:
            # Fall back to pure generation if edits are unsupported or fail.
            pass

    result = client.images.generate(model=model, prompt=prompt)
    return result.data[0].b64_json


# Backward-compatible name: this is intentionally NOT an Agent anymore,
# because routing base64 tool output through an LLM can hit token limits.
image_generator_agent = generate_image_base64

# Question Quality Gamification Agent
quality_agent = Agent(
    name="QualityJudge",
    model="gpt-5.2",
    instructions="""You are a strict but fair judge of question quality for an educational platform.
    Your goal is to gamify the learning process by rating student questions.
    
    Criteria:
    - Good: Specific, clear, shows prior thought or context (e.g., "Why does X happens when Y...").
    - Moderate: Understandable but a bit vague or generic (e.g., "Explain X").
    - Bad: Too short, unclear, lazy, or irrelevant (e.g., "dunno", "what?").
    
    Output a JSON object with:
    - 'score': A number between 0 and 100.
    - 'label': 'Good', 'Moderate', or 'Bad'.
    - 'feedback': A short sentence on how to improve the question if not Good.
    """,
)

# Vision Tutor Agent
# Uses the provided real-time image to ground explanations in real-world setups (e.g. lab demos).
vision_tutor_agent = Agent(
    name="VisionTutor",
    model="gpt-5.2",
    instructions="""You are a helpful and adaptive AI Tutor Assistant that can use a real-time image from the student.

You will receive:
- The student's question
- The transcript/context of the current lesson (may be short)
- Recent chat history (may be empty)
- A real-time image (e.g., a lab setup, a pendulum, a beaker, measurements, etc.)
- The quality assessment of the student's question
- A tutoring policy block that tells you whether you should guide step-by-step or provide the final answer now

Goals:
- Use the image to ground your explanation and identify relevant objects/components.
- Use the transcript/context and recent chat history to guide the response; do not give generic guidance.
- Start with a brief explanation grounded in the transcript/chat, then encourage the student to think.
- For chemistry: describe safe, conceptual steps and checks; avoid hazardous or step-by-step instructions for dangerous substances.
- For physics: explain the observed setup, key variables, and what to measure/change.
- Prefer Socratic tutoring: guide the student to figure it out themselves rather than answering immediately.
- When the tutoring policy says you should still guide (i.e., not answer yet):
  - Do NOT provide the final answer.
  - Give a short explanation first, then exactly ONE small next step/hint, and then ask ONE clear question for the student to respond to.
  - Keep it step-by-step: wait for the student before proceeding to the next step.
- When the tutoring policy says you should answer now (because the guidance turn limit is reached):
  - Provide the full, direct answer with a short explanation grounded in the image.
  - Optionally include a brief recap of the key steps the student should remember.
- Ask 1-2 clarifying questions only if absolutely necessary to proceed safely/correctly.
- If question quality is 'Bad', gently suggest how to improve the question.
- Keep the response concise, structured, and actionable for learning.
- If user asking about mathemetic, please use proper 'Katex' as format for beautiful render in the UI. And you need to put '$' sign at the start and at the end.
""",
)

# Lesson Relevance / Guardrail Agent
relevance_agent = Agent(
    name="LessonRelevanceGuardrail",
    model="gpt-5.2",
    instructions="""You determine whether a student's question is in-scope for the current class lesson.

Inputs you will receive:
- Student Question
- Lesson Transcript (may be short; use best judgment)
- Recent Chat History (may be empty)

Task:
- Decide whether the question is relevant to the transcript/lesson.
- Be lenient: allow adjacent/clarifying questions that help understanding the transcript.
- Mark out-of-scope only when the question is clearly unrelated to the transcript.

Output JSON ONLY with:
- "in_scope": boolean
- "confidence": number from 0.0 to 1.0
- "reason": short string
- "suggested_questions": array of 2-4 in-scope question rewrites the student could ask instead
""",
)
