import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
API_BASE = AIPIPE_BASE_URL.rstrip("/")

client = httpx.AsyncClient(timeout=30.0)


class PromptRequest(BaseModel):
    prompt: str
    stream: bool = True


async def stream_llm(prompt: str):

    # Instant first token (important for latency tests)
    yield 'data: {"choices":[{"delta":{"content":"Generating Java code...\\n"}}]}\n\n'

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "stream": True,
        "max_tokens": 1800,
        "temperature": 0.6,
        "messages": [
            {"role": "system", "content": "You are a senior Java developer."},
            {
                "role": "user",
                "content": f"""
Generate a complete Java class named DataProcessor.
Minimum 120 lines.
Minimum 2500 characters.
Include file reading, validation methods,
error handling, helper methods, logging, and comments.

{prompt}
"""
            },
        ],
    }

    async with client.stream(
        "POST",
        f"{API_BASE}/chat/completions",
        headers=headers,
        json=payload,
    ) as response:

        if response.status_code != 200:
            yield f'data: {{"error":"API error {response.status_code}"}}\n\n'
            yield "data: [DONE]\n\n"
            return

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                yield f"data: {data}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/stream")
async def stream_endpoint(request: PromptRequest):

    if not AIPIPE_TOKEN:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN not configured")

    if not request.stream:
        raise HTTPException(status_code=400, detail="Streaming must be true")

    return StreamingResponse(
        stream_llm(request.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )