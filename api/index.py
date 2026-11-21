import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PLANIT AI Student Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_STUDENT_KEYS = {
    "planit-2026-groupA",
    "planit-2026-groupB",
    "planit-2026-demo",
}

class PlanRequest(BaseModel):
    prompt: str

@app.post("/api/generate-plan")
async def generate_plan(
    body: PlanRequest,
    authorization: str | None = Header(default=None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer API key")
    
    student_key = authorization.replace("Bearer ", "").strip()
    if student_key not in VALID_STUDENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid student API key")

    api_key = (os.getenv("UPSTREAM_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing UPSTREAM_API_KEY in Vercel")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a project planning API. Return ONLY valid raw JSON "
                    "matching the requested schema. Never include markdown fences or explanations."
                )
            },
            {"role": "user", "content": body.prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]

            cleaned = raw_content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e.response.text}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="LLM returned malformed JSON. Please retry.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
