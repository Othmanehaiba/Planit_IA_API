import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Header, Request, Query
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/api/generate-plan")
async def generate_plan(
    request: Request,
    authorization: str | None = Header(default=None),
    key: str | None = Query(default=None)
):
    # Accepte soit Authorization: Bearer <cle>, soit ?key=<cle>
    student_key = ""
    if authorization and authorization.startswith("Bearer "):
        student_key = authorization.replace("Bearer ", "").strip()
    elif key:
        student_key = key.strip()

    if not student_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if student_key not in VALID_STUDENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid student API key")

    api_key = (os.getenv("UPSTREAM_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing UPSTREAM_API_KEY in Vercel")

    body = await request.json()

    is_gemini_format = "contents" in body
    is_openai_format = "messages" in body

    system_msg = {
        "role": "system",
        "content": (
            "You are a project planning API. Return ONLY valid raw JSON "
            "matching the requested schema. Never include markdown fences or explanations."
        )
    }

    if is_gemini_format:
        parts = body.get("contents", [{}])[0].get("parts", [])
        prompt_text = "".join(p.get("text", "") for p in parts)
        messages = [system_msg, {"role": "user", "content": prompt_text}]
    elif is_openai_format:
        messages = [system_msg, *body["messages"]]
    elif "prompt" in body:
        messages = [system_msg, {"role": "user", "content": body["prompt"]}]
    else:
        raise HTTPException(status_code=400, detail="Invalid request format")

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-120b",
                    "messages": messages,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                }
            )
            response.raise_for_status()
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]

            cleaned = raw_content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(cleaned)
            json_str = json.dumps(parsed, ensure_ascii=False)

            # Réponse format Gemini pour leur nouveau api.js
            if is_gemini_format:
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": json_str}]
                            }
                        }
                    ]
                }

            # Réponse format OpenAI
            if is_openai_format:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json_str
                            }
                        }
                    ]
                }

            return parsed

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e.response.text}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="LLM returned malformed JSON. Please retry.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
