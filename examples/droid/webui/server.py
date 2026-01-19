"""
WebUI Backend Server for DROID Robot Voice Control.

This server provides:
- Voice transcription via OpenAI Whisper API
- Prompt mapping via GPT-4o-mini
- WebSocket communication with frontend and robot controller
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel
from starlette.requests import Request

# Initialize FastAPI app
app = FastAPI(title="DROID WebUI Voice Control")

# Setup templates and static files
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# OpenAI client (initialized lazily)
_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# Load task prompts from YAML
def load_prompts() -> list[dict]:
    """Load task prompts from prompts.yaml."""
    prompts_path = BASE_DIR / "prompts.yaml"
    if not prompts_path.exists():
        return []
    with open(prompts_path) as f:
        data = yaml.safe_load(f)
    return data.get("tasks", [])


# GPT System Prompt for task mapping
SYSTEM_PROMPT_TEMPLATE = """You are a robot task interpreter. Your job is to map natural human speech to specific robot tasks.

AVAILABLE TASKS:
{tasks_list}

RULES:
1. If the user's speech clearly matches or is similar to one of the available tasks, respond with:
   {{"status": "match", "task_id": <id>, "task_prompt": "<exact prompt from the list>", "confidence": "high|medium|low"}}

2. If the user's speech is ambiguous and could match multiple tasks, respond with:
   {{"status": "clarify", "candidates": [{{"task_id": <id>, "task_prompt": "<prompt>"}}...], "message": "<ask user to clarify>"}}

3. If the user asks a question (e.g., "what can you do?", "are you ready?", "help"), respond with:
   {{"status": "question", "response": "<helpful answer about robot capabilities or current status>"}}

4. If the user's speech doesn't match any task, respond with:
   {{"status": "no_match", "message": "<explain what tasks are available>"}}

5. Handle fuzzy/casual speech intelligently:
   - "bowl to sink" -> matches "put the white bowl in the sink"
   - "close that oven door" -> matches "close the oven"
   - "teabag drawer" -> matches "put the teabag in the yellow drawer"
   - Ignore filler words like "uh", "um", "like", "you know"
   - Handle typos and speech recognition errors gracefully

6. For greetings or small talk (e.g., "hi", "hello", "thanks"), respond with:
   {{"status": "question", "response": "<friendly response and mention you're ready to help>"}}

Always respond with valid JSON only. No additional text or explanation outside the JSON.
"""


def build_system_prompt() -> str:
    """Build the system prompt with current task list."""
    tasks = load_prompts()
    tasks_list = "\n".join([f"- ID {t['id']}: {t['prompt']}" for t in tasks])
    return SYSTEM_PROMPT_TEMPLATE.format(tasks_list=tasks_list)


# WebSocket connection manager for real-time communication
class ConnectionManager:
    """Manages WebSocket connections for frontend and robot controller."""

    def __init__(self):
        self.frontend_connections: list[WebSocket] = []
        self.robot_connection: Optional[WebSocket] = None
        self.current_status: str = "idle"
        self.current_task: Optional[str] = None

    async def connect_frontend(self, websocket: WebSocket):
        await websocket.accept()
        self.frontend_connections.append(websocket)
        # Send current status to new connection
        await websocket.send_json({
            "type": "status",
            "status": self.current_status,
            "task": self.current_task,
        })

    async def connect_robot(self, websocket: WebSocket):
        await websocket.accept()
        self.robot_connection = websocket

    def disconnect_frontend(self, websocket: WebSocket):
        if websocket in self.frontend_connections:
            self.frontend_connections.remove(websocket)

    def disconnect_robot(self):
        self.robot_connection = None

    async def broadcast_to_frontend(self, message: dict):
        """Send message to all connected frontends."""
        for connection in self.frontend_connections:
            await connection.send_json(message)

    async def send_to_robot(self, message: dict) -> bool:
        """Send message to robot controller."""
        if self.robot_connection:
            await self.robot_connection.send_json(message)
            return True
        return False

    async def update_status(self, status: str, task: Optional[str] = None):
        """Update robot status and broadcast to frontends."""
        self.current_status = status
        self.current_task = task
        await self.broadcast_to_frontend({
            "type": "status",
            "status": status,
            "task": task,
        })


manager = ConnectionManager()


# Pydantic models
class MapPromptRequest(BaseModel):
    text: str


class MapPromptResponse(BaseModel):
    status: str
    task_id: Optional[int] = None
    task_prompt: Optional[str] = None
    confidence: Optional[str] = None
    candidates: Optional[list[dict]] = None
    message: Optional[str] = None
    response: Optional[str] = None


class SendTaskRequest(BaseModel):
    task_prompt: str


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main WebUI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio using OpenAI Whisper API."""
    client = get_openai_client()
    
    # Read audio file content
    audio_content = await audio.read()
    
    # Create a temporary file-like object for the API
    import io
    audio_file = io.BytesIO(audio_content)
    audio_file.name = audio.filename or "audio.webm"
    
    # Call Whisper API
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",
    )
    
    return {"text": transcript.text}


@app.post("/map_prompt", response_model=MapPromptResponse)
async def map_prompt(request: MapPromptRequest):
    """Map user speech to a robot task using GPT."""
    client = get_openai_client()
    
    system_prompt = build_system_prompt()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.text},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    
    # Parse JSON response
    result = json.loads(response.choices[0].message.content)
    return MapPromptResponse(**result)


@app.post("/send_task")
async def send_task(request: SendTaskRequest):
    """Send a task to the robot controller."""
    success = await manager.send_to_robot({
        "type": "new_task",
        "prompt": request.task_prompt,
    })
    
    if success:
        await manager.update_status("executing", request.task_prompt)
        return {"success": True, "message": f"Task sent: {request.task_prompt}"}
    else:
        return {"success": False, "message": "Robot not connected"}


@app.get("/tasks")
async def get_tasks():
    """Get list of available tasks."""
    return {"tasks": load_prompts()}


@app.get("/status")
async def get_status():
    """Get current robot status."""
    return {
        "status": manager.current_status,
        "task": manager.current_task,
        "robot_connected": manager.robot_connection is not None,
    }


# WebSocket endpoints
@app.websocket("/ws/frontend")
async def websocket_frontend(websocket: WebSocket):
    """WebSocket endpoint for frontend connections."""
    await manager.connect_frontend(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle frontend messages if needed
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect_frontend(websocket)


@app.websocket("/ws/robot")
async def websocket_robot(websocket: WebSocket):
    """WebSocket endpoint for robot controller connection."""
    await manager.connect_robot(websocket)
    await manager.broadcast_to_frontend({
        "type": "robot_connected",
        "connected": True,
    })
    try:
        while True:
            data = await websocket.receive_json()
            # Handle robot status updates
            if data.get("type") == "status_update":
                await manager.update_status(
                    data.get("status", "unknown"),
                    data.get("task"),
                )
            elif data.get("type") == "step_update":
                await manager.broadcast_to_frontend({
                    "type": "step_update",
                    "step": data.get("step"),
                    "max_steps": data.get("max_steps"),
                })
            elif data.get("type") == "task_complete":
                await manager.update_status("idle", None)
                await manager.broadcast_to_frontend({
                    "type": "task_complete",
                    "message": data.get("message", "Task completed"),
                })
    except WebSocketDisconnect:
        manager.disconnect_robot()
        await manager.update_status("disconnected", None)
        await manager.broadcast_to_frontend({
            "type": "robot_connected",
            "connected": False,
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


