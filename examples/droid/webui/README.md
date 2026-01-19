# DROID Voice Control WebUI

A minimal WebUI for voice-controlled robot task execution with OpenAI Whisper transcription and GPT-based prompt mapping.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  WebUI Frontend │◄───►│  FastAPI Server  │◄───►│  Robot Control  │
│  (Browser)      │     │  (server.py)     │     │  (main.py)      │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │
        │                       ▼
        │               ┌──────────────────┐
        │               │  OpenAI APIs     │
        │               │  - Whisper       │
        │               │  - GPT-4o-mini   │
        │               └──────────────────┘
        │
        ▼
   Voice Input
   (Microphone)
```

## Setup

### 1. Install dependencies

```bash
cd examples/droid/webui
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Configure task prompts

Edit `prompts.yaml` to add/modify available robot tasks:

```yaml
tasks:
  - id: 1
    prompt: "put the white bowl on the countertop shelf"
  - id: 2
    prompt: "put the white bowl in the sink"
  # Add more tasks...
```

## Running

### Step 1: Start the WebUI Server

```bash
cd examples/droid/webui
python server.py
```

The server will start on `http://localhost:8080`

### Step 2: Start the Robot Controller (WebUI mode)

In a separate terminal:

```bash
cd examples/droid
python main.py --external_camera=left --webui_mode=True --webui_host=localhost --webui_port=8080
```

### Step 3: Open the WebUI

Open your browser to `http://localhost:8080`

## Usage

1. **Hold the microphone button** and speak your command
2. **Release the button** to send the audio for transcription
3. The system will:
   - Transcribe your speech using Whisper
   - Map it to the closest task using GPT
   - Send the task to the robot
4. **Watch the robot execute** the task in real-time
5. **Speak a new command** anytime to interrupt and start a new task

## Example Voice Commands

| You Say | Robot Executes |
|---------|---------------|
| "put the bowl in the sink" | "put the white bowl in the sink" |
| "close that oven" | "close the oven" |
| "teabag in the drawer" | "put the teabag in the yellow drawer" |
| "what can you do?" | (Shows list of available tasks) |

## Troubleshooting

### Microphone not working
- Make sure your browser has microphone permissions
- Check that you're accessing via `http://localhost` (not file://)

### Robot not connected
- Ensure the robot controller is running with `--webui_mode=True`
- Check that the `--webui_host` and `--webui_port` match the server

### OpenAI API errors
- Verify `OPENAI_API_KEY` is set correctly
- Check your API key has access to Whisper and GPT-4o-mini

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI backend with Whisper/GPT integration |
| `templates/index.html` | Frontend HTML |
| `static/style.css` | Styling (dark theme, large fonts) |
| `static/app.js` | Voice capture and WebSocket logic |
| `prompts.yaml` | Configurable task list |
| `requirements.txt` | Python dependencies |


