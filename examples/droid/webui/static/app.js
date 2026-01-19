/**
 * DROID Voice Control - Frontend JavaScript
 * Handles voice capture, WebSocket communication, and UI updates
 */

// State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let websocket = null;
let reconnectInterval = null;

// DOM Elements
const voiceButton = document.getElementById('voiceButton');
const recordingIndicator = document.getElementById('recordingIndicator');
const chatArea = document.getElementById('chatArea');
const connectionStatus = document.getElementById('connectionStatus');
const currentTaskDiv = document.getElementById('currentTask');
const currentTaskText = document.getElementById('currentTaskText');
const progressFill = document.getElementById('progressFill');
const stepCounter = document.getElementById('stepCounter');
const clarifyModal = document.getElementById('clarifyModal');
const clarifyMessage = document.getElementById('clarifyMessage');
const taskOptions = document.getElementById('taskOptions');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    initVoiceCapture();
});

// WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/frontend`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        console.log('WebSocket connected');
        clearInterval(reconnectInterval);
    };
    
    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        // Try to reconnect
        reconnectInterval = setInterval(() => {
            console.log('Attempting to reconnect...');
            initWebSocket();
        }, 5000);
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'status':
            updateRobotStatus(data.status, data.task);
            break;
        case 'robot_connected':
            updateConnectionStatus(data.connected);
            break;
        case 'step_update':
            updateProgress(data.step, data.max_steps);
            break;
        case 'task_complete':
            addMessage('robot', data.message || 'Task completed!', 'success');
            hideCurrentTask();
            break;
    }
}

function updateConnectionStatus(connected) {
    const statusDot = connectionStatus.querySelector('.status-dot');
    const statusText = connectionStatus.querySelector('.status-text');
    
    if (connected) {
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Robot Connected';
    } else {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Robot Disconnected';
    }
}

function updateRobotStatus(status, task) {
    if (status === 'executing' && task) {
        showCurrentTask(task);
    } else if (status === 'idle') {
        hideCurrentTask();
    }
}

function updateProgress(step, maxSteps) {
    const percent = (step / maxSteps) * 100;
    progressFill.style.width = `${percent}%`;
    stepCounter.textContent = `Step ${step} / ${maxSteps}`;
}

function showCurrentTask(task) {
    currentTaskDiv.style.display = 'block';
    currentTaskText.textContent = task;
    progressFill.style.width = '0%';
    stepCounter.textContent = 'Starting...';
}

function hideCurrentTask() {
    currentTaskDiv.style.display = 'none';
}

// Voice capture
async function initVoiceCapture() {
    // Request microphone permission
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop()); // Stop immediately, just checking permission
        
        // Setup button events
        voiceButton.addEventListener('mousedown', startRecording);
        voiceButton.addEventListener('mouseup', stopRecording);
        voiceButton.addEventListener('mouseleave', stopRecording);
        
        // Touch events for mobile
        voiceButton.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startRecording();
        });
        voiceButton.addEventListener('touchend', (e) => {
            e.preventDefault();
            stopRecording();
        });
        
    } catch (error) {
        console.error('Microphone permission denied:', error);
        addMessage('robot', 'Microphone access denied. Please enable microphone permissions.', 'error');
    }
}

async function startRecording() {
    if (isRecording) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            stream.getTracks().forEach(track => track.stop());
            await processAudio(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        voiceButton.classList.add('recording');
        recordingIndicator.classList.add('active');
        
    } catch (error) {
        console.error('Error starting recording:', error);
    }
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    
    mediaRecorder.stop();
    isRecording = false;
    voiceButton.classList.remove('recording');
    recordingIndicator.classList.remove('active');
}

// Process audio and send to backend
async function processAudio(audioBlob) {
    // Show processing state
    const processingDiv = addProcessingMessage('Transcribing...');
    
    try {
        // Step 1: Transcribe audio
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.webm');
        
        const transcribeResponse = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
        });
        
        if (!transcribeResponse.ok) {
            throw new Error('Transcription failed');
        }
        
        const transcribeData = await transcribeResponse.json();
        const userText = transcribeData.text;
        
        // Remove processing message and show user's speech
        removeProcessingMessage(processingDiv);
        addMessage('user', userText);
        
        // Show processing for mapping
        const mappingDiv = addProcessingMessage('Understanding...');
        
        // Step 2: Map prompt to task
        const mapResponse = await fetch('/map_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: userText }),
        });
        
        if (!mapResponse.ok) {
            throw new Error('Prompt mapping failed');
        }
        
        const mapData = await mapResponse.json();
        removeProcessingMessage(mappingDiv);
        
        // Handle response based on status
        handleMappingResponse(mapData);
        
    } catch (error) {
        console.error('Error processing audio:', error);
        removeProcessingMessage(processingDiv);
        addMessage('robot', 'Sorry, something went wrong. Please try again.', 'error');
    }
}

function handleMappingResponse(data) {
    switch (data.status) {
        case 'match':
            // Send task to robot
            addMessage('robot', `Got it! Executing: "${data.task_prompt}"`, 'task');
            sendTaskToRobot(data.task_prompt);
            break;
            
        case 'clarify':
            // Show clarification modal
            showClarifyModal(data.message, data.candidates);
            break;
            
        case 'question':
            // Display response
            addMessage('robot', data.response, 'success');
            break;
            
        case 'no_match':
            // Display message
            addMessage('robot', data.message, 'error');
            break;
            
        default:
            addMessage('robot', 'I didn\'t understand that. Please try again.', 'error');
    }
}

async function sendTaskToRobot(taskPrompt) {
    try {
        const response = await fetch('/send_task', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_prompt: taskPrompt }),
        });
        
        const data = await response.json();
        
        if (!data.success) {
            addMessage('robot', data.message || 'Failed to send task to robot', 'error');
        }
    } catch (error) {
        console.error('Error sending task:', error);
        addMessage('robot', 'Failed to communicate with robot', 'error');
    }
}

// UI helpers
function addMessage(sender, text, type = '') {
    // Remove welcome message if present
    const welcome = chatArea.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    
    const labelDiv = document.createElement('div');
    labelDiv.className = 'message-label';
    labelDiv.textContent = sender === 'user' ? 'You' : 'Robot';
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = `message-bubble ${type}`;
    bubbleDiv.textContent = text;
    
    messageDiv.appendChild(labelDiv);
    messageDiv.appendChild(bubbleDiv);
    chatArea.appendChild(messageDiv);
    
    // Scroll to bottom
    chatArea.scrollTop = chatArea.scrollHeight;
    
    return messageDiv;
}

function addProcessingMessage(text) {
    const div = document.createElement('div');
    div.className = 'processing-message';
    div.innerHTML = `<div class="spinner"></div><span>${text}</span>`;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return div;
}

function removeProcessingMessage(div) {
    if (div && div.parentNode) {
        div.parentNode.removeChild(div);
    }
}

// Clarification modal
function showClarifyModal(message, candidates) {
    clarifyMessage.textContent = message;
    taskOptions.innerHTML = '';
    
    candidates.forEach(candidate => {
        const button = document.createElement('button');
        button.className = 'task-option';
        button.textContent = candidate.task_prompt;
        button.onclick = () => {
            closeClarifyModal();
            addMessage('robot', `Got it! Executing: "${candidate.task_prompt}"`, 'task');
            sendTaskToRobot(candidate.task_prompt);
        };
        taskOptions.appendChild(button);
    });
    
    clarifyModal.style.display = 'flex';
}

function closeClarifyModal() {
    clarifyModal.style.display = 'none';
}

// Close modal on outside click
clarifyModal.addEventListener('click', (e) => {
    if (e.target === clarifyModal) {
        closeClarifyModal();
    }
});


