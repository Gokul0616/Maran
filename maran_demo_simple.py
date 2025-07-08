#!/usr/bin/env python3
"""
🧠 Maran AI Agent - Quick Demo Web Interface
==============================================================

A simplified version of the Maran AI agent for quick demonstration.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any
import os

# Web Framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maran_demo")

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===================== SIMPLE TOKENIZER =====================
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            "hello": 4, "world": 5, "ai": 6, "agent": 7, "maran": 8,
            "generate": 9, "text": 10, "code": 11, "python": 12,
            "the": 13, "a": 14, "is": 15, "and": 16, "to": 17,
            "of": 18, "in": 19, "that": 20, "for": 21, "with": 22
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
    
    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids])

# ===================== SIMPLE AI AGENT =====================
class SimpleMaranAgent:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.memory = []
        self.conversation_history = []
        self.thought_process = []
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        logger.info(f"Processing query: {query}")
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "user",
            "content": query
        })
        
        # Simulate AI thinking process
        thoughts = [
            {"type": "observation", "content": f"User asked: '{query}'", "confidence": 0.9},
            {"type": "hypothesis", "content": "This appears to be a question about AI capabilities", "confidence": 0.8},
            {"type": "action", "content": "Generating response based on context", "confidence": 0.85},
            {"type": "validation", "content": "Response seems appropriate", "confidence": 0.9}
        ]
        
        for thought in thoughts:
            thought["timestamp"] = datetime.now().isoformat()
            self.thought_process.append(thought)
            await asyncio.sleep(0.5)  # Simulate thinking time
        
        # Generate response (simplified)
        responses = {
            "hello": "Hello! I'm Maran, your AI agent. How can I assist you today?",
            "how are you": "I'm functioning optimally! All my systems are running smoothly.",
            "what can you do": "I can process natural language, generate responses, maintain memory, and provide real-time monitoring of my thought processes.",
            "generate code": "I can help generate Python code. What specific functionality would you like me to implement?",
            "explain ai": "AI (Artificial Intelligence) refers to computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving."
        }
        
        # Simple response matching
        response = "I understand your query. As an AI agent, I'm designed to process natural language and provide helpful responses. Is there something specific you'd like to know about?"
        
        for keyword, resp in responses.items():
            if keyword in query.lower():
                response = resp
                break
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "ai",
            "content": response
        })
        
        return {
            "response": response,
            "confidence": 0.87,
            "model_used": "maran-demo",
            "tool_results": {},
            "thoughts": thoughts[-2:]  # Return last 2 thoughts
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "monitor_stats": {
                "total_requests": len(self.conversation_history),
                "total_generations": len([msg for msg in self.conversation_history if msg["type"] == "ai"]),
                "uptime": 300,  # Demo uptime
                "avg_response_time": 1.2
            },
            "memory_stats": {
                "total_memories": len(self.memory),
                "recent_conversations": len(self.conversation_history[-10:])
            },
            "system_info": {
                "cpu_count": 4,
                "memory_available": 8.0,
                "gpu_available": False,
                "gpu_count": 0
            },
            "model_info": {
                "model_parameters": 125000,
                "model_type": "demo-gpt"
            },
            "tool_stats": {
                "text_generator": {"status": "active", "calls": 15},
                "memory_manager": {"status": "active", "calls": 8},
                "conversation_tracker": {"status": "active", "calls": 22}
            }
        }

# ===================== WEB APPLICATION =====================
app = FastAPI(title="Maran AI Agent Demo")
agent = SimpleMaranAgent()

# Store WebSocket connections
websocket_connections = set()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Maran AI Agent - Demo Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        .dashboard { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            grid-template-rows: auto 1fr auto; 
            gap: 20px; 
            padding: 20px; 
            max-width: 1400px; 
            margin: 0 auto;
            min-height: 100vh;
        }
        .header { 
            grid-column: 1 / -1; 
            text-align: center; 
            background: rgba(255, 255, 255, 0.1); 
            padding: 30px; 
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header h1 { 
            font-size: 3rem; 
            margin-bottom: 10px; 
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .panel { 
            background: rgba(255, 255, 255, 0.15); 
            padding: 25px; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .panel h3 { 
            margin-bottom: 20px; 
            font-size: 1.4rem;
            color: #ffd700;
        }
        .chat-messages { 
            height: 400px; 
            overflow-y: auto; 
            border: 1px solid rgba(255, 255, 255, 0.3); 
            border-radius: 10px; 
            padding: 15px; 
            margin-bottom: 15px;
            background: rgba(0, 0, 0, 0.2);
        }
        .message { 
            margin: 10px 0; 
            padding: 10px; 
            border-radius: 8px; 
            animation: fadeIn 0.5s ease-in;
        }
        .user-message { background: rgba(103, 58, 183, 0.3); }
        .ai-message { background: rgba(76, 175, 80, 0.3); }
        .system-message { background: rgba(255, 193, 7, 0.3); }
        .input-area { 
            display: flex; 
            gap: 10px; 
        }
        .input-area input { 
            flex: 1; 
            padding: 12px; 
            border: none; 
            border-radius: 8px; 
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 16px;
        }
        .input-area input::placeholder { color: rgba(255, 255, 255, 0.7); }
        .input-area button { 
            padding: 12px 25px; 
            border: none; 
            border-radius: 8px; 
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white; 
            cursor: pointer; 
            font-size: 16px;
            transition: transform 0.2s;
        }
        .input-area button:hover { transform: scale(1.05); }
        .input-area button:disabled { 
            background: #666; 
            cursor: not-allowed; 
            transform: none;
        }
        .thoughts-container { 
            max-height: 300px; 
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
        }
        .thought-item { 
            background: rgba(255, 255, 255, 0.1); 
            margin: 8px 0; 
            padding: 12px; 
            border-radius: 8px; 
            border-left: 4px solid;
            font-size: 14px;
            animation: slideIn 0.3s ease-out;
        }
        .thought-observation { border-left-color: #17a2b8; }
        .thought-hypothesis { border-left-color: #ffc107; }
        .thought-action { border-left-color: #28a745; }
        .thought-validation { border-left-color: #dc3545; }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 15px; 
            margin-bottom: 20px; 
        }
        .metric-card { 
            background: rgba(255, 255, 255, 0.1); 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .metric-value { 
            font-size: 28px; 
            font-weight: bold; 
            color: #ffd700; 
            margin-bottom: 5px;
        }
        .metric-label { 
            font-size: 12px; 
            opacity: 0.8; 
        }
        .status-indicator { 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            display: inline-block; 
            margin-right: 8px; 
            background: #28a745;
            animation: pulse 2s infinite;
        }
        .footer { 
            grid-column: 1 / -1; 
            text-align: center; 
            padding: 20px; 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 14px;
            background: rgba(40, 167, 69, 0.9);
            backdrop-filter: blur(10px);
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🧠 Maran AI Agent</h1>
            <p>Sophisticated Autonomous AI with Real-time Thought Processing</p>
            <div id="connectionStatus" class="connection-status">
                <span class="status-indicator"></span>Connected
            </div>
        </div>
        
        <div class="panel">
            <h3>💬 AI Conversation</h3>
            <div class="chat-messages" id="chatMessages">
                <div class="message system-message">
                    <strong>🤖 Maran:</strong> Hello! I'm Maran AI Agent. I'm ready to assist you with questions, generate responses, and show you my thought processes in real-time. What would you like to explore?
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="queryInput" placeholder="Ask me anything..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <div class="panel">
            <h3>🧠 AI Thought Process</h3>
            <div class="thoughts-container" id="thoughtsContainer">
                <div class="thought-item thought-observation">
                    <strong>OBSERVATION:</strong> System initialized and ready for interaction
                    <div style="margin-top: 5px; font-size: 12px; opacity: 0.7;">Confidence: 95%</div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h3>📊 System Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="totalRequests">0</div>
                    <div class="metric-label">Total Requests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="totalGenerations">0</div>
                    <div class="metric-label">AI Responses</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="uptime">5m</div>
                    <div class="metric-label">System Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="memoryCount">0</div>
                    <div class="metric-label">Memories Stored</div>
                </div>
            </div>
            <div style="font-size: 14px; opacity: 0.8;">
                <div>🔧 Model: maran-demo (125K parameters)</div>
                <div>💾 Memory: 8GB available</div>
                <div>⚡ Status: <span style="color: #28a745;">Optimal</span></div>
            </div>
        </div>
        
        <div class="panel">
            <h3>🛠️ Active Tools</h3>
            <div style="space-y: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; margin: 8px 0;">
                    <span>Text Generator</span>
                    <span style="background: #28a745; padding: 2px 8px; border-radius: 12px; font-size: 12px;">Active</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; margin: 8px 0;">
                    <span>Memory Manager</span>
                    <span style="background: #28a745; padding: 2px 8px; border-radius: 12px; font-size: 12px;">Active</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; margin: 8px 0;">
                    <span>Conversation Tracker</span>
                    <span style="background: #28a745; padding: 2px 8px; border-radius: 12px; font-size: 12px;">Active</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 Maran AI Agent Demo - Showcasing advanced AI capabilities with real-time monitoring</p>
        </div>
    </div>

    <script>
        class MaranInterface {
            constructor() {
                this.ws = null;
                this.isConnected = false;
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.setupEventListeners();
                this.updateMetrics();
                setInterval(() => this.updateMetrics(), 5000);
            }
            
            connectWebSocket() {
                const wsUrl = `ws://${window.location.host}/ws`;
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.isConnected = true;
                    console.log('Connected to Maran AI Agent');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    this.isConnected = false;
                    setTimeout(() => this.connectWebSocket(), 3000);
                };
            }
            
            setupEventListeners() {
                const queryInput = document.getElementById('queryInput');
                const sendButton = document.getElementById('sendButton');
                
                queryInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendQuery();
                    }
                });
                
                sendButton.addEventListener('click', () => this.sendQuery());
            }
            
            sendQuery() {
                const queryInput = document.getElementById('queryInput');
                const query = queryInput.value.trim();
                
                if (!query || !this.isConnected) return;
                
                queryInput.value = '';
                document.getElementById('sendButton').disabled = true;
                
                this.addMessage('user', query);
                this.ws.send(JSON.stringify({
                    type: 'query',
                    data: { query: query }
                }));
            }
            
            handleMessage(message) {
                const { type, data } = message;
                
                switch (type) {
                    case 'response':
                        this.addMessage('ai', data.response);
                        if (data.thoughts) {
                            data.thoughts.forEach(thought => this.addThought(thought));
                        }
                        document.getElementById('sendButton').disabled = false;
                        break;
                        
                    case 'thought':
                        this.addThought(data);
                        break;
                        
                    case 'status':
                        this.updateSystemMetrics(data);
                        break;
                }
            }
            
            addMessage(sender, content) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                
                const timestamp = new Date().toLocaleTimeString();
                const icon = sender === 'user' ? '👤' : '🤖';
                const name = sender === 'user' ? 'You' : 'Maran';
                
                messageDiv.innerHTML = `
                    <div style="margin-bottom: 5px;">
                        <strong>${icon} ${name}:</strong>
                        <span style="font-size: 12px; opacity: 0.7; float: right;">${timestamp}</span>
                    </div>
                    <div>${content}</div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            addThought(thoughtData) {
                const thoughtsContainer = document.getElementById('thoughtsContainer');
                const thoughtDiv = document.createElement('div');
                thoughtDiv.className = `thought-item thought-${thoughtData.type}`;
                
                const timestamp = new Date().toLocaleTimeString();
                thoughtDiv.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 3px;">
                        ${thoughtData.type.toUpperCase()}:
                    </div>
                    <div>${thoughtData.content}</div>
                    <div style="margin-top: 5px; font-size: 12px; opacity: 0.7;">
                        Confidence: ${(thoughtData.confidence * 100).toFixed(1)}% | ${timestamp}
                    </div>
                `;
                
                thoughtsContainer.appendChild(thoughtDiv);
                thoughtsContainer.scrollTop = thoughtsContainer.scrollHeight;
            }
            
            async updateMetrics() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    document.getElementById('totalRequests').textContent = data.monitor_stats.total_requests;
                    document.getElementById('totalGenerations').textContent = data.monitor_stats.total_generations;
                    document.getElementById('memoryCount').textContent = data.memory_stats.total_memories;
                    
                } catch (error) {
                    console.error('Failed to update metrics:', error);
                }
            }
        }
        
        // Initialize the interface when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new MaranInterface();
        });
    </script>
</body>
</html>
""")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                query = message["data"]["query"]
                
                # Send thoughts as they're generated
                response_data = await agent.process_query(query)
                
                # Send thoughts one by one for streaming effect
                for thought in agent.thought_process[-4:]:  # Last 4 thoughts
                    await websocket.send_text(json.dumps({
                        "type": "thought",
                        "data": thought
                    }))
                    await asyncio.sleep(0.3)
                
                # Send final response
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "data": response_data
                }))
                
    except WebSocketDisconnect:
        websocket_connections.discard(websocket)

@app.get("/api/status")
async def get_status():
    """Get system status for metrics"""
    return JSONResponse(agent.get_system_status())

async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if websocket_connections:
        disconnected = set()
        for websocket in websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.add(websocket)
        websocket_connections -= disconnected

def main():
    """Main entry point"""
    print("🧠 Starting Maran AI Agent Demo...")
    print("=" * 50)
    print("🌐 Web Interface: http://localhost:8000")
    print("🔧 Features:")
    print("   • Real-time AI conversation with thought streaming")
    print("   • Multi-panel sophisticated dashboard")
    print("   • System monitoring and analytics")
    print("   • Memory management interface")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()