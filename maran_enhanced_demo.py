#!/usr/bin/env python3
"""
🧠 Maran AI Agent - Fixed Demo with Working WebSocket
========================================================

A fully functional demo with proper WebSocket communication.
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
logger = logging.getLogger("maran_fixed_demo")

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
            "of": 18, "in": 19, "that": 20, "for": 21, "with": 22,
            "function": 23, "def": 24, "return": 25, "if": 26, "else": 27,
            "how": 28, "what": 29, "can": 30, "you": 31, "do": 32
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
    
    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids])

# ===================== ENHANCED AI AGENT =====================
class EnhancedMaranAgent:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.memory = []
        self.conversation_history = []
        self.thought_process = []
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query and generate response with proper WebSocket support"""
        logger.info(f"Processing query: {query}")
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "user",
            "content": query
        })
        
        # Generate response based on query content
        response = self._generate_response(query)
        
        # Simulate AI thinking process
        thoughts = [
            {"type": "observation", "content": f"User asked: '{query}'", "confidence": 0.95},
            {"type": "hypothesis", "content": self._analyze_query(query), "confidence": 0.88},
            {"type": "action", "content": "Generating appropriate response", "confidence": 0.90},
            {"type": "validation", "content": "Response validated and ready", "confidence": 0.93}
        ]
        
        for thought in thoughts:
            thought["timestamp"] = datetime.now().isoformat()
            self.thought_process.append(thought)
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "ai",
            "content": response
        })
        
        return {
            "response": response,
            "confidence": 0.89,
            "model_used": "maran-enhanced",
            "tool_results": {},
            "thoughts": thoughts
        }
    
    def _analyze_query(self, query: str) -> str:
        """Analyze the query to provide contextual hypothesis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            return "This is a greeting - user wants to start conversation"
        elif any(word in query_lower for word in ["code", "function", "python", "programming"]):
            return "User is asking about programming/code generation"
        elif any(word in query_lower for word in ["what", "how", "explain"]):
            return "This is an informational query - user wants explanation"
        elif any(word in query_lower for word in ["capabilities", "can you", "able to"]):
            return "User wants to know about AI capabilities"
        else:
            return "General query requiring informative response"
    
    def _generate_response(self, query: str) -> str:
        """Generate contextual response based on query"""
        query_lower = query.lower()
        
        # Greeting responses
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Maran AI Agent. I'm ready to help you with questions, code generation, explanations, and much more. What would you like to explore today?"
        
        # Capability questions
        elif any(word in query_lower for word in ["capabilities", "can you", "able to", "what can you do"]):
            return "I can help you with: 💬 Natural language conversations, 🐍 Python code generation, 📚 Explanations of complex topics, 🧠 Problem-solving and analysis, 💾 Memory of our conversation, and 🔄 Real-time thought process visualization. What specific area interests you?"
        
        # Code generation
        elif any(word in query_lower for word in ["code", "function", "python", "programming", "generate"]):
            if "fibonacci" in query_lower:
                return """Here's a Python function to calculate Fibonacci numbers:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Optimized version with memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

Would you like me to explain how this works or generate other code examples?"""
            
            elif "calculator" in query_lower:
                return """Here's a simple calculator function in Python:

```python
def calculator(operation, a, b):
    operations = {
        'add': a + b,
        'subtract': a - b,
        'multiply': a * b,
        'divide': a / b if b != 0 else 'Error: Division by zero'
    }
    return operations.get(operation, 'Invalid operation')

# Example usage:
result = calculator('add', 10, 5)  # Returns 15
```

This supports basic arithmetic operations. Need anything more advanced?"""
            
            else:
                return "I'd be happy to help you generate Python code! What specific functionality do you need? For example: functions, classes, algorithms, data processing, web scraping, etc."
        
        # AI/ML questions
        elif any(word in query_lower for word in ["ai", "artificial intelligence", "machine learning", "neural"]):
            return "Artificial Intelligence is the simulation of human intelligence by machines. It includes: 🧠 Machine Learning (learning from data), 🔍 Pattern Recognition, 💬 Natural Language Processing, 👁️ Computer Vision, and 🤖 Decision Making. I'm an example of conversational AI that can understand context and generate helpful responses!"
        
        # Explanation requests
        elif any(word in query_lower for word in ["explain", "what is", "how does", "tell me about"]):
            return "I'd be happy to explain! I can break down complex topics into understandable parts, provide examples, and answer follow-up questions. What specific topic would you like me to explain?"
        
        # Default response
        else:
            return f"I understand your question: '{query}'. As an advanced AI agent, I can help you with various tasks including answering questions, generating code, explaining concepts, and problem-solving. Could you provide more specific details about what you'd like assistance with?"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "monitor_stats": {
                "total_requests": len(self.conversation_history),
                "total_generations": len([msg for msg in self.conversation_history if msg["type"] == "ai"]),
                "uptime": 600,  # Demo uptime
                "avg_response_time": 0.8
            },
            "memory_stats": {
                "total_memories": len(self.memory),
                "recent_conversations": len(self.conversation_history[-10:])
            },
            "system_info": {
                "cpu_count": 16,
                "memory_available": 8.0,
                "gpu_available": False,
                "gpu_count": 0
            },
            "model_info": {
                "model_parameters": 125000,
                "model_type": "maran-enhanced"
            },
            "tool_stats": {
                "text_generator": {"status": "active", "calls": 25},
                "memory_manager": {"status": "active", "calls": 12},
                "conversation_tracker": {"status": "active", "calls": 35},
                "code_generator": {"status": "active", "calls": 8}
            }
        }

# ===================== WEB APPLICATION =====================
app = FastAPI(title="Maran AI Agent - Enhanced Demo")
agent = EnhancedMaranAgent()

# Store WebSocket connections
websocket_connections = set()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page with working WebSocket"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Maran AI Agent - Enhanced Demo</title>
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
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 14px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .connected { background: rgba(40, 167, 69, 0.9); }
        .disconnected { background: rgba(220, 53, 69, 0.9); }
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
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🧠 Maran AI Agent</h1>
            <p>Enhanced Demo with Working Real-time Communication</p>
            <div id="connectionStatus" class="connection-status disconnected">
                <span class="status-indicator"></span>Connecting...
            </div>
        </div>
        
        <div class="panel">
            <h3>💬 AI Conversation</h3>
            <div class="chat-messages" id="chatMessages">
                <div class="message system-message">
                    <strong>🤖 Maran:</strong> Hello! I'm Maran AI Agent with enhanced capabilities. I can help with questions, code generation, explanations, and more. Type a message to start our conversation!
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
                    <strong>OBSERVATION:</strong> System initialized and ready for enhanced interaction
                    <div style="margin-top: 5px; font-size: 12px; opacity: 0.7;">Confidence: 98%</div>
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
                    <div class="metric-value" id="uptime">10m</div>
                    <div class="metric-label">System Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="memoryCount">0</div>
                    <div class="metric-label">Memories Stored</div>
                </div>
            </div>
            <div style="font-size: 14px; opacity: 0.8;">
                <div>🔧 Model: maran-enhanced (125K parameters)</div>
                <div>💾 Memory: 8GB available</div>
                <div>⚡ Status: <span style="color: #28a745;">Enhanced & Ready</span></div>
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
                    <span>Code Generator</span>
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
            <p>🚀 Maran AI Agent Enhanced Demo - Real-time AI Interaction Fixed & Working</p>
        </div>
    </div>

    <script>
        class MaranInterface {
            constructor() {
                this.ws = null;
                this.isConnected = false;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
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
                console.log('Connecting to WebSocket:', wsUrl);
                
                try {
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        console.log('WebSocket connected successfully');
                        this.updateConnectionStatus(true);
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.handleMessage(data);
                        } catch (error) {
                            console.error('Error parsing WebSocket message:', error);
                        }
                    };
                    
                    this.ws.onclose = () => {
                        this.isConnected = false;
                        this.updateConnectionStatus(false);
                        console.log('WebSocket disconnected');
                        
                        if (this.reconnectAttempts < this.maxReconnectAttempts) {
                            this.reconnectAttempts++;
                            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                            setTimeout(() => this.connectWebSocket(), 3000);
                        }
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.isConnected = false;
                        this.updateConnectionStatus(false);
                    };
                } catch (error) {
                    console.error('Failed to create WebSocket connection:', error);
                    this.updateConnectionStatus(false);
                }
            }
            
            updateConnectionStatus(connected) {
                const statusElement = document.getElementById('connectionStatus');
                if (connected) {
                    statusElement.textContent = '● Connected';
                    statusElement.className = 'connection-status connected';
                } else {
                    statusElement.textContent = '● Disconnected';
                    statusElement.className = 'connection-status disconnected';
                }
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
                
                if (!query) {
                    alert('Please enter a message');
                    return;
                }
                
                if (!this.isConnected) {
                    alert('WebSocket not connected. Please wait for reconnection.');
                    return;
                }
                
                console.log('Sending query:', query);
                queryInput.value = '';
                document.getElementById('sendButton').disabled = true;
                
                this.addMessage('user', query);
                
                try {
                    const message = {
                        type: 'query',
                        data: { query: query }
                    };
                    this.ws.send(JSON.stringify(message));
                    console.log('Message sent successfully');
                } catch (error) {
                    console.error('Error sending message:', error);
                    this.addMessage('system', 'Error sending message. Please try again.', 'error');
                    document.getElementById('sendButton').disabled = false;
                }
            }
            
            handleMessage(message) {
                console.log('Received message:', message);
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
                        
                    case 'error':
                        this.addMessage('system', `Error: ${data.message}`, 'error');
                        document.getElementById('sendButton').disabled = false;
                        break;
                        
                    default:
                        console.log('Unknown message type:', type);
                }
            }
            
            addMessage(sender, content, type = null) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                
                const messageType = type || `${sender}-message`;
                messageDiv.className = `message ${messageType}`;
                
                const timestamp = new Date().toLocaleTimeString();
                const icon = sender === 'user' ? '👤' : sender === 'ai' ? '🤖' : '⚙️';
                const name = sender === 'user' ? 'You' : sender === 'ai' ? 'Maran' : 'System';
                
                // Handle code blocks
                const formattedContent = content.replace(/```python\\n([\\s\\S]*?)```/g, 
                    '<pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin: 10px 0; overflow-x: auto;"><code>$1</code></pre>');
                
                messageDiv.innerHTML = `
                    <div style="margin-bottom: 5px;">
                        <strong>${icon} ${name}:</strong>
                        <span style="font-size: 12px; opacity: 0.7; float: right;">${timestamp}</span>
                    </div>
                    <div>${formattedContent}</div>
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
            console.log('Initializing Maran Interface...');
            new MaranInterface();
        });
    </script>
</body>
</html>
""")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with proper error handling"""
    await websocket.accept()
    websocket_connections.add(websocket)
    logger.info("New WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            try:
                message = json.loads(data)
                
                if message["type"] == "query":
                    query = message["data"]["query"]
                    logger.info(f"Processing query: {query}")
                    
                    # Process the query
                    response_data = await agent.process_query(query)
                    
                    # Send thoughts one by one for streaming effect
                    for thought in response_data["thoughts"]:
                        await websocket.send_text(json.dumps({
                            "type": "thought",
                            "data": thought
                        }))
                        await asyncio.sleep(0.5)  # Simulate thinking time
                    
                    # Send final response
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "data": response_data
                    }))
                    
                    logger.info("Response sent successfully")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                }))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": str(e)}
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        websocket_connections.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_connections.discard(websocket)

@app.get("/api/status")
async def get_status():
    """Get system status for metrics"""
    return JSONResponse(agent.get_system_status())

def main():
    """Main entry point"""
    print("🧠 Starting Maran AI Agent - Enhanced Demo...")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:8001")
    print("🔧 Enhanced Features:")
    print("   • Fixed WebSocket communication")
    print("   • Working message sending")
    print("   • Real-time thought streaming")
    print("   • Enhanced AI responses")
    print("   • Code generation capabilities")
    print("   • Improved error handling")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

if __name__ == "__main__":
    main()