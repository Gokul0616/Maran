#!/usr/bin/env python3
"""
🧠 Maran AI Agent - Quick Demo Version
=====================================

A streamlined demo version of the Maran AI agent for immediate testing.
This version uses simplified components for faster initialization.

Usage: python3 maran_demo.py
Access: http://localhost:8000
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import sqlite3
from pathlib import Path

# Web Framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Basic ML
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===================== LOGGING SETUP =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maran_demo")

# Create directories
os.makedirs("demo_data", exist_ok=True)

# ===================== SIMPLE TOKENIZER =====================
class SimpleDemoTokenizer:
    def __init__(self):
        # Pre-built vocabulary for demo
        self.vocab = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            "hello": 4, "world": 5, "how": 6, "are": 7, "you": 8,
            "I": 9, "am": 10, "fine": 11, "thank": 12, "thanks": 13,
            "help": 14, "me": 15, "with": 16, "code": 17, "python": 18,
            "generate": 19, "create": 20, "write": 21, "function": 22,
            "class": 23, "variable": 24, "loop": 25, "if": 26, "else": 27,
            "for": 28, "while": 29, "def": 30, "return": 31, "print": 32,
            "input": 33, "output": 34, "data": 35, "analysis": 36,
            "machine": 37, "learning": 38, "AI": 39, "artificial": 40,
            "intelligence": 41, "model": 42, "train": 43, "test": 44,
            "predict": 45, "algorithm": 46, "neural": 47, "network": 48,
            "deep": 49, "learning": 50, "what": 51, "when": 52, "where": 53,
            "why": 54, "who": 55, "can": 56, "could": 57, "would": 58,
            "should": 59, "the": 60, "a": 61, "an": 62, "and": 63,
            "or": 64, "but": 65, "in": 66, "on": 67, "at": 68,
            "to": 69, "from": 70, "of": 71, "by": 72, "as": 73,
            "is": 74, "was": 75, "will": 76, "be": 77, "do": 78,
            "does": 79, "did": 80, "have": 81, "has": 82, "had": 83,
            "get": 84, "set": 85, "run": 86, "execute": 87, "start": 88,
            "stop": 89, "end": 90, "begin": 91, "finish": 92, "complete": 93,
            "yes": 94, "no": 95, "ok": 96, "okay": 97, "sure": 98, "good": 99
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        logger.info(f"Demo tokenizer initialized with {len(self.vocab)} tokens")

    def encode(self, text: str, max_length=None) -> List[int]:
        tokens = [self.vocab["<BOS>"]]
        words = text.lower().split()
        for word in words:
            tokens.append(self.vocab.get(word, self.vocab["<UNK>"]))
        tokens.append(self.vocab["<EOS>"])
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "<UNK>")
            if token not in ["<PAD>", "<BOS>", "<EOS>"]:
                words.append(token)
        return " ".join(words)

# ===================== SIMPLE MODEL =====================
class SimpleDemoModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
            num_layers
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Simple forward pass
        embeddings = self.embedding(x)
        # Create a simple causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        output = self.transformer(embeddings, embeddings, tgt_mask=mask)
        return self.output_proj(output)

    def generate(self, prompt: str, tokenizer, max_length=50, temperature=1.0):
        """Simple generation function"""
        self.eval()
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        generated = []
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(input_ids)
                next_token_logits = outputs[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                
                # Add to input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Keep only recent context to prevent memory issues
                if input_ids.size(1) > 100:
                    input_ids = input_ids[:, -50:]
                    
                # Stop if EOS token
                if next_token.item() == tokenizer.vocab.get("<EOS>", -1):
                    break
        
        return tokenizer.decode(generated)

# ===================== SIMPLE MEMORY =====================
class SimpleDemoMemory:
    def __init__(self):
        self.db_path = "demo_data/memory.db"
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT DEFAULT 'conversation',
                importance REAL DEFAULT 0.5
            )
        ''')
        conn.commit()
        conn.close()

    def store(self, content: str, memory_type: str = "conversation", importance: float = 0.5):
        timestamp = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (timestamp, content, type, importance)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, content, memory_type, importance))
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return memory_id

    def query(self, query_text: str, limit: int = 3):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM memories 
            WHERE content LIKE ? 
            ORDER BY importance DESC, timestamp DESC 
            LIMIT ?
        ''', (f'%{query_text}%', limit))
        memories = cursor.fetchall()
        conn.close()
        return memories

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM memories')
        total = cursor.fetchone()[0]
        conn.close()
        return {"total_memories": total, "type_distribution": {"conversation": total}}

# ===================== DEMO TOOLS =====================
class DemoTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0

    def execute(self, **kwargs):
        self.usage_count += 1
        return self._execute(**kwargs)

    def _execute(self, **kwargs):
        return {"success": True, "message": f"Demo {self.name} executed", "input": kwargs}

# ===================== MARAN DEMO AGENT =====================
class MaranDemoAgent:
    def __init__(self):
        self.tokenizer = SimpleDemoTokenizer()
        self.model = SimpleDemoModel(len(self.tokenizer.vocab))
        self.memory = SimpleDemoMemory()
        self.tools = {
            "calculator": DemoTool("calculator", "Perform calculations"),
            "code_helper": DemoTool("code_helper", "Help with code"),
            "web_search": DemoTool("web_search", "Search the web")
        }
        self.stats = {
            "total_requests": 0,
            "total_generations": 0,
            "uptime_start": time.time()
        }
        logger.info("🧠 Maran Demo Agent initialized!")

    async def process_query(self, query: str, stream_callback=None):
        """Process user query with demo responses"""
        self.stats["total_requests"] += 1
        start_time = time.time()

        try:
            # Store query
            memory_id = self.memory.store(f"User: {query}", "query", 0.8)

            # Send status updates
            if stream_callback:
                await stream_callback({"type": "status", "data": "🧠 Thinking..."})
                await asyncio.sleep(0.5)  # Simulate thinking time

            # Search memory for context
            relevant_memories = self.memory.query(query, limit=2)
            
            if stream_callback:
                await stream_callback({"type": "status", "data": "💭 Generating response..."})

            # Generate response based on query patterns
            response = self._generate_contextual_response(query)
            
            if stream_callback:
                # Simulate streaming generation
                for i, char in enumerate(response):
                    if i % 10 == 0:  # Update every 10 characters
                        partial = response[:i+1]
                        await stream_callback({
                            "type": "generation",
                            "data": {
                                "text": partial,
                                "progress": i / len(response)
                            }
                        })
                        await asyncio.sleep(0.1)

            # Store response
            self.memory.store(f"AI: {response}", "response", 0.7)
            
            # Check for tool usage
            tool_results = self._check_tools(query)
            
            self.stats["total_generations"] += 1
            
            return {
                "response": response,
                "thoughts": [
                    {
                        "type": "observation",
                        "content": f"User asked: {query}",
                        "confidence": 1.0,
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "type": "hypothesis", 
                        "content": "I should provide a helpful response based on the query type",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "type": "action",
                        "content": f"Generated response: {response[:50]}...",
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "tool_results": tool_results,
                "relevant_memories": len(relevant_memories),
                "response_time": time.time() - start_time,
                "memory_id": memory_id
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "thoughts": [],
                "tool_results": {},
                "relevant_memories": 0,
                "response_time": time.time() - start_time,
                "error": str(e)
            }

    def _generate_contextual_response(self, query: str) -> str:
        """Generate contextual responses based on query patterns"""
        query_lower = query.lower()
        
        # Code-related queries
        if any(word in query_lower for word in ["code", "python", "function", "program", "script"]):
            return "I can help you with coding! Here's a simple Python function example:\n\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('World'))\n```\n\nWhat specific coding task would you like help with?"
        
        # Greeting patterns
        elif any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm Maran, your AI assistant. I can help you with coding, answer questions, and assist with various tasks. How can I help you today?"
        
        # How are you patterns
        elif any(phrase in query_lower for phrase in ["how are you", "how do you do", "what's up"]):
            return "I'm doing great, thank you for asking! I'm running smoothly and ready to help you with any questions or tasks you have. What would you like to work on?"
        
        # Help patterns
        elif any(word in query_lower for word in ["help", "assist", "support"]):
            return "I'm here to help! I can assist you with:\n\n• Writing and debugging code\n• Answering questions\n• Explaining concepts\n• Problem-solving\n• And much more!\n\nWhat do you need help with today?"
        
        # Math/calculation patterns
        elif any(word in query_lower for word in ["calculate", "math", "compute", "add", "subtract", "multiply", "divide"]):
            return "I can help with calculations! For complex math, I can use my calculator tool. What calculation do you need help with?"
        
        # AI/ML patterns
        elif any(word in query_lower for word in ["ai", "artificial intelligence", "machine learning", "neural network", "model"]):
            return "I'm an AI agent built with neural networks and machine learning! I use transformer architecture for language understanding and generation. I also have memory systems and can learn from our conversations. What would you like to know about AI?"
        
        # Default responses
        else:
            responses = [
                f"That's an interesting question about '{query}'. Let me think about that...",
                f"Regarding '{query}', I can provide some insights based on my knowledge.",
                f"I understand you're asking about '{query}'. Here's what I can tell you:",
                "I'm processing your request and gathering relevant information to provide the best response."
            ]
            
            # Use the model for more sophisticated generation
            try:
                model_response = self.model.generate(query, self.tokenizer, max_length=30)
                if model_response and len(model_response.strip()) > 5:
                    return f"Based on my analysis: {model_response}\n\nIs there anything specific you'd like me to elaborate on?"
            except:
                pass
            
            return responses[hash(query) % len(responses)]

    def _check_tools(self, query: str) -> Dict[str, Any]:
        """Check if any tools should be used based on the query"""
        tool_results = {}
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["calculate", "math", "compute"]):
            result = self.tools["calculator"].execute(query=query)
            tool_results["calculator"] = result
        
        if any(word in query_lower for word in ["code", "program", "function"]):
            result = self.tools["code_helper"].execute(query=query)
            tool_results["code_helper"] = result
        
        if any(word in query_lower for word in ["search", "find", "lookup"]):
            result = self.tools["web_search"].execute(query=query)
            tool_results["web_search"] = result
        
        return tool_results

    def get_system_status(self):
        """Get system status"""
        uptime = time.time() - self.stats["uptime_start"]
        return {
            "agent_status": "running",
            "model_info": {
                "vocab_size": len(self.tokenizer.vocab),
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "device": str(self.model.device)
            },
            "memory_stats": self.memory.get_stats(),
            "tool_stats": {name: {"name": tool.name, "usage_count": tool.usage_count, "last_used": None} 
                         for name, tool in self.tools.items()},
            "monitor_stats": {
                "total_requests": self.stats["total_requests"],
                "total_generations": self.stats["total_generations"],
                "uptime": uptime,
                "errors": 0
            },
            "system_info": {
                "cpu_count": 4,
                "memory_total": 8_000_000_000,
                "memory_available": 4_000_000_000,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

# ===================== WEB APPLICATION =====================
app = FastAPI(title="Maran AI Agent Demo", description="Streamlined AI Agent Demo")
agent = MaranDemoAgent()
active_connections: List[WebSocket] = []

# HTML Template (same sophisticated interface as before)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Maran AI Agent - Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            height: 100vh;
            overflow: hidden;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: auto 1fr 1fr;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }
        
        .header {
            grid-column: 1 / -1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            overflow-y: auto;
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: #ffd700;
            border-bottom: 2px solid #ffd700;
            padding-bottom: 5px;
        }
        
        .chat-panel {
            grid-row: 2 / -1;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            animation: fadeIn 0.3s ease-in;
        }
        
        .user-message {
            background: rgba(0, 123, 255, 0.3);
            border-left: 4px solid #007bff;
        }
        
        .ai-message {
            background: rgba(40, 167, 69, 0.3);
            border-left: 4px solid #28a745;
        }
        
        .status-message {
            background: rgba(255, 193, 7, 0.3);
            border-left: 4px solid #ffc107;
            font-style: italic;
        }
        
        .error-message {
            background: rgba(220, 53, 69, 0.3);
            border-left: 4px solid #dc3545;
        }
        
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        #queryInput {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
            font-size: 16px;
        }
        
        #queryInput::placeholder {
            color: rgba(255, 255, 255, 0.7);
        }
        
        #sendButton {
            padding: 12px 24px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        #sendButton:hover {
            background: #218838;
        }
        
        #sendButton:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .thought-item {
            background: rgba(0, 0, 0, 0.2);
            margin: 5px 0;
            padding: 8px;
            border-radius: 6px;
            border-left: 3px solid;
            font-size: 14px;
        }
        
        .thought-observation { border-left-color: #17a2b8; }
        .thought-hypothesis { border-left-color: #ffc107; }
        .thought-action { border-left-color: #28a745; }
        .thought-validation { border-left-color: #dc3545; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .metric-card {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #ffd700;
        }
        
        .metric-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .tool-item {
            background: rgba(0, 0, 0, 0.2);
            margin: 5px 0;
            padding: 10px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .tool-status {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            background: #28a745;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .thinking {
            animation: pulse 1.5s infinite;
        }
        
        code {
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Header -->
        <div class="header">
            <h1>🧠 Maran AI Agent - Demo</h1>
            <p>Fast Demo Version with Real-time AI Interaction</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar"></div>
            </div>
        </div>
        
        <!-- Chat Panel -->
        <div class="panel chat-panel">
            <h3>💬 Chat with Maran</h3>
            <div class="chat-messages" id="chatMessages">
                <div class="message ai-message">
                    <div style="font-weight: bold; margin-bottom: 5px;">
                        🤖 Maran 
                        <span style="font-size: 12px; opacity: 0.7;">Just now</span>
                    </div>
                    <div>Hello! I'm Maran, your AI assistant. I can help you with coding, answer questions, and demonstrate various AI capabilities. Try asking me something!</div>
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="queryInput" placeholder="Ask me anything... (try: 'hello', 'help with python', 'calculate 2+2')" />
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <!-- Thoughts Panel -->
        <div class="panel">
            <h3>🧠 AI Reasoning Process</h3>
            <div id="thoughtsContainer">
                <div class="thought-item thought-observation">
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 3px;">OBSERVATION</div>
                    <div>System initialized and ready for interaction</div>
                    <div style="font-size: 11px; opacity: 0.6; margin-top: 3px;">Confidence: 100%</div>
                </div>
            </div>
        </div>
        
        <!-- Metrics Panel -->
        <div class="panel">
            <h3>📊 System Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="totalRequests">0</div>
                    <div class="metric-label">Total Requests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="totalGenerations">0</div>
                    <div class="metric-label">Generations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="uptime">0s</div>
                    <div class="metric-label">Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="memoryCount">0</div>
                    <div class="metric-label">Memories</div>
                </div>
            </div>
            <div id="systemInfo"></div>
        </div>
        
        <!-- Tools Panel -->
        <div class="panel">
            <h3>🛠️ Available Tools</h3>
            <div id="toolsContainer">
                <div class="tool-item">
                    <div>
                        <div style="font-weight: bold;">Calculator</div>
                        <div style="font-size: 12px; opacity: 0.8;">Mathematical computations</div>
                    </div>
                    <div class="tool-status">Ready</div>
                </div>
                <div class="tool-item">
                    <div>
                        <div style="font-weight: bold;">Code Helper</div>
                        <div style="font-size: 12px; opacity: 0.8;">Programming assistance</div>
                    </div>
                    <div class="tool-status">Ready</div>
                </div>
                <div class="tool-item">
                    <div>
                        <div style="font-weight: bold;">Web Search</div>
                        <div style="font-size: 12px; opacity: 0.8;">Information lookup</div>
                    </div>
                    <div class="tool-status">Ready</div>
                </div>
            </div>
        </div>
        
        <!-- Memory Panel -->
        <div class="panel">
            <h3>🧠 Memory Status</h3>
            <div id="memoryContainer">
                <div style="background: rgba(0, 0, 0, 0.2); margin: 5px 0; padding: 8px; border-radius: 6px;">
                    <div style="color: #ffd700; font-weight: bold; font-size: 12px;">SYSTEM</div>
                    <div style="font-size: 14px;">Demo agent initialized and ready</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        class MaranDemoInterface {
            constructor() {
                this.ws = null;
                this.isConnected = false;
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.setupEventListeners();
                this.updateSystemStatus();
                
                // Update metrics every 5 seconds
                setInterval(() => this.updateSystemStatus(), 5000);
            }
            
            connectWebSocket() {
                const wsUrl = `ws://${window.location.host}/ws`;
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.isConnected = true;
                    this.addMessage('system', '🔌 Connected to Maran Demo', 'status');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    this.isConnected = false;
                    this.addMessage('system', '🔌 Disconnected from agent', 'error');
                    setTimeout(() => this.connectWebSocket(), 3000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.addMessage('system', '❌ Connection error', 'error');
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
                
                this.addMessage('user', query, 'user');
                document.getElementById('thoughtsContainer').innerHTML = '';
                
                this.ws.send(JSON.stringify({
                    type: 'query',
                    data: { query: query }
                }));
            }
            
            handleWebSocketMessage(message) {
                const { type, data } = message;
                
                switch (type) {
                    case 'response':
                        this.addMessage('ai', data.response, 'ai');
                        this.displayThoughts(data.thoughts);
                        this.displayToolResults(data.tool_results);
                        document.getElementById('sendButton').disabled = false;
                        document.getElementById('progressBar').style.width = '0%';
                        break;
                        
                    case 'thought':
                        this.addThought(data);
                        break;
                        
                    case 'status':
                        this.addMessage('system', data, 'status');
                        break;
                        
                    case 'generation':
                        this.updateGenerationProgress(data);
                        break;
                        
                    case 'error':
                        this.addMessage('system', data.message, 'error');
                        document.getElementById('sendButton').disabled = false;
                        break;
                        
                    case 'system_status':
                        this.updateSystemMetrics(data);
                        break;
                }
            }
            
            addMessage(sender, content, type) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                const timestamp = new Date().toLocaleTimeString();
                
                // Process content for code blocks
                let processedContent = content;
                if (typeof content === 'string') {
                    // Convert markdown code blocks to HTML
                    processedContent = content.replace(/```python\\n([\\s\\S]*?)\\n```/g, '<pre><code>$1</code></pre>');
                    processedContent = processedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
                }
                
                messageDiv.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 5px;">
                        ${sender === 'user' ? '👤 You' : sender === 'ai' ? '🤖 Maran' : '⚙️ System'} 
                        <span style="font-size: 12px; opacity: 0.7;">${timestamp}</span>
                    </div>
                    <div>${processedContent}</div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            displayThoughts(thoughts) {
                const thoughtsContainer = document.getElementById('thoughtsContainer');
                thoughtsContainer.innerHTML = '';
                
                thoughts.forEach(thought => {
                    const thoughtDiv = document.createElement('div');
                    thoughtDiv.className = `thought-item thought-${thought.type}`;
                    
                    const timestamp = new Date(thought.timestamp).toLocaleTimeString();
                    thoughtDiv.innerHTML = `
                        <div style="font-size: 12px; opacity: 0.8; margin-bottom: 3px;">
                            ${thought.type.toUpperCase()} - ${timestamp}
                        </div>
                        <div>${thought.content}</div>
                        <div style="font-size: 11px; opacity: 0.6; margin-top: 3px;">
                            Confidence: ${(thought.confidence * 100).toFixed(1)}%
                        </div>
                    `;
                    
                    thoughtsContainer.appendChild(thoughtDiv);
                });
            }
            
            updateGenerationProgress(data) {
                const progressBar = document.getElementById('progressBar');
                progressBar.style.width = `${data.progress * 100}%`;
                
                if (data.text) {
                    const lastMessage = document.querySelector('.ai-message:last-child');
                    if (lastMessage) {
                        const contentDiv = lastMessage.querySelector('div:last-child');
                        contentDiv.innerHTML = data.text + ' <span class="thinking">▋</span>';
                    }
                }
            }
            
            displayToolResults(toolResults) {
                if (!toolResults || Object.keys(toolResults).length === 0) return;
                
                let resultsHtml = '<div style="margin-top: 10px;"><strong>🔧 Tool Results:</strong></div>';
                
                for (const [toolName, result] of Object.entries(toolResults)) {
                    const status = result.success ? '✅' : '❌';
                    resultsHtml += `
                        <div style="background: rgba(0,0,0,0.2); margin: 5px 0; padding: 8px; border-radius: 4px;">
                            <strong>${status} ${toolName}:</strong><br>
                            ${result.success ? result.message : result.error}
                        </div>
                    `;
                }
                
                this.addMessage('system', resultsHtml, 'status');
            }
            
            async updateSystemStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    this.updateSystemMetrics(data);
                } catch (error) {
                    console.error('Failed to fetch system status:', error);
                }
            }
            
            updateSystemMetrics(data) {
                document.getElementById('totalRequests').textContent = data.monitor_stats.total_requests;
                document.getElementById('totalGenerations').textContent = data.monitor_stats.total_generations;
                document.getElementById('uptime').textContent = `${Math.floor(data.monitor_stats.uptime)}s`;
                document.getElementById('memoryCount').textContent = data.memory_stats.total_memories;
                
                const systemInfo = document.getElementById('systemInfo');
                systemInfo.innerHTML = `
                    <div style="font-size: 12px; margin-top: 10px;">
                        <div>Device: ${data.model_info.device}</div>
                        <div>Model Parameters: ${data.model_info.model_parameters.toLocaleString()}</div>
                        <div>Vocab Size: ${data.model_info.vocab_size}</div>
                        <div>GPU Available: ${data.system_info.gpu_available ? 'Yes' : 'No'}</div>
                    </div>
                `;
                
                this.updateToolsStatus(data.tool_stats);
            }
            
            updateToolsStatus(toolStats) {
                const toolsContainer = document.getElementById('toolsContainer');
                toolsContainer.innerHTML = '';
                
                for (const [toolName, stats] of Object.entries(toolStats)) {
                    const toolDiv = document.createElement('div');
                    toolDiv.className = 'tool-item';
                    toolDiv.innerHTML = `
                        <div>
                            <div style="font-weight: bold;">${stats.name}</div>
                            <div style="font-size: 12px; opacity: 0.8;">Used: ${stats.usage_count} times</div>
                        </div>
                        <div class="tool-status">Ready</div>
                    `;
                    
                    toolsContainer.appendChild(toolDiv);
                }
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            new MaranDemoInterface();
        });
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/api/status")
async def get_status():
    return agent.get_system_status()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                query = message["data"]["query"]
                
                async def stream_callback(data):
                    try:
                        await websocket.send_text(json.dumps(data))
                    except:
                        pass
                
                result = await agent.process_query(query, stream_callback)
                
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "data": result
                }))
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e)}
            }))
        except:
            pass
        active_connections.remove(websocket)

def main():
    print("🧠 Starting Maran AI Agent Demo...")
    print("=" * 50)
    print(f"🌐 Demo Interface: http://localhost:8000")
    print("🚀 Features:")
    print("   • Fast initialization (no training required)")
    print("   • Real-time AI conversation")
    print("   • Interactive reasoning display")
    print("   • Demo tools integration")
    print("   • Live system monitoring")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()