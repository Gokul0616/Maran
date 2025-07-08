#!/usr/bin/env python3
"""
🧠 Maran AI Agent - Enhanced & Fixed Web Interface
===============================================================

A comprehensive web-based interface for the Maran AI agent system featuring:
- Fixed AI processing with proper error handling
- Real-time streaming of AI thought processes
- Dark/Light mode toggle
- Integrated terminal interface
- Advanced monitoring and analytics
- Responsive multi-panel dashboard

Usage: python3 maran_enhanced_fixed.py
Access: http://localhost:8001
"""

import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
import hashlib
import subprocess
import tempfile
import threading
from collections import deque, defaultdict, Counter as CollectionsCounter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import secrets

# Core ML and Data Science
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import psutil

# Web Framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Monitoring
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# ===================== CONFIGURATION =====================
CONFIG = {
    "model": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "vocab_size": 1000,
        "max_len": 256,
        "dropout": 0.1
    },
    "training": {
        "batch_size": 4,
        "lr": 1e-4,
        "epochs": 1,
        "grad_accum_steps": 2
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8000,
        "metrics_port": 9090
    },
    "safety": {
        "max_execution_time": 10,
        "memory_limit_mb": 512,
        "forbidden_patterns": [
            r"rm\s+-rf", r"format\s+c:", r"del\s+/", r"shutdown", r"reboot"
        ]
    }
}

# ===================== LOGGING SETUP =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maran_enhanced")

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("memory", exist_ok=True)

# ===================== SIMPLE TOKENIZER =====================
class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        # Initialize with basic vocabulary
        self._build_basic_vocab()
        
    def _build_basic_vocab(self):
        # Add special tokens
        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
        
        # Add common words and characters
        common_words = [
            "hello", "world", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "must", "shall", "this", "that", "these", "those", "here", "there", "where", "when", "why", "how", "what", "who",
            "yes", "no", "not", "never", "always", "sometimes", "often", "usually", "rarely", "good", "bad", "big", "small", "new", "old", "young",
            "python", "code", "function", "class", "method", "variable", "string", "number", "list", "dict", "if", "else", "for", "while", "try", "except",
            "import", "from", "def", "return", "print", "input", "output", "data", "file", "read", "write", "open", "close", "true", "false", "none"
        ]
        
        # Add characters and digits
        characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()-[]{}'\""
        
        all_tokens = common_words + list(characters)
        
        for token in all_tokens:
            if token not in self.token_to_id and len(self.token_to_id) < self.vocab_size:
                self.token_to_id[token] = len(self.token_to_id)
        
        # Fill remaining slots with generated tokens
        while len(self.token_to_id) < self.vocab_size:
            token = f"token_{len(self.token_to_id)}"
            self.token_to_id[token] = len(self.token_to_id)
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
    def encode(self, text, max_length=None):
        if not text.strip():
            return [self.token_to_id["<PAD>"]]
            
        tokens = [self.token_to_id["<BOS>"]]
        
        # Simple word-level tokenization
        words = text.lower().strip().split()
        for word in words:
            # Add word tokens
            for char in word:
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.token_to_id["<UNK>"])
            
            # Add space token if available
            if " " in self.token_to_id:
                tokens.append(self.token_to_id[" "])
        
        tokens.append(self.token_to_id["<EOS>"])
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.token_to_id["<EOS>"]]
        
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return ''.join(tokens).replace('  ', ' ').strip()

# ===================== SIMPLE TRANSFORMER MODEL =====================
class SimpleTransformerModel(nn.Module):
    def __init__(self, tokenizer, d_model=128, nhead=4, num_layers=2, max_len=256, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.max_len = max_len
        
        vocab_size = len(tokenizer.token_to_id)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        B, T = x.shape
        
        # Create attention mask for padding
        pad_token_id = self.tokenizer.token_to_id.get("<PAD>", 0)
        src_key_padding_mask = (x == pad_token_id)
        
        token_embeddings = self.token_embed(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_embeddings = self.pos_embed(position_ids)
        
        x = token_embeddings + position_embeddings
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        
        return self.head(x)

    def generate(self, prompt: str, max_length=50, temperature=0.8, top_k=10, stream_callback=None):
        """Generate text with improved sampling"""
        self.eval()
        
        try:
            tokens = self.tokenizer.encode(prompt, max_length=self.max_len//2)
            if not tokens:
                tokens = [self.tokenizer.token_to_id["<BOS>"]]
            
            input_ids = torch.tensor([tokens], device=self.device)
            
            generated = []
            for step in range(max_length):
                with torch.no_grad():
                    outputs = self(input_ids)
                    logits = outputs[0, -1, :]  # Get last token logits
                    
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_actual = min(top_k, logits.size(-1))
                        topk_values, topk_indices = torch.topk(logits, top_k_actual)
                        
                        # Create a mask for top-k values
                        mask = torch.full_like(logits, float('-inf'))
                        mask[topk_indices] = topk_values
                        logits = mask
                    
                    # Sample from the distribution
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Ensure we don't sample invalid indices
                    valid_probs = probs[probs > 0]
                    if len(valid_probs) == 0:
                        # Fallback to uniform distribution over valid tokens
                        next_token = torch.randint(0, len(self.tokenizer.token_to_id), (1,))
                    else:
                        next_token = torch.multinomial(probs, 1)
                    
                    # Ensure token ID is valid
                    token_id = next_token.item()
                    if token_id >= len(self.tokenizer.token_to_id):
                        token_id = self.tokenizer.token_to_id.get("<UNK>", 1)
                    
                    generated.append(token_id)
                    
                    # Stream intermediate results
                    if stream_callback and step % 5 == 0:
                        partial_text = self.tokenizer.decode(generated)
                        stream_callback(partial_text, step, max_length)
                    
                    # Update input
                    next_token_tensor = torch.tensor([[token_id]], device=self.device)
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                    
                    # Keep only recent context
                    if input_ids.size(1) > self.max_len:
                        input_ids = input_ids[:, -self.max_len:]
                        
                    # Early stopping
                    if token_id == self.tokenizer.token_to_id.get("<EOS>", -1):
                        break
            
            result = self.tokenizer.decode(generated)
            return result if result.strip() else "I understand your query. Let me help you with that."
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"I encountered an issue while processing your request. How can I help you differently?"

# ===================== REASONING SYSTEM =====================
@dataclass
class ThoughtStep:
    type: str  # "observation", "hypothesis", "action", "validation"
    content: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

class SimpleReasoner:
    def __init__(self, model):
        self.model = model
        
    def generate_plan(self, query: str, context: str = "", stream_callback=None):
        """Generate a reasoning plan"""
        thoughts = []
        
        # Initial observation
        initial_thought = ThoughtStep(
            type="observation",
            content=f"Analyzing query: {query}",
            confidence=1.0,
            timestamp=datetime.now()
        )
        thoughts.append(initial_thought)
        
        if stream_callback:
            thought_dict = asdict(initial_thought)
            thought_dict['timestamp'] = thought_dict['timestamp'].isoformat()
            asyncio.create_task(stream_callback({"type": "thought", "data": thought_dict}))
        
        # Generate hypothesis
        hypothesis_thought = ThoughtStep(
            type="hypothesis",
            content=f"This appears to be a request about: {query[:50]}...",
            confidence=0.8,
            timestamp=datetime.now()
        )
        thoughts.append(hypothesis_thought)
        
        if stream_callback:
            thought_dict = asdict(hypothesis_thought)
            thought_dict['timestamp'] = thought_dict['timestamp'].isoformat()
            asyncio.create_task(stream_callback({"type": "thought", "data": thought_dict}))
        
        # Generate action
        action_thought = ThoughtStep(
            type="action",
            content="I should provide a helpful response based on my understanding.",
            confidence=0.9,
            timestamp=datetime.now()
        )
        thoughts.append(action_thought)
        
        if stream_callback:
            thought_dict = asdict(action_thought)
            thought_dict['timestamp'] = thought_dict['timestamp'].isoformat()
            asyncio.create_task(stream_callback({"type": "thought", "data": thought_dict}))
        
        return thoughts

# ===================== MEMORY SYSTEM =====================
class SimpleMemory:
    def __init__(self, db_path="memory/maran_memory.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def store(self, content: str, memory_type: str = "conversation", context: str = "", importance: float = 0.5):
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (timestamp, type, content, context, importance)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, memory_type, content, context, importance))
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id

    def query(self, query_text: str, limit: int = 5, memory_type: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = '''
            SELECT * FROM memories 
            WHERE content LIKE ? 
            {} 
            ORDER BY importance DESC, access_count DESC 
            LIMIT ?
        '''.format('AND type = ?' if memory_type else '')
        
        params = [f'%{query_text}%']
        if memory_type:
            params.append(memory_type)
        params.append(limit)
        
        cursor.execute(sql, params)
        memories = cursor.fetchall()
        conn.close()
        
        return memories

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM memories')
        total_memories = cursor.fetchone()[0]
        
        cursor.execute('SELECT type, COUNT(*) FROM memories GROUP BY type')
        type_counts = dict(cursor.fetchall())
        
        cursor.execute('SELECT AVG(importance) FROM memories')
        avg_importance = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_memories": total_memories,
            "type_distribution": type_counts,
            "average_importance": avg_importance
        }

# ===================== TOOL SYSTEM =====================
class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.last_used = None

    def execute(self, *args, **kwargs):
        self.usage_count += 1
        self.last_used = datetime.now()
        return self._execute(*args, **kwargs)

    def _execute(self, *args, **kwargs):
        raise NotImplementedError

    def get_stats(self):
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }

class TerminalTool(BaseTool):
    def __init__(self):
        super().__init__("terminal", "Execute terminal commands safely")
        
    def _execute(self, command: str, **kwargs):
        # Security validation
        for pattern in CONFIG["safety"]["forbidden_patterns"]:
            if re.search(pattern, command, re.IGNORECASE):
                return {
                    "success": False,
                    "error": f"Command blocked for security: {pattern}",
                    "output": "",
                    "return_code": 1
                }

        try:
            # Safe command execution
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=CONFIG["safety"]["max_execution_time"],
                cwd="/tmp"  # Run in safe directory
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timeout",
                "output": "",
                "return_code": 124
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "return_code": 1
            }

class CodeExecutorTool(BaseTool):
    def __init__(self):
        super().__init__("code_executor", "Execute Python code safely")
        
    def _execute(self, code: str, **kwargs):
        # Security validation
        dangerous_imports = ['os', 'sys', 'subprocess', 'shutil', 'glob']
        for imp in dangerous_imports:
            if f'import {imp}' in code or f'from {imp}' in code:
                return {
                    "success": False,
                    "error": f"Import '{imp}' not allowed for security",
                    "output": "",
                    "execution_time": 0
                }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            start_time = time.time()
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=CONFIG["safety"]["max_execution_time"]
            )
            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": execution_time,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout ({CONFIG['safety']['max_execution_time']}s)",
                "output": "",
                "execution_time": CONFIG["safety"]["max_execution_time"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0
            }
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

# ===================== MONITORING SYSTEM =====================
class SimpleMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.system_stats = {
            "total_requests": 0,
            "total_generations": 0,
            "total_memories": 0,
            "uptime": 0,
            "errors": 0
        }
        
    def track_request(self):
        self.system_stats["total_requests"] += 1
        
    def track_generation(self):
        self.system_stats["total_generations"] += 1
        
    def track_error(self):
        self.system_stats["errors"] += 1
        
    def get_stats(self):
        self.system_stats["uptime"] = time.time() - self.start_time
        return self.system_stats.copy()

# ===================== MAIN AI AGENT =====================
class MaranAgent:
    def __init__(self):
        self.tokenizer = SimpleTokenizer(vocab_size=CONFIG["model"]["vocab_size"])
        self.model = SimpleTransformerModel(
            tokenizer=self.tokenizer,
            d_model=CONFIG["model"]["d_model"],
            nhead=CONFIG["model"]["nhead"],
            num_layers=CONFIG["model"]["num_layers"],
            max_len=CONFIG["model"]["max_len"],
            dropout=CONFIG["model"]["dropout"]
        )
        self.memory = SimpleMemory()
        self.reasoner = SimpleReasoner(self.model)
        self.tools = {
            "terminal": TerminalTool(),
            "code_executor": CodeExecutorTool()
        }
        self.monitor = SimpleMonitor()
        self.is_initialized = True
        
        logger.info("🧠 Maran Agent initialized successfully!")

    async def process_query(self, query: str, stream_callback=None):
        """Process a user query with full AI pipeline"""
        self.monitor.track_request()
        start_time = time.time()
        
        try:
            # Store query in memory
            memory_id = self.memory.store(f"User query: {query}", "query", importance=0.8)
            
            # Generate reasoning plan
            if stream_callback:
                await stream_callback({"type": "status", "data": "🧠 Analyzing query..."})
            
            thoughts = self.reasoner.generate_plan(query, stream_callback=stream_callback)
            
            # Extract relevant memories
            if stream_callback:
                await stream_callback({"type": "status", "data": "🔍 Searching memory..."})
            
            relevant_memories = self.memory.query(query, limit=3)
            
            # Generate response
            if stream_callback:
                await stream_callback({"type": "status", "data": "💭 Generating response..."})
            
            # Generate response with streaming
            response = ""
            if stream_callback:
                def response_stream(partial_text, step, total_steps):
                    asyncio.create_task(stream_callback({
                        "type": "generation", 
                        "data": {
                            "text": partial_text,
                            "progress": step / total_steps if total_steps > 0 else 0
                        }
                    }))
                
                response = self.model.generate(
                    query, 
                    max_length=80, 
                    temperature=0.7,
                    stream_callback=response_stream
                )
            else:
                response = self.model.generate(query, max_length=80, temperature=0.7)
            
            # Store response in memory
            self.memory.store(f"AI response: {response}", "response", importance=0.7)
            
            # Check if any tools should be used
            tool_results = await self._analyze_for_tools(query, response, stream_callback)
            
            self.monitor.track_generation()
            
            # Calculate response time
            response_time = time.time() - start_time
            
            return {
                "response": response,
                "thoughts": [asdict(t) for t in thoughts],
                "tool_results": tool_results,
                "relevant_memories": len(relevant_memories),
                "response_time": response_time,
                "memory_id": memory_id
            }
            
        except Exception as e:
            self.monitor.track_error()
            logger.error(f"Error processing query: {e}")
            if stream_callback:
                await stream_callback({
                    "type": "error", 
                    "data": {"message": f"Error processing query: {str(e)}"}
                })
            return {
                "response": f"I encountered an error while processing your query. Please try rephrasing your question.",
                "thoughts": [],
                "tool_results": {},
                "relevant_memories": 0,
                "response_time": time.time() - start_time,
                "error": str(e)
            }

    async def _analyze_for_tools(self, query: str, response: str, stream_callback=None):
        """Analyze if any tools should be used"""
        tool_results = {}
        query_lower = query.lower()
        
        # Terminal commands
        if any(keyword in query_lower for keyword in ["terminal", "command", "run", "execute", "shell"]):
            if stream_callback:
                await stream_callback({"type": "status", "data": "🖥️ Processing terminal command..."})
            
            # Extract command (simple heuristic)
            command_match = re.search(r'(?:run|execute|terminal)\s+(.+)', query_lower)
            if command_match:
                command = command_match.group(1).strip()
                result = self.tools["terminal"].execute(command)
                tool_results["terminal"] = result
        
        # Code execution
        if any(keyword in query_lower for keyword in ["code", "python", "script"]) and ("```" in query or "print" in query_lower):
            if stream_callback:
                await stream_callback({"type": "status", "data": "🐍 Executing Python code..."})
            
            # Extract code from query or response
            code_match = re.search(r'```python\n(.*?)\n```', query + response, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\n(.*?)\n```', query + response, re.DOTALL)
            
            if code_match:
                code = code_match.group(1)
                result = self.tools["code_executor"].execute(code)
                tool_results["code_execution"] = result
        
        return tool_results

    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            "agent_status": "initialized" if self.is_initialized else "initializing",
            "model_info": {
                "vocab_size": len(self.tokenizer.token_to_id),
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "device": str(self.model.device)
            },
            "memory_stats": self.memory.get_stats(),
            "tool_stats": {name: tool.get_stats() for name, tool in self.tools.items()},
            "monitor_stats": self.monitor.get_stats(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

# ===================== WEB APPLICATION =====================
app = FastAPI(title="Maran AI Agent - Enhanced", description="Enhanced AI Agent with Dark/Light Mode and Terminal")
agent = MaranAgent()

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Enhanced HTML Template with Dark/Light Mode and Terminal
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Maran AI Agent - Enhanced</title>
    <style>
        :root {
            --primary-bg: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --panel-bg: rgba(255, 255, 255, 0.1);
            --panel-border: rgba(255, 255, 255, 0.2);
            --text-color: #fff;
            --accent-color: #ffd700;
            --button-bg: #28a745;
            --button-hover: #218838;
            --input-bg: rgba(255, 255, 255, 0.2);
            --terminal-bg: rgba(0, 0, 0, 0.3);
            --terminal-text: #00ff00;
        }
        
        [data-theme="light"] {
            --primary-bg: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            --panel-bg: rgba(255, 255, 255, 0.9);
            --panel-border: rgba(0, 0, 0, 0.1);
            --text-color: #333;
            --accent-color: #1976d2;
            --button-bg: #1976d2;
            --button-hover: #1565c0;
            --input-bg: rgba(0, 0, 0, 0.1);
            --terminal-bg: rgba(0, 0, 0, 0.8);
            --terminal-text: #00ff00;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--primary-bg);
            color: var(--text-color);
            height: 100vh;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--panel-bg);
            border-radius: 10px;
            padding: 15px 20px;
            margin: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid var(--panel-border);
        }
        
        .header h1 {
            margin: 0;
        }
        
        .theme-toggle {
            background: var(--button-bg);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .theme-toggle:hover {
            background: var(--button-hover);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            height: calc(100vh - 100px);
            gap: 10px;
            padding: 0 10px 10px 10px;
        }
        
        .panel {
            background: var(--panel-bg);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid var(--panel-border);
            overflow-y: auto;
            transition: all 0.3s ease;
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: var(--accent-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 5px;
        }
        
        .chat-panel {
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 15px;
            padding: 10px;
            background: var(--input-bg);
            border-radius: 8px;
            max-height: 300px;
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
            background: var(--input-bg);
            color: var(--text-color);
            font-size: 16px;
        }
        
        #queryInput::placeholder {
            color: rgba(128, 128, 128, 0.7);
        }
        
        #sendButton {
            padding: 12px 24px;
            background: var(--button-bg);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        #sendButton:hover:not(:disabled) {
            background: var(--button-hover);
        }
        
        #sendButton:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .terminal-panel {
            font-family: 'Courier New', monospace;
        }
        
        .terminal {
            background: var(--terminal-bg);
            color: var(--terminal-text);
            padding: 15px;
            border-radius: 8px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            margin-bottom: 15px;
        }
        
        .terminal-input {
            display: flex;
            gap: 10px;
        }
        
        .terminal-input input {
            flex: 1;
            background: var(--terminal-bg);
            color: var(--terminal-text);
            border: 1px solid var(--terminal-text);
            padding: 8px;
            font-family: 'Courier New', monospace;
            border-radius: 4px;
        }
        
        .terminal-input button {
            background: var(--terminal-text);
            color: black;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
        }
        
        .thought-item {
            background: var(--input-bg);
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
            background: var(--input-bg);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent-color);
        }
        
        .metric-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: var(--input-bg);
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
    </style>
</head>
<body>
    <!-- Header with Theme Toggle -->
    <div class="header">
        <h1>🧠 Maran AI Agent - Enhanced</h1>
        <div class="progress-bar">
            <div class="progress-fill" id="progressBar"></div>
        </div>
        <button class="theme-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
    </div>
    
    <div class="dashboard">
        <!-- Chat Panel -->
        <div class="panel chat-panel">
            <h3>💬 AI Conversation</h3>
            <div class="chat-messages" id="chatMessages"></div>
            <div class="input-area">
                <input type="text" id="queryInput" placeholder="Ask me anything..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <!-- Terminal Panel -->
        <div class="panel terminal-panel">
            <h3>🖥️ Terminal</h3>
            <div class="terminal" id="terminal"></div>
            <div class="terminal-input">
                <input type="text" id="terminalInput" placeholder="Enter command..." />
                <button onclick="executeCommand()">Run</button>
            </div>
        </div>
        
        <!-- Thoughts Panel -->
        <div class="panel">
            <h3>🧠 AI Reasoning</h3>
            <div id="thoughtsContainer"></div>
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
    </div>

    <script>
        class MaranEnhancedInterface {
            constructor() {
                this.ws = null;
                this.isConnected = false;
                this.currentThoughts = [];
                this.theme = localStorage.getItem('theme') || 'dark';
                this.init();
            }
            
            init() {
                this.setTheme(this.theme);
                this.connectWebSocket();
                this.setupEventListeners();
                this.updateSystemStatus();
                this.addTerminalWelcome();
                
                // Update metrics every 5 seconds
                setInterval(() => this.updateSystemStatus(), 5000);
            }
            
            setTheme(theme) {
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('theme', theme);
                this.theme = theme;
            }
            
            toggleTheme() {
                this.setTheme(this.theme === 'dark' ? 'light' : 'dark');
            }
            
            connectWebSocket() {
                const wsUrl = `ws://${window.location.host}/ws`;
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.isConnected = true;
                    this.addMessage('system', '🔌 Connected to Maran AI Agent', 'status');
                    this.addTerminalLine('System: WebSocket connected');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    this.isConnected = false;
                    this.addMessage('system', '🔌 Disconnected from agent', 'error');
                    this.addTerminalLine('System: WebSocket disconnected');
                    // Attempt to reconnect after 3 seconds
                    setTimeout(() => this.connectWebSocket(), 3000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.addMessage('system', '❌ Connection error', 'error');
                    this.addTerminalLine('System: Connection error');
                };
            }
            
            setupEventListeners() {
                const queryInput = document.getElementById('queryInput');
                const sendButton = document.getElementById('sendButton');
                const terminalInput = document.getElementById('terminalInput');
                
                queryInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendQuery();
                    }
                });
                
                terminalInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        this.executeCommand();
                    }
                });
                
                sendButton.addEventListener('click', () => this.sendQuery());
            }
            
            addTerminalWelcome() {
                this.addTerminalLine('Maran AI Agent Terminal v1.0');
                this.addTerminalLine('Type commands or ask AI to execute them');
                this.addTerminalLine('Example: "run ls -la" or "execute echo hello"');
                this.addTerminalLine('---');
            }
            
            addTerminalLine(text) {
                const terminal = document.getElementById('terminal');
                const line = document.createElement('div');
                line.textContent = new Date().toLocaleTimeString() + ' ' + text;
                terminal.appendChild(line);
                terminal.scrollTop = terminal.scrollHeight;
            }
            
            executeCommand() {
                const terminalInput = document.getElementById('terminalInput');
                const command = terminalInput.value.trim();
                
                if (!command) return;
                
                this.addTerminalLine(`$ ${command}`);
                terminalInput.value = '';
                
                // Send command via AI or direct execution
                if (this.isConnected) {
                    const message = {
                        type: 'query',
                        data: { query: `terminal ${command}` }
                    };
                    this.ws.send(JSON.stringify(message));
                }
            }
            
            sendQuery() {
                const queryInput = document.getElementById('queryInput');
                const query = queryInput.value.trim();
                
                if (!query || !this.isConnected) return;
                
                // Clear input and disable button
                queryInput.value = '';
                document.getElementById('sendButton').disabled = true;
                
                // Add user message
                this.addMessage('user', query, 'user');
                
                // Clear previous thoughts
                this.currentThoughts = [];
                document.getElementById('thoughtsContainer').innerHTML = '';
                
                // Send query via WebSocket
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
                messageDiv.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 5px;">
                        ${sender === 'user' ? '👤 You' : sender === 'ai' ? '🤖 Maran' : '⚙️ System'} 
                        <span style="font-size: 12px; opacity: 0.7;">${timestamp}</span>
                    </div>
                    <div>${content}</div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            addThought(thoughtData) {
                this.currentThoughts.push(thoughtData);
                
                const thoughtsContainer = document.getElementById('thoughtsContainer');
                const thoughtDiv = document.createElement('div');
                thoughtDiv.className = `thought-item thought-${thoughtData.type}`;
                
                const timestamp = new Date(thoughtData.timestamp).toLocaleTimeString();
                thoughtDiv.innerHTML = `
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 3px;">
                        ${thoughtData.type.toUpperCase()} - ${timestamp}
                    </div>
                    <div>${thoughtData.content}</div>
                    <div style="font-size: 11px; opacity: 0.6; margin-top: 3px;">
                        Confidence: ${(thoughtData.confidence * 100).toFixed(1)}%
                    </div>
                `;
                
                thoughtsContainer.appendChild(thoughtDiv);
                thoughtsContainer.scrollTop = thoughtsContainer.scrollHeight;
            }
            
            updateGenerationProgress(data) {
                const progressBar = document.getElementById('progressBar');
                progressBar.style.width = `${data.progress * 100}%`;
                
                // Show partial text if available
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
                
                for (const [toolName, result] of Object.entries(toolResults)) {
                    const status = result.success ? '✅' : '❌';
                    let output = result.success ? 
                        (result.output || result.stdout || 'Success') : 
                        (result.error || 'Failed');
                    
                    // Add to terminal if it's a terminal command
                    if (toolName === 'terminal') {
                        this.addTerminalLine(`${status} ${output}`);
                        if (result.error) {
                            this.addTerminalLine(`Error: ${result.error}`);
                        }
                    }
                    
                    // Add to chat as well
                    let resultsHtml = `<strong>🔧 ${toolName}:</strong><br>`;
                    resultsHtml += `<div style="background: rgba(0,0,0,0.2); margin: 5px 0; padding: 8px; border-radius: 4px; font-family: monospace;">`;
                    resultsHtml += `${status} ${output}`;
                    resultsHtml += `</div>`;
                    
                    this.addMessage('system', resultsHtml, 'status');
                }
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
                // Update metric cards
                document.getElementById('totalRequests').textContent = data.monitor_stats.total_requests;
                document.getElementById('totalGenerations').textContent = data.monitor_stats.total_generations;
                document.getElementById('uptime').textContent = `${Math.floor(data.monitor_stats.uptime)}s`;
                document.getElementById('memoryCount').textContent = data.memory_stats.total_memories;
                
                // Update system info
                const systemInfo = document.getElementById('systemInfo');
                systemInfo.innerHTML = `
                    <div style="font-size: 12px; margin-top: 10px;">
                        <div>💾 Memory: ${(data.system_info.memory_available / 1024 / 1024 / 1024).toFixed(1)}GB available</div>
                        <div>🖥️ CPU: ${data.system_info.cpu_count} cores</div>
                        <div>🎯 Model: ${data.model_info.model_parameters.toLocaleString()} parameters</div>
                        <div>📚 Vocab: ${data.model_info.vocab_size} tokens</div>
                    </div>
                `;
            }
        }
        
        // Global functions
        function toggleTheme() {
            window.maranInterface.toggleTheme();
        }
        
        function executeCommand() {
            window.maranInterface.executeCommand();
        }
        
        // Initialize interface when page loads
        window.addEventListener('DOMContentLoaded', () => {
            window.maranInterface = new MaranEnhancedInterface();
        });
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard"""
    return HTML_TEMPLATE

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return agent.get_system_status()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                query = message["data"]["query"]
                
                # Process query with streaming callback
                async def stream_callback(data):
                    try:
                        await websocket.send_text(json.dumps(data))
                    except:
                        pass  # Connection might be closed
                
                # Process the query
                result = await agent.process_query(query, stream_callback)
                
                # Send final response
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
        if websocket in active_connections:
            active_connections.remove(websocket)

def main():
    """Main entry point"""
    print("🧠 Starting Maran AI Agent - Enhanced Web Interface...")
    print("=" * 70)
    print(f"🌐 Web Interface: http://localhost:{CONFIG['web']['port']}")
    print("🔧 Features:")
    print("   • Fixed AI processing with error handling")
    print("   • Real-time conversation with thought streaming")  
    print("   • Dark/Light mode toggle")
    print("   • Integrated terminal interface")
    print("   • Enhanced monitoring and analytics")
    print("=" * 70)
    
    # Configure uvicorn for production-ready deployment
    uvicorn.run(
        app,
        host=CONFIG["web"]["host"],
        port=CONFIG["web"]["port"],
        log_level="info",
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    main()