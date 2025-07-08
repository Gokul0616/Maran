#!/usr/bin/env python3
"""
🧠 Maran Web Interface - Complete Autonomous AI Agent System
===============================================================

A comprehensive web-based interface for the Maran AI agent system featuring:
- Multi-panel sophisticated dashboard
- Real-time streaming of AI thought processes
- Hardware and software tool integrations
- Advanced monitoring and analytics
- Self-improvement capabilities
- Memory management and conversation history

Usage: python3 maran_web_complete.py
Access: http://localhost:8000
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
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import psutil

# Web Framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Monitoring
from prometheus_client import Gauge, Counter, Histogram, start_http_server, CollectorRegistry

# Additional imports for complete functionality
FULL_FEATURES = True
optional_imports = {}

# Try importing optional dependencies
try:
    import faiss
    optional_imports['faiss'] = faiss
except ImportError:
    FULL_FEATURES = False
    print("⚠️  FAISS not available - vector similarity search disabled")

try:
    import pickle
    optional_imports['pickle'] = pickle
except ImportError:
    pass

try:
    from datasets import load_dataset
    import itertools
    optional_imports['datasets'] = True
except ImportError:
    print("⚠️  Datasets library not available - will use sample data for training")
    FULL_FEATURES = False

try:
    from gpiozero import LED, Button, Servo
    optional_imports['gpio'] = True
except ImportError:
    print("⚠️  GPIO not available - hardware tools disabled")

try:
    # Desktop automation disabled for headless environment
    print("⚠️  Desktop automation disabled in headless environment")
    pass
except Exception as e:
    print(f"⚠️  Desktop automation setup failed: {e}")

try:
    import requests
    optional_imports['requests'] = requests
except ImportError:
    print("⚠️  Requests not available - web requests disabled")

print(f"ℹ️  Available features: {list(optional_imports.keys())}")

# ===================== CONFIGURATION =====================
CONFIG = {
    "model": {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "vocab_size": 10000,
        "max_len": 1024,
        "dropout": 0.1
    },
    "training": {
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 3,
        "grad_accum_steps": 4
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8000,
        "metrics_port": 9090
    },
    "hardware": {
        "led_pin": 17,
        "servo_pin": 18,
        "sensor_pin": 27
    },
    "safety": {
        "max_execution_time": 10,
        "memory_limit_mb": 512,
        "forbidden_patterns": [
            r"os\.system", r"subprocess\.call", r"open\(.*['\"]w['\"]", 
            r"import\s+(os|sys|subprocess)", r"__import__", r"eval\("
        ]
    }
}

# ===================== LOGGING SETUP =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maran_web")

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("memory", exist_ok=True)

# ===================== CUSTOM BPE TOKENIZER =====================
class MaranBPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.token_to_id = {}
        self.id_to_token = {}
        self.bpe_ranks = {}
        
        # Initialize special tokens
        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = pattern.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def build_vocab(self, texts):
        logger.info("Training BPE tokenizer...")
        corpus = Counter()
        for line in texts:
            words = line.strip().split()
            for word in words:
                byte_word = ' '.join([f'{b}' for b in word.encode('utf-8')]) + ' </w>'
                corpus[byte_word] += 1

        vocab_size = len(self.token_to_id)
        while vocab_size < self.vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.bpe_ranks[best] = len(self.bpe_ranks)
            vocab_size += 1

        # Extract vocab
        tokens = set()
        for word in corpus:
            tokens.update(word.split())
        for token in sorted(tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        logger.info(f"Tokenizer trained. Vocab size: {len(self.token_to_id)}")

    def bpe(self, word):
        word = list(word) + ['</w>']
        word = tuple(word)
        pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        while True:
            bigram = min(
                ((pair, self.bpe_ranks.get(pair, float('inf'))) for pair in pairs),
                key=lambda x: x[1],
                default=((None, None), None)
            )[0]
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    if j < len(word) - 1 and word[j + 1] == second:
                        new_word.append(first + second)
                        i = j + 2
                    else:
                        new_word.append(word[i])
                        i += 1
                except:
                    new_word.extend(word[i:])
                    break
            word = tuple(new_word)
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        return word

    def encode(self, text, max_length=None):
        if not text.strip():
            return [self.token_to_id["<PAD>"]]
            
        tokens = [self.token_to_id["<BOS>"]]
        for word in text.strip().split():
            byte_word = ' '.join([f'{b}' for b in word.encode('utf-8')])
            for bpe_token in self.bpe(byte_word):
                tokens.append(self.token_to_id.get(bpe_token, self.token_to_id["<UNK>"]))
        tokens.append(self.token_to_id["<EOS>"])
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in token_ids]
        text = []
        for token in tokens:
            if skip_special_tokens and token in self.special_tokens:
                continue
            try:
                if token == "<UNK>":
                    text.append("?")
                else:
                    decoded = bytes([int(t) for t in token.split() if t.isdigit()]).decode('utf-8', errors='ignore')
                    text.append(decoded.replace('</w>', ''))
            except:
                text.append("?")
        return ' '.join(text).strip()

    def save(self, path):
        serialized_ranks = {'|'.join(k): v for k, v in self.bpe_ranks.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "bpe_ranks": serialized_ranks,
                "config": {
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens
                }
            }, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token_to_id = data['token_to_id']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.bpe_ranks = {tuple(k.split('|')): v for k, v in data['bpe_ranks'].items()}
        config = data.get('config', {})
        self.vocab_size = config.get('vocab_size', len(self.token_to_id))
        self.special_tokens = config.get('special_tokens', ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

# ===================== TRANSFORMER MODEL =====================
class OptimizedAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch's optimized attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.1, is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_output)

class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = OptimizedAttention(d_model, nhead)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out = self.attn(self.attn_norm(x), attn_mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ff_norm(x))
        return x + self.dropout(ff_out)

class MaranGPTModel(nn.Module):
    def __init__(self, tokenizer, d_model=768, nhead=12, num_layers=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_embed = nn.Embedding(len(tokenizer.token_to_id), d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, len(tokenizer.token_to_id))
        self.max_len = max_len
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.token_embed(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        position_embeddings = self.pos_embed(position_ids)
        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)

    def generate(self, prompt: str, max_length=100, temperature=1.0, top_k=40, stream_callback=None):
        """Generate text with optional streaming callback for real-time updates"""
        self.eval()
        tokens = self.tokenizer.encode(prompt)
        tokens = tokens[-self.max_len:] if len(tokens) > self.max_len else tokens
        input_ids = torch.tensor([tokens], device=self.device)
        
        generated = []
        for step in range(max_length):
            with torch.no_grad():
                outputs = self(input_ids)
                next_token = self._sample(outputs[0, -1], temperature, top_k)
                generated.append(next_token.item())
                
                # Stream intermediate results
                if stream_callback and step % 5 == 0:  # Update every 5 tokens
                    partial_text = self.tokenizer.decode(generated)
                    stream_callback(partial_text, step, max_length)
                
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                
                # Keep only recent context
                if input_ids.size(1) > self.max_len:
                    input_ids = input_ids[:, -self.max_len:]
                    
                # Early stopping
                if next_token.item() == self.tokenizer.token_to_id.get("<EOS>", -1):
                    break
        
        return self.tokenizer.decode(generated)

    def _sample(self, logits, temperature, top_k):
        logits = logits / temperature
        if top_k > 0:
            topk = torch.topk(logits, top_k)
            logits[logits < topk.values[-1]] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

# ===================== REASONING SYSTEM =====================
@dataclass
class ThoughtStep:
    type: str  # "observation", "hypothesis", "action", "validation"
    content: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

class TreeOfThoughtReasoner:
    def __init__(self, model, max_depth=3, candidates_per_step=3):
        self.model = model
        self.max_depth = max_depth
        self.candidates_per_step = candidates_per_step
        
    def generate_plan(self, query: str, context: str = "", stream_callback=None):
        """Generate a reasoning plan with real-time streaming"""
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
            stream_callback({"type": "thought", "data": asdict(initial_thought)})
        
        # Generate reasoning chain
        current_prompt = f"Query: {query}\nContext: {context}\n\nThinking step by step:"
        
        for depth in range(self.max_depth):
            # Generate hypothesis
            hypothesis_prompt = f"{current_prompt}\n\nGenerate a hypothesis about how to solve this:"
            hypothesis = self.model.generate(hypothesis_prompt, max_length=50, temperature=0.7)
            
            hypothesis_thought = ThoughtStep(
                type="hypothesis",
                content=hypothesis,
                confidence=0.8,
                timestamp=datetime.now()
            )
            thoughts.append(hypothesis_thought)
            
            if stream_callback:
                stream_callback({"type": "thought", "data": asdict(hypothesis_thought)})
            
            # Generate action
            action_prompt = f"{current_prompt}\n\nHypothesis: {hypothesis}\n\nWhat specific action should I take?"
            action = self.model.generate(action_prompt, max_length=80, temperature=0.6)
            
            action_thought = ThoughtStep(
                type="action",
                content=action,
                confidence=0.9,
                timestamp=datetime.now()
            )
            thoughts.append(action_thought)
            
            if stream_callback:
                stream_callback({"type": "thought", "data": asdict(action_thought)})
            
            current_prompt += f"\nStep {depth + 1}: {hypothesis} -> {action}"
        
        return thoughts

# ===================== MEMORY SYSTEM =====================
class MaranMemory:
    def __init__(self, model, tokenizer, db_path="memory/maran_memory.db"):
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self._init_database()
        
        # FAISS index for vector similarity search
        if 'faiss' in optional_imports:
            try:
                self.dimension = model.d_model
                self.index = optional_imports['faiss'].IndexFlatL2(self.dimension)
                self.memory_vectors = []
            except:
                self.index = None
                logger.warning("FAISS initialization failed, using text-based memory only")
        else:
            self.index = None

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
        """Store a memory with vector embedding if available"""
        timestamp = datetime.now().isoformat()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (timestamp, type, content, context, importance)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, memory_type, content, context, importance))
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Store vector embedding if FAISS is available
        if self.index is not None:
            try:
                embedding = self._get_embedding(content)
                self.index.add(embedding)
                self.memory_vectors.append(memory_id)
            except Exception as e:
                logger.warning(f"Failed to store vector embedding: {e}")
        
        return memory_id

    def query(self, query_text: str, limit: int = 5, memory_type: str = None):
        """Query memories with hybrid text/vector search"""
        memories = []
        
        # Vector similarity search if available
        if self.index is not None and len(self.memory_vectors) > 0:
            try:
                query_embedding = self._get_embedding(query_text)
                _, indices = self.index.search(query_embedding, min(limit, len(self.memory_vectors)))
                memory_ids = [self.memory_vectors[i] for i in indices[0] if i < len(self.memory_vectors)]
                
                # Fetch from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                placeholders = ','.join(['?' for _ in memory_ids])
                query_sql = f'''
                    SELECT * FROM memories WHERE id IN ({placeholders})
                    ORDER BY importance DESC, access_count DESC
                '''
                cursor.execute(query_sql, memory_ids)
                memories = cursor.fetchall()
                conn.close()
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text search: {e}")
        
        # Fallback to text-based search
        if not memories:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            query_sql = '''
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
            
            cursor.execute(query_sql, params)
            memories = cursor.fetchall()
            conn.close()
        
        # Update access counts
        if memories:
            memory_ids = [m[0] for m in memories]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for mid in memory_ids:
                cursor.execute('UPDATE memories SET access_count = access_count + 1 WHERE id = ?', (mid,))
            conn.commit()
            conn.close()
        
        return memories

    def _get_embedding(self, text: str):
        """Get embedding vector for text"""
        tokens = self.tokenizer.encode(text, max_length=512)
        input_ids = torch.tensor([tokens], device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Use mean pooling of last layer
            embedding = outputs.mean(dim=1).cpu().numpy().astype('float32')
        
        return embedding

    def get_stats(self):
        """Get memory statistics"""
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
            "average_importance": avg_importance,
            "vector_index_size": len(self.memory_vectors) if self.index else 0
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

class CodeExecutorTool(BaseTool):
    def __init__(self):
        super().__init__("code_executor", "Execute Python code safely")
        self.timeout = CONFIG["safety"]["max_execution_time"]

    def _execute(self, code: str, **kwargs):
        """Execute code with safety checks"""
        # Security validation
        for pattern in CONFIG["safety"]["forbidden_patterns"]:
            if re.search(pattern, code):
                return {
                    "success": False,
                    "error": f"Forbidden pattern detected: {pattern}",
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
                timeout=self.timeout
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
                "error": f"Execution timeout ({self.timeout}s)",
                "output": "",
                "execution_time": self.timeout
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

class ShellTool(BaseTool):
    def __init__(self):
        super().__init__("shell", "Execute shell commands")
        
    def _execute(self, command: str, **kwargs):
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=CONFIG["safety"]["max_execution_time"]
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timeout",
                "stdout": "",
                "stderr": ""
            }

class WebRequestTool(BaseTool):
    def __init__(self):
        super().__init__("web_request", "Make HTTP requests")
        
    def _execute(self, method: str, url: str, **kwargs):
        try:
            if 'requests' in optional_imports:
                response = optional_imports['requests'].request(method, url, timeout=10, **kwargs)
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.text[:1000],  # Limit content size
                    "json": response.json() if response.headers.get('content-type', '').startswith('application/json') else None
                }
            else:
                return {"success": False, "error": "Web requests not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Hardware tools (only if GPIO is available)
if 'gpio' in optional_imports:
    class LEDTool(BaseTool):
        def __init__(self, pin=17):
            super().__init__("led", f"Control LED on GPIO pin {pin}")
            try:
                self.led = LED(pin)
                self.available = True
            except:
                self.available = False
                logger.warning(f"LED on pin {pin} not available")
            
        def _execute(self, state: str, **kwargs):
            if not self.available:
                return {"success": False, "error": "LED not available"}
            
            try:
                if state.lower() == "on":
                    self.led.on()
                elif state.lower() == "off":
                    self.led.off()
                else:
                    return {"success": False, "error": "Invalid state. Use 'on' or 'off'"}
                
                return {"success": True, "state": state, "pin": self.led.pin.number}
            except Exception as e:
                return {"success": False, "error": str(e)}

    class ServoTool(BaseTool):
        def __init__(self, pin=18):
            super().__init__("servo", f"Control servo on GPIO pin {pin}")
            try:
                self.servo = Servo(pin)
                self.available = True
            except:
                self.available = False
                logger.warning(f"Servo on pin {pin} not available")
            
        def _execute(self, angle: float, **kwargs):
            if not self.available:
                return {"success": False, "error": "Servo not available"}
            
            try:
                # Convert angle to servo value (-1 to 1)
                servo_value = max(-1, min(1, angle / 90.0))
                self.servo.value = servo_value
                return {"success": True, "angle": angle, "servo_value": servo_value}
            except Exception as e:
                return {"success": False, "error": str(e)}

    class SensorTool(BaseTool):
        def __init__(self, pin=27):
            super().__init__("sensor", f"Read digital sensor on GPIO pin {pin}")
            try:
                self.sensor = Button(pin)
                self.available = True
            except:
                self.available = False
                logger.warning(f"Sensor on pin {pin} not available")
            
        def _execute(self, **kwargs):
            if not self.available:
                return {"success": False, "error": "Sensor not available"}
            
            try:
                return {
                    "success": True,
                    "is_pressed": self.sensor.is_pressed,
                    "pin": self.sensor.pin.number
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
else:
    # Dummy implementations when GPIO is not available
    class LEDTool(BaseTool):
        def __init__(self, pin=17):
            super().__init__("led", f"LED simulation (pin {pin})")
            self.available = False
            
        def _execute(self, state: str, **kwargs):
            return {"success": False, "error": "GPIO not available - LED simulation mode"}

    class ServoTool(BaseTool):
        def __init__(self, pin=18):
            super().__init__("servo", f"Servo simulation (pin {pin})")
            self.available = False
            
        def _execute(self, angle: float, **kwargs):
            return {"success": False, "error": "GPIO not available - Servo simulation mode"}

    class SensorTool(BaseTool):
        def __init__(self, pin=27):
            super().__init__("sensor", f"Sensor simulation (pin {pin})")
            self.available = False
            
        def _execute(self, **kwargs):
            return {"success": False, "error": "GPIO not available - Sensor simulation mode"}

# ===================== MONITORING SYSTEM =====================
class MaranMonitor:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.start_time = time.time()
        
        # Metrics
        self.request_count = Counter('maran_requests_total', 'Total requests', registry=self.registry)
        self.response_time = Histogram('maran_response_seconds', 'Response time', registry=self.registry)
        self.memory_usage = Gauge('maran_memory_bytes', 'Memory usage', registry=self.registry)
        self.cpu_usage = Gauge('maran_cpu_percent', 'CPU usage', registry=self.registry)
        self.generation_count = Counter('maran_generations_total', 'Total generations', registry=self.registry)
        self.tool_usage = Counter('maran_tool_usage_total', 'Tool usage', ['tool_name'], registry=self.registry)
        
        # System stats
        self.system_stats = {
            "total_requests": 0,
            "total_generations": 0,
            "total_memories": 0,
            "uptime": 0,
            "errors": 0
        }
        
    def track_request(self):
        self.request_count.inc()
        self.system_stats["total_requests"] += 1
        
    def track_generation(self):
        self.generation_count.inc()
        self.system_stats["total_generations"] += 1
        
    def track_tool_usage(self, tool_name: str):
        self.tool_usage.labels(tool_name=tool_name).inc()
        
    def track_error(self):
        self.system_stats["errors"] += 1
        
    def update_system_metrics(self):
        """Update system metrics"""
        process = psutil.Process()
        self.memory_usage.set(process.memory_info().rss)
        self.cpu_usage.set(process.cpu_percent())
        self.system_stats["uptime"] = time.time() - self.start_time
        
    def get_stats(self):
        self.update_system_metrics()
        return self.system_stats.copy()

# ===================== MARAN AI AGENT =====================
class MaranAgent:
    def __init__(self):
        self.tokenizer = MaranBPETokenizer(vocab_size=CONFIG["model"]["vocab_size"])
        self.model = None
        self.memory = None
        self.reasoner = None
        self.tools = {}
        self.monitor = MaranMonitor()
        self.is_initialized = False
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        # Initialize model
        self._init_model()
        
        # Initialize memory
        self._init_memory()
        
        # Initialize reasoning
        self._init_reasoning()
        
        # Initialize tools
        self._init_tools()
        
        self.is_initialized = True
        logger.info("🧠 Maran Agent initialized successfully!")

    def _init_tokenizer(self):
        """Initialize or load tokenizer"""
        tokenizer_path = "models/tokenizer.json"
        if os.path.exists(tokenizer_path):
            logger.info("Loading existing tokenizer...")
            self.tokenizer.load(tokenizer_path)
        else:
            logger.info("Training new tokenizer...")
            # Use sample texts if datasets not available
            sample_texts = [
                "Hello world, this is a test.",
                "Python programming language is powerful.",
                "Artificial intelligence and machine learning.",
                "Natural language processing with transformers.",
                "Generate code from natural language descriptions."
            ] * 100  # Repeat for more variety
            
            if 'datasets' in optional_imports:
                try:
                    # Load datasets for training
                    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
                    code_stream = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
                    wiki_texts = list(itertools.islice((item['text'] for item in wiki_stream), 10000))
                    code_texts = list(itertools.islice((item['whole_func_string'] for item in code_stream), 10000))
                    sample_texts = wiki_texts + code_texts
                except Exception as e:
                    logger.warning(f"Failed to load datasets, using sample texts: {e}")
            
            self.tokenizer.build_vocab(sample_texts)
            self.tokenizer.save(tokenizer_path)

    def _init_model(self):
        """Initialize or load model"""
        model_path = "models/maran_model.pth"
        
        self.model = MaranGPTModel(
            tokenizer=self.tokenizer,
            d_model=CONFIG["model"]["d_model"],
            nhead=CONFIG["model"]["nhead"],
            num_layers=CONFIG["model"]["num_layers"],
            max_len=CONFIG["model"]["max_len"],
            dropout=CONFIG["model"]["dropout"]
        )
        
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model, using random weights: {e}")
        else:
            logger.info("Using randomly initialized model")

    def _init_memory(self):
        """Initialize memory system"""
        self.memory = MaranMemory(self.model, self.tokenizer)

    def _init_reasoning(self):
        """Initialize reasoning system"""
        self.reasoner = TreeOfThoughtReasoner(self.model)

    def _init_tools(self):
        """Initialize all available tools"""
        self.tools = {
            "code_executor": CodeExecutorTool(),
            "shell": ShellTool(),
            "web_request": WebRequestTool()
        }
        
        # Add hardware tools if available
        if 'gpio' in optional_imports:
            try:
                self.tools["led"] = LEDTool(CONFIG["hardware"]["led_pin"])
                self.tools["servo"] = ServoTool(CONFIG["hardware"]["servo_pin"])
                self.tools["sensor"] = SensorTool(CONFIG["hardware"]["sensor_pin"])
                logger.info("Hardware tools initialized")
            except Exception as e:
                logger.warning(f"Hardware tools not available: {e}")

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
            
            thoughts = self.reasoner.generate_plan(
                query, 
                stream_callback=lambda data: asyncio.create_task(stream_callback(data)) if stream_callback else None
            )
            
            # Extract relevant memories
            if stream_callback:
                await stream_callback({"type": "status", "data": "🔍 Searching memory..."})
            
            relevant_memories = self.memory.query(query, limit=3)
            memory_context = "\n".join([m[3] for m in relevant_memories])  # content field
            
            # Generate response
            if stream_callback:
                await stream_callback({"type": "status", "data": "💭 Generating response..."})
            
            # Build context from thoughts and memories
            context = f"Query: {query}\n\nRelevant memories:\n{memory_context}\n\nReasoning:\n"
            context += "\n".join([f"- {t.content}" for t in thoughts])
            context += f"\n\nProvide a helpful response to the user's query:"
            
            # Generate response with streaming
            response = ""
            if stream_callback:
                def response_stream(partial_text, step, total_steps):
                    asyncio.create_task(stream_callback({
                        "type": "generation", 
                        "data": {
                            "text": partial_text,
                            "progress": step / total_steps
                        }
                    }))
                
                response = self.model.generate(
                    context, 
                    max_length=150, 
                    temperature=0.7,
                    stream_callback=response_stream
                )
            else:
                response = self.model.generate(context, max_length=150, temperature=0.7)
            
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
                "response": f"I encountered an error while processing your query: {str(e)}",
                "thoughts": [],
                "tool_results": {},
                "relevant_memories": 0,
                "response_time": time.time() - start_time,
                "error": str(e)
            }

    async def _analyze_for_tools(self, query: str, response: str, stream_callback=None):
        """Analyze if any tools should be used"""
        tool_results = {}
        
        # Simple heuristics for tool usage
        query_lower = query.lower()
        
        # Code execution
        if any(keyword in query_lower for keyword in ["code", "python", "execute", "run", "script"]):
            if stream_callback:
                await stream_callback({"type": "status", "data": "🔧 Executing code..."})
            
            # Extract code from response (simple heuristic)
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                result = self.tools["code_executor"].execute(code)
                tool_results["code_execution"] = result
                self.monitor.track_tool_usage("code_executor")
        
        # Hardware control
        if "led" in query_lower and "led" in self.tools:
            if stream_callback:
                await stream_callback({"type": "status", "data": "💡 Controlling LED..."})
            
            state = "on" if "on" in query_lower else "off"
            result = self.tools["led"].execute(state)
            tool_results["led"] = result
            self.monitor.track_tool_usage("led")
        
        # Web requests
        if any(keyword in query_lower for keyword in ["http", "request", "api", "url"]):
            if stream_callback:
                await stream_callback({"type": "status", "data": "🌐 Making web request..."})
            
            # Simple URL extraction
            url_match = re.search(r'https?://[^\s]+', query)
            if url_match:
                url = url_match.group(0)
                result = self.tools["web_request"].execute("GET", url)
                tool_results["web_request"] = result
                self.monitor.track_tool_usage("web_request")
        
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
app = FastAPI(title="Maran AI Agent", description="Sophisticated AI Agent with Web Interface")
agent = MaranAgent()

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Maran AI Agent</title>
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
        
        .thoughts-panel {
            max-height: 400px;
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
        }
        
        .tool-active { background: #28a745; }
        .tool-idle { background: #6c757d; }
        
        .memory-item {
            background: rgba(0, 0, 0, 0.2);
            margin: 5px 0;
            padding: 8px;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .memory-type {
            color: #ffd700;
            font-weight: bold;
            font-size: 12px;
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
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Header -->
        <div class="header">
            <h1>🧠 Maran AI Agent</h1>
            <p>Sophisticated Autonomous AI with Real-time Monitoring</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar"></div>
            </div>
        </div>
        
        <!-- Chat Panel -->
        <div class="panel chat-panel">
            <h3>💬 Conversation</h3>
            <div class="chat-messages" id="chatMessages"></div>
            <div class="input-area">
                <input type="text" id="queryInput" placeholder="Ask me anything..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <!-- Thoughts Panel -->
        <div class="panel thoughts-panel">
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
        
        <!-- Tools Panel -->
        <div class="panel">
            <h3>🛠️ Tools Status</h3>
            <div id="toolsContainer"></div>
        </div>
        
        <!-- Memory Panel -->
        <div class="panel">
            <h3>🧠 Recent Memories</h3>
            <div id="memoryContainer"></div>
        </div>
    </div>

    <script>
        class MaranWebInterface {
            constructor() {
                this.ws = null;
                this.isConnected = false;
                this.currentThoughts = [];
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
                    this.addMessage('system', '🔌 Connected to Maran AI Agent', 'status');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    this.isConnected = false;
                    this.addMessage('system', '🔌 Disconnected from agent', 'error');
                    // Attempt to reconnect after 3 seconds
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
                
                let resultsHtml = '<div style="margin-top: 10px;"><strong>🔧 Tool Results:</strong></div>';
                
                for (const [toolName, result] of Object.entries(toolResults)) {
                    const status = result.success ? '✅' : '❌';
                    resultsHtml += `
                        <div style="background: rgba(0,0,0,0.2); margin: 5px 0; padding: 8px; border-radius: 4px;">
                            <strong>${status} ${toolName}:</strong><br>
                            ${result.success ? 
                                (result.output || result.stdout || 'Success') : 
                                (result.error || 'Failed')
                            }
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
                // Update metric cards
                document.getElementById('totalRequests').textContent = data.monitor_stats.total_requests;
                document.getElementById('totalGenerations').textContent = data.monitor_stats.total_generations;
                document.getElementById('uptime').textContent = `${Math.floor(data.monitor_stats.uptime)}s`;
                document.getElementById('memoryCount').textContent = data.memory_stats.total_memories;
                
                // Update system info
                const systemInfo = document.getElementById('systemInfo');
                systemInfo.innerHTML = `
                    <div style="font-size: 12px; margin-top: 10px;">
                        <div>CPU: ${data.system_info.cpu_count} cores</div>
                        <div>Memory: ${(data.system_info.memory_available / 1024 / 1024 / 1024).toFixed(1)}GB available</div>
                        <div>GPU: ${data.system_info.gpu_available ? `${data.system_info.gpu_count} available` : 'Not available'}</div>
                        <div>Model: ${data.model_info.model_parameters.toLocaleString()} parameters</div>
                    </div>
                `;
                
                // Update tools status
                this.updateToolsStatus(data.tool_stats);
                
                // Update memory display
                this.updateMemoryDisplay(data.memory_stats);
            }
            
            updateToolsStatus(toolStats) {
                const toolsContainer = document.getElementById('toolsContainer');
                toolsContainer.innerHTML = '';
                
                for (const [toolName, stats] of Object.entries(toolStats)) {
                    const isActive = stats.last_used && 
                        (Date.now() - new Date(stats.last_used).getTime()) < 60000; // Active if used in last minute
                    
                    const toolDiv = document.createElement('div');
                    toolDiv.className = 'tool-item';
                    toolDiv.innerHTML = `
                        <div>
                            <div style="font-weight: bold;">${toolName}</div>
                            <div style="font-size: 12px; opacity: 0.8;">Used: ${stats.usage_count} times</div>
                        </div>
                        <div class="tool-status ${isActive ? 'tool-active' : 'tool-idle'}">
                            ${isActive ? 'Active' : 'Idle'}
                        </div>
                    `;
                    
                    toolsContainer.appendChild(toolDiv);
                }
            }
            
            updateMemoryDisplay(memoryStats) {
                const memoryContainer = document.getElementById('memoryContainer');
                // For now, just show stats. In a full implementation, 
                // we could fetch recent memories via API
                memoryContainer.innerHTML = `
                    <div class="memory-item">
                        <div class="memory-type">Total Memories</div>
                        <div>${memoryStats.total_memories}</div>
                    </div>
                    <div class="memory-item">
                        <div class="memory-type">Average Importance</div>
                        <div>${memoryStats.average_importance.toFixed(2)}</div>
                    </div>
                    <div class="memory-item">
                        <div class="memory-type">Vector Index Size</div>
                        <div>${memoryStats.vector_index_size}</div>
                    </div>
                `;
            }
        }
        
        // Initialize the interface when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new MaranWebInterface();
        });
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard"""
    return HTMLResponse(content=HTML_TEMPLATE)

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
        active_connections.remove(websocket)

async def broadcast_system_status():
    """Broadcast system status to all connected clients"""
    if not active_connections:
        return
    
    status = agent.get_system_status()
    message = json.dumps({
        "type": "system_status",
        "data": status
    })
    
    # Remove disconnected connections
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            disconnected.append(connection)
    
    for conn in disconnected:
        active_connections.remove(conn)

# Background task to broadcast status updates
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    async def status_broadcaster():
        while True:
            await asyncio.sleep(5)  # Broadcast every 5 seconds
            await broadcast_system_status()
    
    # Start the broadcaster
    asyncio.create_task(status_broadcaster())
    
    # Start Prometheus metrics server
    try:
        start_http_server(CONFIG["web"]["metrics_port"], registry=agent.monitor.registry)
        logger.info(f"📊 Metrics server started on port {CONFIG['web']['metrics_port']}")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")

# ===================== MAIN EXECUTION =====================
def main():
    """Main entry point"""
    print("🧠 Starting Maran AI Agent Web Interface...")
    print("=" * 60)
    print(f"🌐 Web Interface: http://localhost:{CONFIG['web']['port']}")
    print(f"📊 Metrics: http://localhost:{CONFIG['web']['metrics_port']}/metrics")
    print("🔧 Features:")
    print("   • Real-time AI conversation with thought streaming")
    print("   • Multi-panel sophisticated dashboard")
    print("   • Hardware & software tool integrations")
    print("   • Advanced memory and reasoning systems")
    print("   • Comprehensive monitoring and analytics")
    print("=" * 60)
    
    # Configure uvicorn for production-ready deployment
    uvicorn.run(
        app,
        host=CONFIG["web"]["host"],
        port=CONFIG["web"]["port"],
        log_level="info",
        reload=False,  # Disable reload for production
        access_log=True
    )

if __name__ == "__main__":
    main()