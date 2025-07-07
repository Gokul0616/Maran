# enhanced_deepmodel.py
"""
Enhanced version of deepmodel.py with integrated advanced monitoring
"""

import logging
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from datasets import load_dataset
import itertools
from tokenizers.BPETokenizer import CustomBPETokenizer
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from model_versioning import ModelVersioner
from rate_limiter import RateLimiter

# Import our advanced monitoring system
from advanced_monitoring import (
    AdvancedMonitoringSystem, 
    monitor_operation,
    StructuredLogger
)

logger = StructuredLogger("enhanced_deepmodel")

class MonitoredOptimizedAttention(nn.Module):
    """Enhanced attention mechanism with built-in monitoring"""
    
    def __init__(self, d_model=768, n_heads=12, monitoring_system=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.monitoring_system = monitoring_system
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Rotary positional embeddings
        self.register_buffer("freqs", self._precompute_freqs())
        
    def _precompute_freqs(self, base=10000):
        dim = self.head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return freqs

    def _apply_rotary(self, x, seq_dim=1):
        B, T, H, D = x.shape
        x = x.view(B, T, H, D//2, 2)
        x_rot = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        x_rot = x_rot.view(*x.shape[:-1], D)
        return x_rot

    @monitor_operation("attention_forward", "model")
    def forward(self, x, sparse_mask=None):
        B, T, _ = x.shape
        
        # Track attention computation metrics
        if self.monitoring_system:
            trace_id = self.monitoring_system.performance_profiler.start_operation(
                "attention_computation",
                {"batch_size": B, "sequence_length": T, "model_dim": self.d_model}
            )
        
        try:
            # Project queries, keys, values
            q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings
            q = self._apply_rotary(q)
            k = self._apply_rotary(k)
            
            # Efficient attention using PyTorch's built-in optimized kernel
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sparse_mask,
                dropout_p=0.1,
                is_causal=True
            )
            
            # Combine heads and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
            result = self.out_proj(attn_output)
            
            if self.monitoring_system:
                self.monitoring_system.performance_profiler.end_operation(
                    trace_id, "success", 
                    {"output_shape": str(result.shape)}
                )
            
            return result
            
        except Exception as e:
            if self.monitoring_system:
                self.monitoring_system.error_tracker.track_error(
                    error_type=type(e).__name__,
                    severity="error",
                    component="attention",
                    message=str(e),
                    trace_id=trace_id if 'trace_id' in locals() else None
                )
                if 'trace_id' in locals():
                    self.monitoring_system.performance_profiler.end_operation(
                        trace_id, "error", {"error": str(e)}
                    )
            raise

class MonitoredGPTModel(nn.Module):
    """Enhanced GPT model with comprehensive monitoring"""
    
    def __init__(self, tokenizer, d_model=768, nhead=12, num_layers=12, 
                 max_len=1024, dropout=0.1, monitoring_system=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.monitoring_system = monitoring_system or AdvancedMonitoringSystem()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model architecture
        self.token_embed = nn.Embedding(tokenizer.vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            MonitoredGPTDecoderBlock(d_model, nhead, dropout, self.monitoring_system) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tokenizer.vocab_size)
        
        # Monitoring components
        self.versioner = ModelVersioner()
        self.rate_limiter = RateLimiter(rpm=1000)
        self.max_len = max_len
        self.d_model = d_model
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0.0
        }
        
        logger.log_event("INFO", "model_init", "Enhanced GPT model initialized with monitoring")

    @monitor_operation("forward_pass", "model")
    def forward(self, x):
        B, T = x.shape
        
        # Track token usage for cost calculation
        if self.monitoring_system:
            self.monitoring_system.cost_tracker.track_inference_cost(
                model_name="gpt_custom",
                tokens_used=B * T
            )
        
        token_embeddings = self.token_embed(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        position_embeddings = self.pos_embed(position_ids)
        x = token_embeddings + position_embeddings

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).expand(B, -1, -1)
        attn_mask = attn_mask == 0

        for i, block in enumerate(self.blocks):
            try:
                x = block(x, attn_mask=attn_mask)
            except Exception as e:
                self.monitoring_system.error_tracker.track_error(
                    error_type=type(e).__name__,
                    severity="error",
                    component=f"transformer_block_{i}",
                    message=f"Error in transformer block {i}: {str(e)}"
                )
                raise

        x = self.ln_f(x)
        return self.head(x)

    @monitor_operation("model_generation", "model")
    def generate(self, prompt: str, **kwargs) -> str:
        """Enhanced generation with comprehensive monitoring"""
        start_time = time.time()
        
        # Rate limiting
        try:
            self.rate_limiter()
        except Exception as e:
            self.monitoring_system.error_tracker.track_error(
                error_type="RateLimitExceeded",
                severity="warning",
                component="rate_limiter",
                message="Rate limit exceeded for generation request"
            )
            raise
        
        # Track generation metrics
        trace_id = self.monitoring_system.performance_profiler.start_operation(
            "text_generation",
            {
                "prompt_length": len(prompt),
                "max_length": kwargs.get('max_length', 100),
                "temperature": kwargs.get('temperature', 1.0)
            }
        )
        
        try:
            # Prepare inputs
            inputs = self.tokenizer.encode(prompt)
            inputs = inputs[-self.max_len:]  # Truncate to max length
            inputs = torch.tensor([inputs], device=self.device)
            
            generated = []
            max_length = kwargs.get('max_length', 100)
            
            for step in range(max_length):
                with torch.no_grad():
                    outputs = self(inputs)
                    next_token = self._sample(
                        outputs[0, -1], 
                        kwargs.get('temperature', 1.0),
                        kwargs.get('top_k', 40)
                    )
                    generated.append(next_token.item())
                    inputs = torch.cat([inputs, torch.tensor([[next_token]], device=self.device)], dim=1)
                    
                    # Early stopping for end token
                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            result = self.tokenizer.decode(generated)
            
            # Update statistics
            generation_time = time.time() - start_time
            self.generation_stats["total_generations"] += 1
            self.generation_stats["total_tokens_generated"] += len(generated)
            self.generation_stats["total_inference_time"] += generation_time
            
            # Track costs
            self.monitoring_system.cost_tracker.track_inference_cost(
                model_name="gpt_custom",
                tokens_used=len(generated),
                trace_id=trace_id
            )
            
            # Log successful generation
            self.monitoring_system.performance_profiler.end_operation(
                trace_id, "success",
                {
                    "tokens_generated": len(generated),
                    "generation_time": generation_time,
                    "tokens_per_second": len(generated) / generation_time if generation_time > 0 else 0
                }
            )
            
            logger.log_event(
                "INFO", "generation_complete",
                f"Successfully generated {len(generated)} tokens",
                context={
                    "prompt_length": len(prompt),
                    "tokens_generated": len(generated),
                    "generation_time": generation_time,
                    "trace_id": trace_id
                }
            )
            
            return result
            
        except Exception as e:
            self.monitoring_system.error_tracker.track_error(
                error_type=type(e).__name__,
                severity="error",
                component="text_generation",
                message=f"Generation failed: {str(e)}",
                trace_id=trace_id
            )
            self.monitoring_system.performance_profiler.end_operation(
                trace_id, "error", {"error": str(e)}
            )
            raise
    
    def _sample(self, logits, temperature, top_k):
        """Sample next token with monitoring"""
        logits = logits / temperature
        if top_k > 0:
            topk = torch.topk(logits, top_k)
            logits[logits < topk.values[-1]] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)
    
    def get_model_stats(self) -> dict:
        """Get comprehensive model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        stats = {
            "model_architecture": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assume float32
                "layers": len(self.blocks),
                "hidden_size": self.d_model,
                "max_sequence_length": self.max_len
            },
            "generation_statistics": self.generation_stats.copy(),
            "device": str(self.device),
            "memory_usage": {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
                "cached_mb": torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
            }
        }
        
        # Add performance metrics if available
        if hasattr(self.monitoring_system, 'performance_profiler'):
            performance_summary = self.monitoring_system.performance_profiler.get_performance_summary(
                operation="text_generation", last_n_minutes=60
            )
            stats["performance_last_hour"] = performance_summary
        
        return stats

class MonitoredGPTDecoderBlock(nn.Module):
    """Enhanced decoder block with monitoring"""
    
    def __init__(self, d_model, nhead, dropout=0.1, monitoring_system=None):
        super().__init__()
        self.monitoring_system = monitoring_system
        self.attn = MonitoredOptimizedAttention(d_model, nhead, monitoring_system)
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
        # Sparse attention with monitoring
        attn_out = self.attn(self.attn_norm(x), attn_mask)
        x = x + self.dropout(attn_out)
        
        # Feedforward with monitoring
        ff_out = self.ff(self.ff_norm(x))
        return x + self.dropout(ff_out)

# Enhanced training function with monitoring
@monitor_operation("model_training", "training")
def enhanced_train_model(model, dataset, tokenizer, epochs=5, batch_size=8, 
                        lr=1e-4, grad_accum_steps=4, eval_interval=1, 
                        monitoring_system=None):
    """Enhanced training loop with comprehensive monitoring"""
    
    if monitoring_system is None:
        monitoring_system = AdvancedMonitoringSystem()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id.get("<PAD>", 0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    scaler = GradScaler()

    # Track compute costs
    training_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_trace_id = monitoring_system.performance_profiler.start_operation(
            f"training_epoch_{epoch}",
            {"epoch": epoch, "total_epochs": epochs, "batch_size": batch_size}
        )
        
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        
        try:
            optimizer.zero_grad()

            for i, batch in enumerate(loader, 1):
                batch_trace_id = monitoring_system.performance_profiler.start_operation(
                    "training_batch",
                    {"epoch": epoch, "batch": i, "batch_size": batch.size(0)}
                )
                
                try:
                    batch = batch.to(device)
                    with autocast():
                        outputs = model(batch)
                        logits = outputs[:, :-1, :].reshape(-1, outputs.size(-1))
                        labels = batch[:, 1:].reshape(-1)
                        loss = criterion(logits, labels)

                    scaler.scale(loss).backward()

                    if i % grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    total_loss += loss.item()
                    
                    monitoring_system.performance_profiler.end_operation(
                        batch_trace_id, "success",
                        {"loss": loss.item(), "batch_tokens": batch.numel()}
                    )

                    if i % 50 == 0:
                        logger.log_event(
                            "INFO", "training_progress",
                            f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}",
                            context={
                                "epoch": epoch,
                                "batch": i,
                                "total_batches": len(loader),
                                "loss": loss.item(),
                                "learning_rate": lr
                            }
                        )
                        
                except Exception as e:
                    monitoring_system.error_tracker.track_error(
                        error_type=type(e).__name__,
                        severity="error",
                        component="training_batch",
                        message=f"Batch training failed: {str(e)}",
                        trace_id=batch_trace_id
                    )
                    monitoring_system.performance_profiler.end_operation(
                        batch_trace_id, "error", {"error": str(e)}
                    )
                    raise

            avg_loss = total_loss / len(loader)
            epoch_time = time.time() - epoch_start
            
            # Track compute costs for this epoch
            monitoring_system.cost_tracker.track_compute_cost(
                "gpu", epoch_time / 3600,  # Convert to hours
                trace_id=epoch_trace_id
            )
            
            monitoring_system.performance_profiler.end_operation(
                epoch_trace_id, "success",
                {
                    "avg_loss": avg_loss,
                    "epoch_time": epoch_time,
                    "batches_processed": len(loader)
                }
            )
            
            logger.log_event(
                "INFO", "epoch_complete",
                f"Epoch {epoch} completed. Avg Training Loss: {avg_loss:.4f}",
                context={
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "epoch_time": epoch_time,
                    "total_batches": len(loader)
                }
            )

            # Evaluation
            if epoch % eval_interval == 0:
                eval_trace_id = monitoring_system.performance_profiler.start_operation(
                    f"evaluation_epoch_{epoch}",
                    {"epoch": epoch}
                )
                
                try:
                    eval_loss, perplexity = enhanced_evaluate_model(
                        model, dataset, tokenizer, monitoring_system
                    )
                    
                    monitoring_system.performance_profiler.end_operation(
                        eval_trace_id, "success",
                        {"eval_loss": eval_loss, "perplexity": perplexity}
                    )
                    
                    logger.log_event(
                        "INFO", "evaluation_complete",
                        f"Epoch {epoch} | Evaluation Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}",
                        context={
                            "epoch": epoch,
                            "eval_loss": eval_loss,
                            "perplexity": perplexity
                        }
                    )
                    
                except Exception as e:
                    monitoring_system.error_tracker.track_error(
                        error_type=type(e).__name__,
                        severity="error",
                        component="evaluation",
                        message=f"Evaluation failed: {str(e)}",
                        trace_id=eval_trace_id
                    )
                    monitoring_system.performance_profiler.end_operation(
                        eval_trace_id, "error", {"error": str(e)}
                    )

        except Exception as e:
            monitoring_system.error_tracker.track_error(
                error_type=type(e).__name__,
                severity="critical",
                component="training_epoch",
                message=f"Epoch {epoch} failed: {str(e)}",
                trace_id=epoch_trace_id
            )
            monitoring_system.performance_profiler.end_operation(
                epoch_trace_id, "error", {"error": str(e)}
            )
            raise
    
    # Track total training time and cost
    total_training_time = time.time() - training_start
    monitoring_system.cost_tracker.track_compute_cost(
        "gpu", total_training_time / 3600,
        trace_id="full_training"
    )
    
    logger.log_event(
        "INFO", "training_complete",
        f"Training completed successfully in {total_training_time:.2f} seconds",
        context={
            "total_epochs": epochs,
            "total_training_time": total_training_time,
            "final_learning_rate": lr
        }
    )

@monitor_operation("model_evaluation", "evaluation")
def enhanced_evaluate_model(model, dataset, tokenizer, monitoring_system, batch_size=8):
    """Enhanced evaluation with monitoring"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.get("<PAD>", 0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            try:
                batch = batch.to(device)
                outputs = model(batch)
                logits = outputs[:, :-1, :].reshape(-1, outputs.size(-1))
                labels = batch[:, 1:].reshape(-1)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                total_batches += 1
                
            except Exception as e:
                monitoring_system.error_tracker.track_error(
                    error_type=type(e).__name__,
                    severity="error",
                    component="evaluation_batch",
                    message=f"Evaluation batch {i} failed: {str(e)}"
                )
                continue
    
    if total_batches == 0:
        raise ValueError("No batches were successfully evaluated")
    
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

# Example integration function
def create_monitored_model(tokenizer, monitoring_system=None):
    """Create a monitored GPT model with all enhancements"""
    if monitoring_system is None:
        monitoring_system = AdvancedMonitoringSystem()
        monitoring_system.start()
    
    model = MonitoredGPTModel(
        tokenizer=tokenizer,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        monitoring_system=monitoring_system
    )
    
    return model, monitoring_system