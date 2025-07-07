# advanced_monitoring.py
import logging
import time
import psutil
import threading
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import torch
import numpy as np
from prometheus_client import Gauge, Counter, Histogram, Info, start_http_server
import sqlite3
from pathlib import Path

# Create a single Prometheus registry to avoid conflicts
from prometheus_client import CollectorRegistry
PROMETHEUS_REGISTRY = CollectorRegistry()

# ===================== ENHANCED LOGGING SYSTEM =====================

class StructuredLogger:
    """Enhanced structured logging with different levels and contexts"""
    
    def __init__(self, name: str, log_file: str = "logs/maran.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # File handler with JSON formatting
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_event(self, level: str, event_type: str, message: str, 
                  context: Dict[str, Any] = None, trace_id: str = None):
        """Log structured events with context"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "message": message,
            "trace_id": trace_id or str(uuid.uuid4()),
            "context": context or {}
        }
        
        log_message = json.dumps(log_data)
        
        if level.upper() == "DEBUG":
            self.logger.debug(log_message)
        elif level.upper() == "INFO":
            self.logger.info(log_message)
        elif level.upper() == "WARNING":
            self.logger.warning(log_message)
        elif level.upper() == "ERROR":
            self.logger.error(log_message)
        elif level.upper() == "CRITICAL":
            self.logger.critical(log_message)

# ===================== PERFORMANCE PROFILER =====================

@dataclass
class PerformanceMetrics:
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    gpu_utilization: float
    trace_id: str
    context: Dict[str, Any]

class PerformanceProfiler:
    """Advanced performance profiling and tracking"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.active_operations = {}
        self.logger = StructuredLogger("performance_profiler")
        
        # Prometheus metrics
        self.operation_duration = Histogram(
            'operation_duration_seconds',
            'Duration of operations',
            ['operation_type', 'status'],
            registry=PROMETHEUS_REGISTRY
        )
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['operation_type'],
            registry=PROMETHEUS_REGISTRY
        )
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['operation_type'],
            registry=PROMETHEUS_REGISTRY
        )
        self.gpu_usage = Gauge(
            'gpu_usage_percent', 
            'GPU usage percentage',
            ['operation_type'],
            registry=PROMETHEUS_REGISTRY
        )
    
    def start_operation(self, operation: str, context: Dict[str, Any] = None) -> str:
        """Start tracking an operation"""
        trace_id = str(uuid.uuid4())
        
        self.active_operations[trace_id] = {
            "operation": operation,
            "start_time": time.time(),
            "memory_before": psutil.Process().memory_info().rss,
            "context": context or {}
        }
        
        self.logger.log_event(
            "DEBUG", "operation_start", f"Started operation: {operation}",
            context={"trace_id": trace_id, **context} if context else {"trace_id": trace_id}
        )
        
        return trace_id
    
    def end_operation(self, trace_id: str, status: str = "success", 
                     result_context: Dict[str, Any] = None):
        """End tracking an operation"""
        if trace_id not in self.active_operations:
            return
        
        op_data = self.active_operations.pop(trace_id)
        end_time = time.time()
        duration = end_time - op_data["start_time"]
        memory_after = psutil.Process().memory_info().rss
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        gpu_util = self._get_gpu_utilization()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation=op_data["operation"],
            start_time=op_data["start_time"],
            end_time=end_time,
            duration=duration,
            memory_before=op_data["memory_before"],
            memory_after=memory_after,
            memory_peak=max(op_data["memory_before"], memory_after),
            cpu_percent=cpu_percent,
            gpu_utilization=gpu_util,
            trace_id=trace_id,
            context={**op_data["context"], **(result_context or {})}
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Update Prometheus metrics
        self.operation_duration.labels(
            operation_type=op_data["operation"], 
            status=status
        ).observe(duration)
        
        self.memory_usage.labels(
            operation_type=op_data["operation"]
        ).set(memory_after)
        
        self.cpu_usage.labels(
            operation_type=op_data["operation"]
        ).set(cpu_percent)
        
        self.gpu_usage.labels(
            operation_type=op_data["operation"]
        ).set(gpu_util)
        
        # Log completion
        self.logger.log_event(
            "INFO", "operation_complete", 
            f"Completed operation: {op_data['operation']}",
            context={
                "trace_id": trace_id,
                "duration": duration,
                "status": status,
                "memory_used": memory_after - op_data["memory_before"],
                **metrics.context
            }
        )
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization if available"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return 0.0
        except:
            return 0.0
    
    def get_performance_summary(self, operation: str = None, 
                              last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for operations"""
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        relevant_metrics = [
            m for m in self.metrics_history 
            if m.end_time > cutoff_time and (operation is None or m.operation == operation)
        ]
        
        if not relevant_metrics:
            return {"message": "No metrics found for the specified criteria"}
        
        durations = [m.duration for m in relevant_metrics]
        memory_usage = [m.memory_after - m.memory_before for m in relevant_metrics]
        
        return {
            "operation": operation or "all",
            "time_window_minutes": last_n_minutes,
            "total_operations": len(relevant_metrics),
            "avg_duration": np.mean(durations),
            "max_duration": np.max(durations),
            "min_duration": np.min(durations),
            "p95_duration": np.percentile(durations, 95),
            "avg_memory_usage": np.mean(memory_usage),
            "max_memory_usage": np.max(memory_usage),
            "avg_cpu_percent": np.mean([m.cpu_percent for m in relevant_metrics]),
            "avg_gpu_utilization": np.mean([m.gpu_utilization for m in relevant_metrics])
        }

# ===================== SYSTEM HEALTH MONITOR =====================

class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.logger = StructuredLogger("system_health")
        self.is_running = False
        self.monitor_thread = None
        
        # Health thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "gpu_memory_percent": 90.0,
            "response_time_ms": 5000.0
        }
        
        # Prometheus metrics
        self.system_cpu = Gauge('system_cpu_percent', 'System CPU usage', registry=PROMETHEUS_REGISTRY)
        self.system_memory = Gauge('system_memory_percent', 'System memory usage', registry=PROMETHEUS_REGISTRY)
        self.system_disk = Gauge('system_disk_percent', 'System disk usage', registry=PROMETHEUS_REGISTRY)
        self.system_gpu_memory = Gauge('system_gpu_memory_percent', 'GPU memory usage', registry=PROMETHEUS_REGISTRY)
        self.health_score = Gauge('system_health_score', 'Overall system health score', registry=PROMETHEUS_REGISTRY)
        
    def start_monitoring(self):
        """Start the health monitoring thread"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.log_event("INFO", "monitoring_start", "System health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.log_event("INFO", "monitoring_stop", "System health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                health_data = self._collect_health_metrics()
                self._check_thresholds(health_data)
                self.health_history.append(health_data)
                
                # Update Prometheus metrics
                self.system_cpu.set(health_data["cpu_percent"])
                self.system_memory.set(health_data["memory_percent"])
                self.system_disk.set(health_data["disk_percent"])
                self.system_gpu_memory.set(health_data["gpu_memory_percent"])
                self.health_score.set(health_data["health_score"])
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.log_event(
                    "ERROR", "monitoring_error", 
                    f"Error in health monitoring: {str(e)}"
                )
                time.sleep(self.check_interval)
    
    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive health metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_memory_percent = 0.0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.max_memory_allocated()
            if gpu_memory_total > 0:
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        
        # Calculate health score (0-100)
        health_score = self._calculate_health_score(
            cpu_percent, memory.percent, disk.percent, gpu_memory_percent
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "gpu_memory_percent": gpu_memory_percent,
            "health_score": health_score,
            "processes_count": len(psutil.pids()),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    
    def _calculate_health_score(self, cpu: float, memory: float, 
                               disk: float, gpu_memory: float) -> float:
        """Calculate overall health score"""
        # Weight different metrics
        weights = {"cpu": 0.3, "memory": 0.3, "disk": 0.2, "gpu": 0.2}
        
        # Convert usage percentages to health scores (100 - usage)
        scores = {
            "cpu": max(0, 100 - cpu),
            "memory": max(0, 100 - memory),
            "disk": max(0, 100 - disk),
            "gpu": max(0, 100 - gpu_memory)
        }
        
        # Calculate weighted average
        health_score = sum(scores[metric] * weights[metric] for metric in scores)
        return round(health_score, 2)
    
    def _check_thresholds(self, health_data: Dict[str, Any]):
        """Check if any metrics exceed thresholds"""
        for metric, threshold in self.thresholds.items():
            if metric in health_data and health_data[metric] > threshold:
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "warning" if health_data[metric] < threshold * 1.2 else "critical",
                    "metric": metric,
                    "value": health_data[metric],
                    "threshold": threshold,
                    "message": f"{metric} is {health_data[metric]:.1f}%, exceeding threshold of {threshold}%"
                }
                
                self.alerts.append(alert)
                self.logger.log_event(
                    alert["severity"].upper(), "threshold_breach",
                    alert["message"], context=alert
                )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if not self.health_history:
            return {"status": "unknown", "message": "No health data available"}
        
        latest = self.health_history[-1]
        recent_alerts = [a for a in self.alerts if 
                        datetime.fromisoformat(a["timestamp"]) > 
                        datetime.utcnow() - timedelta(hours=1)]
        
        return {
            "status": "healthy" if latest["health_score"] > 70 else "degraded" if latest["health_score"] > 40 else "critical",
            "health_score": latest["health_score"],
            "latest_metrics": latest,
            "recent_alerts_count": len(recent_alerts),
            "recent_alerts": recent_alerts[-5:] if recent_alerts else []
        }

# ===================== ERROR TRACKING SYSTEM =====================

class ErrorTracker:
    """Advanced error tracking and analysis"""
    
    def __init__(self, db_path: str = "logs/errors.db"):
        self.db_path = db_path
        self.logger = StructuredLogger("error_tracker")
        self._init_database()
        
        # Prometheus metrics
        self.error_count = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'severity', 'component'],
            registry=PROMETHEUS_REGISTRY
        )
        self.error_rate = Gauge(
            'error_rate_per_minute',
            'Error rate per minute',
            ['error_type'],
            registry=PROMETHEUS_REGISTRY
        )
    
    def _init_database(self):
        """Initialize SQLite database for error storage"""
        Path("logs").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                error_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                stack_trace TEXT,
                context TEXT,
                trace_id TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def track_error(self, error_type: str, severity: str, component: str,
                   message: str, stack_trace: str = None, context: Dict[str, Any] = None,
                   trace_id: str = None):
        """Track a new error"""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "severity": severity,
            "component": component,
            "message": message,
            "stack_trace": stack_trace,
            "context": json.dumps(context) if context else None,
            "trace_id": trace_id or str(uuid.uuid4())
        }
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO errors (timestamp, error_type, severity, component, message, 
                              stack_trace, context, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            error_data["timestamp"], error_data["error_type"], error_data["severity"],
            error_data["component"], error_data["message"], error_data["stack_trace"],
            error_data["context"], error_data["trace_id"]
        ))
        conn.commit()
        conn.close()
        
        # Update Prometheus metrics
        self.error_count.labels(
            error_type=error_type,
            severity=severity,
            component=component
        ).inc()
        
        # Log the error
        self.logger.log_event(
            severity.upper(), "error_tracked",
            f"Error tracked: {message}",
            context=error_data
        )
    
    def get_error_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error analytics for the specified time period"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get error counts by type
        cursor.execute('''
            SELECT error_type, severity, COUNT(*) as count
            FROM errors 
            WHERE datetime(timestamp) > datetime(?)
            GROUP BY error_type, severity
            ORDER BY count DESC
        ''', (cutoff.isoformat(),))
        
        error_counts = cursor.fetchall()
        
        # Get error trends (hourly)
        cursor.execute('''
            SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour, COUNT(*) as count
            FROM errors
            WHERE datetime(timestamp) > datetime(?)
            GROUP BY hour
            ORDER BY hour
        ''', (cutoff.isoformat(),))
        
        hourly_trends = cursor.fetchall()
        
        conn.close()
        
        return {
            "time_period_hours": hours,
            "error_counts": [
                {"error_type": row[0], "severity": row[1], "count": row[2]}
                for row in error_counts
            ],
            "hourly_trends": [
                {"hour": row[0], "count": row[1]}
                for row in hourly_trends
            ],
            "total_errors": sum(row[2] for row in error_counts)
        }

# ===================== COST TRACKING SYSTEM =====================

class CostTracker:
    """Track costs for model usage and compute resources"""
    
    def __init__(self):
        self.cost_history = deque(maxlen=10000)
        self.logger = StructuredLogger("cost_tracker")
        
        # Cost rates (example values - adjust based on actual costs)
        self.cost_rates = {
            "gpt_inference": 0.0001,  # per token
            "gpu_hour": 0.50,         # per hour
            "cpu_hour": 0.05,         # per hour
            "storage_gb_month": 0.02, # per GB per month
            "bandwidth_gb": 0.01      # per GB
        }
        
        # Prometheus metrics
        self.total_cost = Gauge('total_cost_usd', 'Total accumulated cost in USD')
        self.hourly_cost = Gauge('hourly_cost_usd', 'Cost per hour', ['resource_type'])
        self.token_cost = Counter('token_cost_total', 'Total token processing cost')
    
    def track_inference_cost(self, model_name: str, tokens_used: int, trace_id: str = None):
        """Track cost for model inference"""
        cost = tokens_used * self.cost_rates["gpt_inference"]
        
        cost_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "cost_type": "inference",
            "resource": model_name,
            "quantity": tokens_used,
            "rate": self.cost_rates["gpt_inference"],
            "cost_usd": cost,
            "trace_id": trace_id
        }
        
        self.cost_history.append(cost_entry)
        self.token_cost.inc(cost)
        
        self.logger.log_event(
            "DEBUG", "cost_tracked",
            f"Inference cost tracked: ${cost:.4f} for {tokens_used} tokens",
            context=cost_entry
        )
    
    def track_compute_cost(self, resource_type: str, duration_hours: float, trace_id: str = None):
        """Track compute resource costs"""
        rate_key = f"{resource_type}_hour"
        if rate_key not in self.cost_rates:
            self.logger.log_event(
                "WARNING", "unknown_resource",
                f"Unknown resource type for cost tracking: {resource_type}"
            )
            return
        
        cost = duration_hours * self.cost_rates[rate_key]
        
        cost_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "cost_type": "compute",
            "resource": resource_type,
            "quantity": duration_hours,
            "rate": self.cost_rates[rate_key],
            "cost_usd": cost,
            "trace_id": trace_id
        }
        
        self.cost_history.append(cost_entry)
        self.hourly_cost.labels(resource_type=resource_type).set(cost / duration_hours if duration_hours > 0 else 0)
        
        self.logger.log_event(
            "DEBUG", "cost_tracked",
            f"Compute cost tracked: ${cost:.4f} for {duration_hours:.2f} hours of {resource_type}",
            context=cost_entry
        )
    
    def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        relevant_costs = [
            cost for cost in self.cost_history
            if datetime.fromisoformat(cost["timestamp"]) > cutoff_time
        ]
        
        if not relevant_costs:
            return {"message": "No cost data found for the specified period"}
        
        total_cost = sum(cost["cost_usd"] for cost in relevant_costs)
        
        # Group by resource type
        by_resource = defaultdict(lambda: {"count": 0, "total_cost": 0.0})
        for cost in relevant_costs:
            resource = cost["resource"]
            by_resource[resource]["count"] += 1
            by_resource[resource]["total_cost"] += cost["cost_usd"]
        
        return {
            "time_period_hours": hours,
            "total_cost_usd": round(total_cost, 4),
            "cost_by_resource": dict(by_resource),
            "total_operations": len(relevant_costs),
            "average_cost_per_operation": round(total_cost / len(relevant_costs), 6) if relevant_costs else 0
        }

# ===================== MAIN MONITORING SYSTEM =====================

class AdvancedMonitoringSystem:
    """Main monitoring system that orchestrates all monitoring components"""
    
    def __init__(self, metrics_port: int = 9090):
        self.metrics_port = metrics_port
        self.logger = StructuredLogger("monitoring_system")
        
        # Initialize all monitoring components
        self.performance_profiler = PerformanceProfiler()
        self.health_monitor = SystemHealthMonitor()
        self.error_tracker = ErrorTracker()
        self.cost_tracker = CostTracker()
        
        # System info
        self.system_info = Info('system_info', 'System information')
        self._set_system_info()
    
    def _set_system_info(self):
        """Set system information metrics"""
        import platform
        self.system_info.info({
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'memory_gb': str(round(psutil.virtual_memory().total / (1024**3), 2)),
            'cpu_cores': str(psutil.cpu_count()),
            'gpu_available': str(torch.cuda.is_available()),
            'gpu_count': str(torch.cuda.device_count()) if torch.cuda.is_available() else "0"
        })
    
    def start(self):
        """Start all monitoring components"""
        try:
            # Start Prometheus metrics server
            start_http_server(self.metrics_port)
            self.logger.log_event(
                "INFO", "metrics_server_start",
                f"Metrics server started on port {self.metrics_port}"
            )
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            self.logger.log_event(
                "INFO", "monitoring_system_start",
                "Advanced monitoring system started successfully"
            )
            
        except Exception as e:
            self.logger.log_event(
                "ERROR", "monitoring_start_error",
                f"Failed to start monitoring system: {str(e)}"
            )
            raise
    
    def stop(self):
        """Stop all monitoring components"""
        try:
            self.health_monitor.stop_monitoring()
            self.logger.log_event(
                "INFO", "monitoring_system_stop",
                "Advanced monitoring system stopped"
            )
        except Exception as e:
            self.logger.log_event(
                "ERROR", "monitoring_stop_error",
                f"Error stopping monitoring system: {str(e)}"
            )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": self.health_monitor.get_health_status(),
            "performance_summary": self.performance_profiler.get_performance_summary(),
            "error_analytics": self.error_tracker.get_error_analytics(),
            "cost_summary": self.cost_tracker.get_cost_summary(),
            "metrics_endpoint": f"http://localhost:{self.metrics_port}/metrics"
        }

# ===================== DECORATOR FOR EASY INTEGRATION =====================

def monitor_operation(operation_name: str, component: str = "default"):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitoring_system = getattr(wrapper, '_monitoring_system', None)
            if not monitoring_system:
                # Create a default monitoring system if none exists
                monitoring_system = AdvancedMonitoringSystem()
            
            trace_id = monitoring_system.performance_profiler.start_operation(
                operation_name, {"component": component, "function": func.__name__}
            )
            
            try:
                result = func(*args, **kwargs)
                monitoring_system.performance_profiler.end_operation(
                    trace_id, "success", {"result_type": type(result).__name__}
                )
                return result
                
            except Exception as e:
                monitoring_system.error_tracker.track_error(
                    error_type=type(e).__name__,
                    severity="error",
                    component=component,
                    message=str(e),
                    stack_trace=str(e.__traceback__),
                    trace_id=trace_id
                )
                monitoring_system.performance_profiler.end_operation(
                    trace_id, "error", {"error": str(e)}
                )
                raise
        
        return wrapper
    return decorator