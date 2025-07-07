# 🧠 Maran AI Agent - Advanced Monitoring System

## 📊 Overview

This document describes the comprehensive monitoring system implemented for the Maran AI Agent. The monitoring system provides real-time observability, performance tracking, error analysis, cost management, and system health monitoring.

## 🚀 Features

### ✅ **Comprehensive Monitoring**
- **Performance Profiling**: Track operation durations, memory usage, CPU/GPU utilization
- **System Health**: Real-time health scoring with configurable thresholds
- **Error Tracking**: Detailed error analytics with categorization and trends
- **Cost Management**: Track computational costs and resource usage
- **Real-time Dashboard**: Web-based monitoring interface with live updates

### ✅ **Advanced Logging**
- **Structured Logging**: JSON-formatted logs with context and trace IDs
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Distributed Tracing**: Track operations across components
- **Log Aggregation**: Centralized logging with search and filtering

### ✅ **Alerting System**
- **Configurable Thresholds**: Set custom alerts for various metrics
- **Multiple Severities**: Warning, critical, and informational alerts
- **Alert Channels**: Console, file, webhook, and email notifications
- **Alert History**: Track and analyze alert patterns

### ✅ **Performance Analytics**
- **Operation Metrics**: Duration, throughput, success rates
- **Resource Usage**: Memory, CPU, GPU utilization tracking
- **Trend Analysis**: Historical performance data and patterns
- **Bottleneck Identification**: Identify performance issues

## 📁 File Structure

```
/app/
├── advanced_monitoring.py          # Core monitoring system
├── monitoring_dashboard.py         # Web dashboard
├── enhanced_deepmodel.py          # Enhanced model with monitoring
├── monitoring_integration_example.py # Integration examples
├── monitoring_config.yaml         # Configuration file
└── README_MONITORING.md           # This documentation
```

## 🛠️ Components

### 1. **AdvancedMonitoringSystem**
Main orchestrator that manages all monitoring components:
- Performance profiler
- System health monitor
- Error tracker
- Cost tracker
- Metrics collection

### 2. **PerformanceProfiler**
Tracks and analyzes performance metrics:
```python
from advanced_monitoring import PerformanceProfiler, monitor_operation

# Using decorator
@monitor_operation("my_operation", "component_name")
def my_function():
    # Your code here
    pass

# Manual usage
profiler = PerformanceProfiler()
trace_id = profiler.start_operation("operation_name")
# ... perform operation ...
profiler.end_operation(trace_id, "success")
```

### 3. **SystemHealthMonitor**
Monitors system health with configurable thresholds:
- CPU usage
- Memory usage
- Disk usage
- GPU memory
- Health scoring algorithm

### 4. **ErrorTracker**
Comprehensive error tracking and analysis:
- Error categorization
- Stack trace capture
- Context preservation
- Trend analysis
- SQLite database storage

### 5. **CostTracker**
Tracks computational costs and resource usage:
- Model inference costs
- Compute resource costs
- Cost breakdowns by resource type
- Historical cost analysis

### 6. **MonitoringDashboard**
Real-time web dashboard with:
- Live metrics display
- Interactive charts
- System status overview
- Alert management
- WebSocket-based real-time updates

## 🚀 Quick Start

### 1. **Basic Setup**
```python
from advanced_monitoring import AdvancedMonitoringSystem
from monitoring_dashboard import MonitoringDashboard

# Initialize monitoring system
monitoring_system = AdvancedMonitoringSystem()
monitoring_system.start()

# Start dashboard
dashboard = MonitoringDashboard(monitoring_system)
dashboard.run()  # Access at http://localhost:8080
```

### 2. **Integration with Existing Code**
```python
from enhanced_deepmodel import create_monitored_model

# Create monitored model
model, monitoring_system = create_monitored_model(tokenizer)

# Use model normally - monitoring is automatic
result = model.generate("Your prompt here")
```

### 3. **Complete System Integration**
```python
from monitoring_integration_example import MonitoredMaranSystem

# Create and start complete system
maran_system = MonitoredMaranSystem()
maran_system.start()

# Process queries with full monitoring
result = maran_system.process_query("Your query here")
```

## 📈 Dashboard Features

Access the dashboard at `http://localhost:8080` (default)

### **Overview Section**
- System health score
- CPU/Memory usage
- Cost summary
- Recent alerts

### **Performance Section**
- Operation duration trends
- Throughput metrics
- Resource utilization charts
- Performance bottlenecks

### **Health Section**
- Health score over time
- System metrics
- Threshold violations
- Health trends

### **Errors Section**
- Error distribution
- Error trends
- Error analytics
- Recent errors

### **Costs Section**
- Cost breakdown
- Cost trends
- Resource utilization costs
- Cost optimization insights

### **Logs Section**
- Real-time log streaming
- Log filtering
- Log search
- Context-aware logging

## 📊 Metrics Available

### **Prometheus Metrics** (http://localhost:9090/metrics)
- `operation_duration_seconds` - Operation durations
- `memory_usage_bytes` - Memory usage by operation
- `cpu_usage_percent` - CPU usage by operation
- `gpu_usage_percent` - GPU usage by operation
- `errors_total` - Total error count by type
- `system_health_score` - Overall system health
- `total_cost_usd` - Total accumulated costs

### **Custom Metrics**
- Model generation statistics
- Token processing rates
- Memory allocation patterns
- Error categorization
- Cost breakdowns

## ⚙️ Configuration

### **Environment Variables**
```bash
export MARAN_LOG_LEVEL=INFO
export MARAN_METRICS_PORT=9090
export MARAN_DASHBOARD_PORT=8080
export MARAN_HEALTH_CHECK_INTERVAL=30
```

### **Configuration File** (`monitoring_config.yaml`)
```yaml
monitoring:
  metrics_port: 9090
  dashboard_port: 8080
  health_check_interval: 30
  health_thresholds:
    cpu_percent: 80.0
    memory_percent: 85.0
    disk_percent: 90.0
```

## 🔧 Customization

### **Custom Metrics**
```python
from prometheus_client import Gauge, Counter

# Create custom metrics
custom_metric = Gauge('custom_metric', 'Description')
custom_counter = Counter('custom_operations', 'Operations count')

# Use in your code
custom_metric.set(42)
custom_counter.inc()
```

### **Custom Alerts**
```python
# Add custom alert rules
monitoring_system.health_monitor.thresholds["custom_metric"] = 100.0
```

### **Custom Dashboard Sections**
Extend the dashboard by adding new API endpoints and frontend sections.

## 🐛 Troubleshooting

### **Common Issues**

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :9090
   # Kill the process or change the port
   ```

2. **Permission Errors**
   ```bash
   # Ensure logs directory is writable
   mkdir -p logs
   chmod 755 logs
   ```

3. **Memory Issues**
   ```python
   # Reduce metrics history size
   monitoring_system.performance_profiler.metrics_history.maxlen = 1000
   ```

4. **Dashboard Not Loading**
   - Check if the monitoring system is started
   - Verify the dashboard port is available
   - Check browser console for errors

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system status
status = monitoring_system.get_dashboard_data()
print(json.dumps(status, indent=2))
```

## 📚 API Reference

### **AdvancedMonitoringSystem**
```python
class AdvancedMonitoringSystem:
    def start()                    # Start monitoring
    def stop()                     # Stop monitoring
    def get_dashboard_data()       # Get all dashboard data
```

### **PerformanceProfiler**
```python
class PerformanceProfiler:
    def start_operation(name, context)     # Start tracking
    def end_operation(trace_id, status)    # End tracking
    def get_performance_summary()          # Get summary
```

### **SystemHealthMonitor**
```python
class SystemHealthMonitor:
    def start_monitoring()         # Start health checks
    def stop_monitoring()          # Stop health checks
    def get_health_status()        # Get current status
```

### **ErrorTracker**
```python
class ErrorTracker:
    def track_error(type, severity, component, message)  # Track error
    def get_error_analytics(hours)                       # Get analytics
```

## 🔐 Security Considerations

- Dashboard runs on localhost by default
- No authentication required (development mode)
- Logs may contain sensitive information
- Configure properly for production use

## 🎯 Performance Impact

The monitoring system is designed to have minimal performance impact:
- Async operations where possible
- Efficient metrics collection
- Configurable sampling rates
- Memory-efficient data structures

Expected overhead: < 5% in most scenarios

## 🚀 Production Deployment

### **Docker Setup**
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080 9090
CMD ["python", "monitoring_integration_example.py"]
```

### **Production Configuration**
```yaml
production:
  debug_mode: false
  ssl_enabled: true
  security:
    encrypt_logs: true
    audit_all_operations: true
```

## 📞 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the configuration in `monitoring_config.yaml`
3. Enable debug mode for detailed information
4. Check system resources and permissions

## 🔮 Future Enhancements

- **Machine Learning-based Anomaly Detection**
- **Advanced Alerting with ML Predictions**
- **Integration with External Monitoring Tools**
- **Auto-scaling Based on Metrics**
- **Enhanced Security and Authentication**
- **Distributed Monitoring for Multi-node Setup**

---

## 💡 Examples

See `monitoring_integration_example.py` for complete working examples of:
- System initialization
- Component integration
- Query processing with monitoring
- Dashboard usage
- Error handling
- Performance optimization

The monitoring system transforms Maran from a basic AI agent into a production-ready, observable, and maintainable system with comprehensive monitoring capabilities.