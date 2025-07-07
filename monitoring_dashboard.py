# monitoring_dashboard.py
"""
Real-time monitoring dashboard for Maran AI Agent
Provides a web-based interface for monitoring system health, performance, and costs
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path
import asyncio
import aiohttp
from aiohttp import web, web_ws
import aiohttp_cors
import sqlite3
from advanced_monitoring import AdvancedMonitoringSystem, StructuredLogger

class MonitoringDashboard:
    """Real-time web dashboard for monitoring"""
    
    def __init__(self, monitoring_system: AdvancedMonitoringSystem, port: int = 8080):
        self.monitoring_system = monitoring_system
        self.port = port
        self.logger = StructuredLogger("dashboard")
        self.app = web.Application()
        self.websocket_connections = set()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup web routes for dashboard"""
        # Static files
        self.app.router.add_static('/', 'dashboard/static', name='static')
        
        # API endpoints
        self.app.router.add_get('/api/health', self.get_health)
        self.app.router.add_get('/api/performance', self.get_performance)
        self.app.router.add_get('/api/errors', self.get_errors)
        self.app.router.add_get('/api/costs', self.get_costs)
        self.app.router.add_get('/api/dashboard', self.get_dashboard_data)
        self.app.router.add_get('/api/alerts', self.get_alerts)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Main dashboard page
        self.app.router.add_get('/', self.dashboard_page)
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def dashboard_page(self, request):
        """Serve the main dashboard page"""
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maran AI Agent - Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
        .dashboard { display: grid; grid-template-columns: 250px 1fr; height: 100vh; }
        .sidebar { background: #1e293b; padding: 20px; border-right: 1px solid #334155; }
        .main-content { padding: 20px; overflow-y: auto; }
        .logo { font-size: 24px; font-weight: bold; color: #60a5fa; margin-bottom: 30px; }
        .nav-item { padding: 12px; margin: 5px 0; border-radius: 8px; cursor: pointer; transition: all 0.3s; }
        .nav-item:hover, .nav-item.active { background: #3730a3; color: white; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .metric-title { font-size: 14px; color: #94a3b8; margin-bottom: 8px; }
        .metric-value { font-size: 32px; font-weight: bold; margin-bottom: 8px; }
        .metric-change { font-size: 12px; }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .warning { color: #f59e0b; }
        .chart-container { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-healthy { background: #10b981; }
        .status-warning { background: #f59e0b; }
        .status-critical { background: #ef4444; }
        .alert-item { background: #374151; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid #ef4444; }
        .logs-container { background: #1e293b; border-radius: 12px; padding: 20px; height: 300px; overflow-y: auto; border: 1px solid #334155; }
        .log-entry { font-family: 'Courier New', monospace; font-size: 12px; margin: 5px 0; padding: 5px; border-radius: 4px; }
        .log-error { background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; }
        .log-warning { background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; }
        .log-info { background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; }
        .connection-status { position: fixed; top: 20px; right: 20px; padding: 10px; border-radius: 8px; font-size: 12px; }
        .connected { background: #10b981; color: white; }
        .disconnected { background: #ef4444; color: white; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <div class="logo">🧠 Maran Monitor</div>
            <div class="nav-item active" onclick="showSection('overview')">📊 Overview</div>
            <div class="nav-item" onclick="showSection('performance')">⚡ Performance</div>
            <div class="nav-item" onclick="showSection('health')">❤️ Health</div>
            <div class="nav-item" onclick="showSection('errors')">🚨 Errors</div>
            <div class="nav-item" onclick="showSection('costs')">💰 Costs</div>
            <div class="nav-item" onclick="showSection('logs')">📝 Logs</div>
        </div>
        
        <div class="main-content">
            <div id="connection-status" class="connection-status disconnected">● Connecting...</div>
            
            <div id="overview-section">
                <h1>System Overview</h1>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">System Health</div>
                        <div class="metric-value" id="health-score">--</div>
                        <div class="metric-change" id="health-status">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">CPU Usage</div>
                        <div class="metric-value" id="cpu-usage">--%</div>
                        <div class="metric-change" id="cpu-trend">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Memory Usage</div>
                        <div class="metric-value" id="memory-usage">--%</div>
                        <div class="metric-change" id="memory-trend">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Total Cost (24h)</div>
                        <div class="metric-value" id="total-cost">$--</div>
                        <div class="metric-change" id="cost-trend">--</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>Performance Trends</h3>
                    <canvas id="performance-chart"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>Recent Alerts</h3>
                    <div id="alerts-container">Loading alerts...</div>
                </div>
            </div>
            
            <div id="performance-section" style="display:none;">
                <h1>Performance Metrics</h1>
                <div class="chart-container">
                    <h3>Operation Duration Trends</h3>
                    <canvas id="duration-chart"></canvas>
                </div>
                <div id="performance-details"></div>
            </div>
            
            <div id="health-section" style="display:none;">
                <h1>System Health</h1>
                <div class="chart-container">
                    <h3>Health Score Over Time</h3>
                    <canvas id="health-chart"></canvas>
                </div>
                <div id="health-details"></div>
            </div>
            
            <div id="errors-section" style="display:none;">
                <h1>Error Analytics</h1>
                <div class="chart-container">
                    <h3>Error Distribution</h3>
                    <canvas id="errors-chart"></canvas>
                </div>
                <div id="error-details"></div>
            </div>
            
            <div id="costs-section" style="display:none;">
                <h1>Cost Analysis</h1>
                <div class="chart-container">
                    <h3>Cost Breakdown</h3>
                    <canvas id="costs-chart"></canvas>
                </div>
                <div id="cost-details"></div>
            </div>
            
            <div id="logs-section" style="display:none;">
                <h1>System Logs</h1>
                <div class="logs-container" id="logs-container">
                    <div>Loading logs...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let charts = {};
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                document.getElementById('connection-status').textContent = '● Connected';
                document.getElementById('connection-status').className = 'connection-status connected';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                document.getElementById('connection-status').textContent = '● Disconnected';
                document.getElementById('connection-status').className = 'connection-status disconnected';
                setTimeout(initWebSocket, 5000); // Reconnect after 5 seconds
            };
        }
        
        function updateDashboard(data) {
            // Update overview metrics
            if (data.system_health) {
                const health = data.system_health;
                document.getElementById('health-score').textContent = health.health_score || '--';
                document.getElementById('health-status').textContent = health.status || 'Unknown';
                
                if (health.latest_metrics) {
                    document.getElementById('cpu-usage').textContent = `${health.latest_metrics.cpu_percent?.toFixed(1) || '--'}%`;
                    document.getElementById('memory-usage').textContent = `${health.latest_metrics.memory_percent?.toFixed(1) || '--'}%`;
                }
            }
            
            if (data.cost_summary) {
                document.getElementById('total-cost').textContent = `$${data.cost_summary.total_cost_usd?.toFixed(4) || '--'}`;
            }
            
            // Update alerts
            if (data.system_health && data.system_health.recent_alerts) {
                updateAlerts(data.system_health.recent_alerts);
            }
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<div>No recent alerts</div>';
                return;
            }
            
            container.innerHTML = alerts.map(alert => `
                <div class="alert-item">
                    <strong>${alert.severity.toUpperCase()}</strong>: ${alert.message}
                    <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }
        
        function showSection(sectionName) {
            // Hide all sections
            const sections = ['overview', 'performance', 'health', 'errors', 'costs', 'logs'];
            sections.forEach(section => {
                document.getElementById(`${section}-section`).style.display = 'none';
            });
            
            // Show selected section
            document.getElementById(`${sectionName}-section`).style.display = 'block';
            
            // Update nav
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        function initCharts() {
            // Initialize Chart.js charts
            const ctx = document.getElementById('performance-chart');
            if (ctx) {
                charts.performance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Response Time (ms)',
                            data: [],
                            borderColor: '#60a5fa',
                            backgroundColor: 'rgba(96, 165, 250, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { labels: { color: '#e2e8f0' } } },
                        scales: {
                            x: { ticks: { color: '#94a3b8' } },
                            y: { ticks: { color: '#94a3b8' } }
                        }
                    }
                });
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            initCharts();
            
            // Fetch initial data
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(console.error);
        });
    </script>
</body>
</html>
"""
        return web.Response(text=dashboard_html, content_type='text/html')
    
    async def get_health(self, request):
        """Get system health data"""
        try:
            health_data = self.monitoring_system.health_monitor.get_health_status()
            return web.json_response(health_data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_performance(self, request):
        """Get performance data"""
        try:
            hours = int(request.query.get('hours', 1))
            performance_data = self.monitoring_system.performance_profiler.get_performance_summary(
                last_n_minutes=hours * 60
            )
            return web.json_response(performance_data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_errors(self, request):
        """Get error analytics"""
        try:
            hours = int(request.query.get('hours', 24))
            error_data = self.monitoring_system.error_tracker.get_error_analytics(hours)
            return web.json_response(error_data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_costs(self, request):
        """Get cost data"""
        try:
            hours = int(request.query.get('hours', 24))
            cost_data = self.monitoring_system.cost_tracker.get_cost_summary(hours)
            return web.json_response(cost_data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_dashboard_data(self, request):
        """Get comprehensive dashboard data"""
        try:
            dashboard_data = self.monitoring_system.get_dashboard_data()
            return web.json_response(dashboard_data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_alerts(self, request):
        """Get recent alerts"""
        try:
            health_status = self.monitoring_system.health_monitor.get_health_status()
            alerts = health_status.get('recent_alerts', [])
            return web.json_response({"alerts": alerts})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web_ws.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        self.logger.log_event("INFO", "websocket_connect", "New WebSocket connection established")
        
        try:
            # Send initial data
            dashboard_data = self.monitoring_system.get_dashboard_data()
            await ws.send_str(json.dumps(dashboard_data))
            
            async for msg in ws:
                if msg.type == aiohttp.web_ws.WSMsgType.TEXT:
                    # Handle incoming messages if needed
                    pass
                elif msg.type == aiohttp.web_ws.WSMsgType.ERROR:
                    self.logger.log_event("ERROR", "websocket_error", f"WebSocket error: {ws.exception()}")
                    
        except Exception as e:
            self.logger.log_event("ERROR", "websocket_handler_error", f"WebSocket handler error: {str(e)}")
        finally:
            self.websocket_connections.discard(ws)
            self.logger.log_event("INFO", "websocket_disconnect", "WebSocket connection closed")
        
        return ws
    
    async def broadcast_update(self, data):
        """Broadcast updates to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for ws in self.websocket_connections:
            try:
                await ws.send_str(message)
            except Exception as e:
                self.logger.log_event("WARNING", "websocket_broadcast_error", f"Failed to send to WebSocket: {str(e)}")
                disconnected.add(ws)
        
        # Remove disconnected connections
        self.websocket_connections -= disconnected
    
    def start_background_updates(self):
        """Start background task to send periodic updates"""
        async def update_loop():
            while True:
                try:
                    dashboard_data = self.monitoring_system.get_dashboard_data()
                    await self.broadcast_update(dashboard_data)
                    await asyncio.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    self.logger.log_event("ERROR", "background_update_error", f"Background update error: {str(e)}")
                    await asyncio.sleep(30)  # Wait longer on error
        
        asyncio.create_task(update_loop())
    
    def run(self):
        """Start the dashboard server"""
        self.start_background_updates()
        self.logger.log_event("INFO", "dashboard_start", f"Starting monitoring dashboard on port {self.port}")
        
        web.run_app(
            self.app,
            host='0.0.0.0',
            port=self.port,
            access_log=None  # Disable access log for cleaner output
        )

if __name__ == "__main__":
    # Example usage
    monitoring_system = AdvancedMonitoringSystem()
    monitoring_system.start()
    
    dashboard = MonitoringDashboard(monitoring_system)
    dashboard.run()