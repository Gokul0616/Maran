# monitoring_integration_example.py
"""
Example demonstrating how to integrate the advanced monitoring system with Maran
"""

import asyncio
import time
import threading
from pathlib import Path

# Import monitoring components
from advanced_monitoring import (
    AdvancedMonitoringSystem,
    monitor_operation,
    StructuredLogger
)
from monitoring_dashboard import MonitoringDashboard
from enhanced_deepmodel import create_monitored_model, enhanced_train_model

# Import existing Maran components
from tokenizers.BPETokenizer import CustomBPETokenizer
from reasoning import ReasoningAgent
from self_improvement import SelfImprovementEngine, CodeValidator
from autonomous.agent.agent import AutonomousAgent
from autonomous.Memory.memory_store import GPTMemoryStore

class MonitoredMaranSystem:
    """Enhanced Maran system with comprehensive monitoring"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.logger = StructuredLogger("maran_system")
        
        # Initialize monitoring system
        self.monitoring_system = AdvancedMonitoringSystem(
            metrics_port=self.config['monitoring']['metrics_port']
        )
        
        # Initialize dashboard
        self.dashboard = MonitoringDashboard(
            self.monitoring_system,
            port=self.config['monitoring']['dashboard_port']
        )
        
        # Core components (will be initialized)
        self.model = None
        self.tokenizer = None
        self.reasoning_agent = None
        self.improvement_engine = None
        self.autonomous_agent = None
        self.memory_store = None
        
        self.logger.log_event(
            "INFO", "system_init", 
            "Monitored Maran system initialized"
        )
    
    def _default_config(self):
        """Default configuration for the system"""
        return {
            'model': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'max_len': 512,
                'dropout': 0.1,
                'vocab_size': 10000
            },
            'training': {
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'grad_accum_steps': 4
            },
            'monitoring': {
                'metrics_port': 9090,
                'dashboard_port': 8080,
                'health_check_interval': 30,
                'log_level': 'INFO'
            },
            'safety': {
                'code_execution_timeout': 5,
                'memory_limit_mb': 512,
                'use_docker': True
            }
        }
    
    @monitor_operation("system_startup", "system")
    def start(self):
        """Start the complete monitored system"""
        try:
            # Start monitoring first
            self.monitoring_system.start()
            self.logger.log_event("INFO", "monitoring_started", "Monitoring system started")
            
            # Initialize core components
            self._initialize_components()
            
            # Start dashboard in background
            dashboard_thread = threading.Thread(
                target=self._start_dashboard,
                daemon=True
            )
            dashboard_thread.start()
            
            self.logger.log_event(
                "INFO", "system_startup_complete", 
                "Maran system started successfully with monitoring"
            )
            
            return True
            
        except Exception as e:
            self.monitoring_system.error_tracker.track_error(
                error_type=type(e).__name__,
                severity="critical",
                component="system_startup",
                message=f"System startup failed: {str(e)}"
            )
            raise
    
    def _start_dashboard(self):
        """Start the monitoring dashboard"""
        try:
            self.dashboard.run()
        except Exception as e:
            self.logger.log_event(
                "ERROR", "dashboard_error",
                f"Dashboard failed to start: {str(e)}"
            )
    
    @monitor_operation("component_initialization", "system")
    def _initialize_components(self):
        """Initialize all core Maran components with monitoring"""
        
        # Initialize tokenizer
        self.logger.log_event("INFO", "tokenizer_init", "Initializing tokenizer")
        self.tokenizer = CustomBPETokenizer(vocab_size=self.config['model']['vocab_size'])
        
        # Create monitored model
        self.logger.log_event("INFO", "model_init", "Initializing monitored model")
        self.model, _ = create_monitored_model(self.tokenizer, self.monitoring_system)
        
        # Initialize reasoning agent with monitoring
        self.logger.log_event("INFO", "reasoning_init", "Initializing reasoning agent")
        self.reasoning_agent = MonitoredReasoningAgent(
            self.model, self.monitoring_system
        )
        
        # Initialize memory store with monitoring
        self.logger.log_event("INFO", "memory_init", "Initializing memory store")
        self.memory_store = MonitoredMemoryStore(
            self.model, self.tokenizer, self.monitoring_system
        )
        
        # Initialize self-improvement engine with monitoring
        self.logger.log_event("INFO", "improvement_init", "Initializing improvement engine")
        code_validator = MonitoredCodeValidator(self.monitoring_system)
        self.improvement_engine = MonitoredSelfImprovementEngine(
            self.model, code_validator, self.monitoring_system
        )
        
        # Initialize autonomous agent
        self.logger.log_event("INFO", "agent_init", "Initializing autonomous agent")
        self.autonomous_agent = MonitoredAutonomousAgent(
            self.model, self.memory_store, self.reasoning_agent, self.monitoring_system
        )
    
    @monitor_operation("user_query_processing", "interaction")
    def process_query(self, query: str) -> dict:
        """Process a user query with full monitoring"""
        try:
            # Start query processing
            trace_id = self.monitoring_system.performance_profiler.start_operation(
                "user_query",
                {"query_length": len(query), "query_type": "text"}
            )
            
            # Use reasoning agent to process query
            plan = self.reasoning_agent.process_query(query)
            
            # Execute plan with autonomous agent
            result = self.autonomous_agent.execute_plan(plan)
            
            # Learn from interaction
            self.improvement_engine.learn_from_interaction(query, plan, result)
            
            self.monitoring_system.performance_profiler.end_operation(
                trace_id, "success",
                {"result_type": type(result).__name__, "plan_steps": len(plan) if hasattr(plan, '__len__') else 1}
            )
            
            return {
                "query": query,
                "plan": str(plan),
                "result": result,
                "trace_id": trace_id
            }
            
        except Exception as e:
            self.monitoring_system.error_tracker.track_error(
                error_type=type(e).__name__,
                severity="error",
                component="query_processing",
                message=f"Query processing failed: {str(e)}",
                trace_id=trace_id if 'trace_id' in locals() else None
            )
            raise
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            "system_health": self.monitoring_system.health_monitor.get_health_status(),
            "model_stats": self.model.get_model_stats() if self.model else {},
            "dashboard_url": f"http://localhost:{self.config['monitoring']['dashboard_port']}",
            "metrics_url": f"http://localhost:{self.config['monitoring']['metrics_port']}/metrics",
            "component_status": {
                "model_initialized": self.model is not None,
                "tokenizer_initialized": self.tokenizer is not None,
                "reasoning_agent_initialized": self.reasoning_agent is not None,
                "autonomous_agent_initialized": self.autonomous_agent is not None,
                "memory_store_initialized": self.memory_store is not None
            }
        }
    
    def stop(self):
        """Gracefully stop the system"""
        try:
            self.logger.log_event("INFO", "system_shutdown", "Stopping Maran system")
            self.monitoring_system.stop()
            self.logger.log_event("INFO", "system_stopped", "Maran system stopped")
        except Exception as e:
            self.logger.log_event("ERROR", "shutdown_error", f"Error during shutdown: {str(e)}")

# Enhanced wrapper classes for existing components

class MonitoredReasoningAgent:
    """Wrapper for ReasoningAgent with monitoring"""
    
    def __init__(self, model, monitoring_system):
        self.monitoring_system = monitoring_system
        self.logger = StructuredLogger("reasoning_agent")
        # Initialize actual reasoning agent when dependencies are available
        self.agent = None
    
    @monitor_operation("reasoning_process", "reasoning")
    def process_query(self, query: str):
        """Process query with monitoring"""
        self.logger.log_event("INFO", "reasoning_start", f"Processing query: {query[:100]}...")
        
        # Simulate reasoning process (replace with actual ReasoningAgent when available)
        plan = f"Plan for: {query}"
        
        self.logger.log_event("INFO", "reasoning_complete", "Reasoning completed")
        return plan

class MonitoredMemoryStore:
    """Wrapper for GPTMemoryStore with monitoring"""
    
    def __init__(self, model, tokenizer, monitoring_system):
        self.monitoring_system = monitoring_system
        self.logger = StructuredLogger("memory_store")
        # Initialize actual memory store when dependencies are available
        self.store = None
    
    @monitor_operation("memory_query", "memory")
    def query(self, query_text: str, top_k: int = 5):
        """Query memory with monitoring"""
        self.logger.log_event("INFO", "memory_query", f"Querying memory: {query_text[:100]}...")
        # Simulate memory query
        return [f"Memory result {i}" for i in range(top_k)]
    
    @monitor_operation("memory_write", "memory")
    def write(self, text: str, metadata: dict = None):
        """Write to memory with monitoring"""
        self.logger.log_event("INFO", "memory_write", f"Writing to memory: {text[:100]}...")
        # Simulate memory write
        pass

class MonitoredCodeValidator:
    """Wrapper for CodeValidator with monitoring"""
    
    def __init__(self, monitoring_system):
        self.monitoring_system = monitoring_system
        self.logger = StructuredLogger("code_validator")
        self.validator = CodeValidator()
    
    @monitor_operation("code_validation", "safety")
    def validate_code(self, code: str, tests: str = None):
        """Validate code with monitoring"""
        try:
            result = self.validator.validate_code(code, tests)
            
            self.logger.log_event(
                "INFO", "code_validation_complete",
                f"Code validation result: {result.get('valid_syntax', False)}",
                context=result
            )
            
            return result
            
        except Exception as e:
            self.monitoring_system.error_tracker.track_error(
                error_type=type(e).__name__,
                severity="error",
                component="code_validation",
                message=f"Code validation failed: {str(e)}"
            )
            raise

class MonitoredSelfImprovementEngine:
    """Wrapper for SelfImprovementEngine with monitoring"""
    
    def __init__(self, model, code_validator, monitoring_system):
        self.monitoring_system = monitoring_system
        self.logger = StructuredLogger("improvement_engine")
        self.model = model
        self.code_validator = code_validator
    
    @monitor_operation("self_improvement", "learning")
    def learn_from_interaction(self, query: str, plan, result):
        """Learn from interaction with monitoring"""
        self.logger.log_event(
            "INFO", "learning_interaction",
            "Learning from user interaction",
            context={"query_length": len(query), "result_type": type(result).__name__}
        )
        # Simulate learning process
        pass

class MonitoredAutonomousAgent:
    """Wrapper for AutonomousAgent with monitoring"""
    
    def __init__(self, model, memory_store, reasoning_agent, monitoring_system):
        self.monitoring_system = monitoring_system
        self.logger = StructuredLogger("autonomous_agent")
        self.model = model
        self.memory_store = memory_store
        self.reasoning_agent = reasoning_agent
    
    @monitor_operation("plan_execution", "agent")
    def execute_plan(self, plan):
        """Execute plan with monitoring"""
        self.logger.log_event("INFO", "plan_execution", f"Executing plan: {str(plan)[:100]}...")
        
        # Simulate plan execution
        result = f"Executed: {plan}"
        
        self.logger.log_event("INFO", "plan_complete", "Plan execution completed")
        return result

# Example usage and demo functions

async def run_monitoring_demo():
    """Run a demonstration of the monitoring system"""
    print("🧠 Starting Maran Advanced Monitoring Demo...")
    
    # Create and start the monitored system
    maran_system = MonitoredMaranSystem()
    
    try:
        # Start the system
        maran_system.start()
        
        print(f"✅ System started successfully!")
        print(f"📊 Dashboard: http://localhost:8080")
        print(f"📈 Metrics: http://localhost:9090/metrics")
        
        # Simulate some operations
        print("\n🔄 Running sample operations...")
        
        # Process some sample queries
        sample_queries = [
            "Write a function to calculate fibonacci numbers",
            "Create a simple web scraper",
            "Implement a binary search algorithm",
            "Build a REST API endpoint"
        ]
        
        for i, query in enumerate(sample_queries):
            print(f"Processing query {i+1}: {query}")
            result = maran_system.process_query(query)
            print(f"Result: {result['result'][:100]}...")
            
            # Add some delay to see monitoring in action
            time.sleep(2)
        
        # Show system status
        print("\n📊 System Status:")
        status = maran_system.get_system_status()
        print(f"Health Score: {status['system_health'].get('health_score', 'N/A')}")
        print(f"Status: {status['system_health'].get('status', 'Unknown')}")
        
        # Keep running for demonstration
        print("\n⏳ System running... Check the dashboard for real-time monitoring!")
        print("Press Ctrl+C to stop")
        
        # Keep the demo running
        while True:
            await asyncio.sleep(10)
            
            # Simulate periodic activity
            result = maran_system.process_query("Generate a random code snippet")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        maran_system.stop()
        print("✅ System stopped successfully")
    
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        maran_system.stop()
        raise

def main():
    """Main entry point for the monitoring demo"""
    try:
        asyncio.run(run_monitoring_demo())
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()