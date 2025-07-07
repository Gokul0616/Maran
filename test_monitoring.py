#!/usr/bin/env python3
# test_monitoring.py
"""
Quick test script to demonstrate the advanced monitoring system
"""

import time
import random
import threading
from advanced_monitoring import (
    AdvancedMonitoringSystem, 
    monitor_operation,
    StructuredLogger
)

logger = StructuredLogger("test_monitoring")

class MonitoringTest:
    """Test class to demonstrate monitoring capabilities"""
    
    def __init__(self):
        self.monitoring_system = AdvancedMonitoringSystem()
        
    def start_monitoring(self):
        """Start the monitoring system"""
        print("🚀 Starting Advanced Monitoring System...")
        self.monitoring_system.start()
        print(f"✅ Monitoring started!")
        print(f"📊 Metrics: http://localhost:9090/metrics")
        print(f"🎯 Dashboard: http://localhost:8080 (if dashboard is running)")
        
    @monitor_operation("sample_computation", "test")
    def sample_computation(self, duration=1):
        """Sample computation to demonstrate monitoring"""
        logger.log_event("INFO", "computation_start", f"Starting computation for {duration}s")
        
        # Simulate some work
        time.sleep(duration)
        
        # Simulate some memory allocation
        data = [random.random() for _ in range(random.randint(1000, 10000))]
        
        logger.log_event("INFO", "computation_complete", 
                         f"Computation completed with {len(data)} data points")
        return data
    
    @monitor_operation("error_simulation", "test")
    def simulate_error(self):
        """Simulate an error to test error tracking"""
        logger.log_event("WARNING", "error_simulation", "Simulating an error...")
        
        # Randomly succeed or fail
        if random.choice([True, False]):
            raise ValueError("This is a simulated error for testing")
        
        return "Success!"
    
    def run_load_test(self, operations=10):
        """Run a load test to generate monitoring data"""
        print(f"\n🔄 Running load test with {operations} operations...")
        
        successful_ops = 0
        failed_ops = 0
        
        for i in range(operations):
            try:
                print(f"Operation {i+1}/{operations}: ", end="")
                
                # Random operation type
                if random.choice([True, False]):
                    # Computation operation
                    duration = random.uniform(0.1, 2.0)
                    result = self.sample_computation(duration)
                    print(f"✅ Computation ({len(result)} items)")
                    successful_ops += 1
                else:
                    # Error simulation
                    result = self.simulate_error()
                    print(f"✅ Success: {result}")
                    successful_ops += 1
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                failed_ops += 1
            
            # Random delay between operations
            time.sleep(random.uniform(0.1, 0.5))
        
        print(f"\n📊 Load test complete:")
        print(f"   ✅ Successful operations: {successful_ops}")
        print(f"   ❌ Failed operations: {failed_ops}")
        print(f"   📈 Success rate: {successful_ops/(successful_ops+failed_ops)*100:.1f}%")
    
    def show_system_status(self):
        """Display current system status"""
        print("\n📊 System Status:")
        
        # Health status
        health = self.monitoring_system.health_monitor.get_health_status()
        print(f"   💚 Health Score: {health.get('health_score', 'N/A')}")
        print(f"   🎯 Status: {health.get('status', 'Unknown')}")
        
        if 'latest_metrics' in health:
            metrics = health['latest_metrics']
            print(f"   🖥️  CPU Usage: {metrics.get('cpu_percent', 'N/A'):.1f}%")
            print(f"   💾 Memory Usage: {metrics.get('memory_percent', 'N/A'):.1f}%")
            print(f"   💿 Disk Usage: {metrics.get('disk_percent', 'N/A'):.1f}%")
        
        # Performance summary
        perf = self.monitoring_system.performance_profiler.get_performance_summary(last_n_minutes=5)
        if perf.get('total_operations', 0) > 0:
            print(f"   ⚡ Avg Duration: {perf.get('avg_duration', 0):.3f}s")
            print(f"   🔢 Total Operations (5m): {perf.get('total_operations', 0)}")
        
        # Error analytics
        errors = self.monitoring_system.error_tracker.get_error_analytics(hours=1)
        if errors.get('total_errors', 0) > 0:
            print(f"   🚨 Total Errors (1h): {errors.get('total_errors', 0)}")
        
        # Cost summary
        costs = self.monitoring_system.cost_tracker.get_cost_summary(hours=1)
        if costs.get('total_cost_usd', 0) > 0:
            print(f"   💰 Total Cost (1h): ${costs.get('total_cost_usd', 0):.4f}")
    
    def run_continuous_monitoring(self, duration_minutes=5):
        """Run continuous monitoring for specified duration"""
        print(f"\n⏰ Running continuous monitoring for {duration_minutes} minutes...")
        print("   This will generate periodic operations to test monitoring")
        
        end_time = time.time() + (duration_minutes * 60)
        operation_count = 0
        
        try:
            while time.time() < end_time:
                operation_count += 1
                
                try:
                    # Perform a random operation
                    if random.choice([True, False, True]):  # 2/3 chance of computation
                        duration = random.uniform(0.1, 1.0)
                        self.sample_computation(duration)
                    else:
                        self.simulate_error()
                        
                except Exception:
                    pass  # Ignore errors for continuous monitoring
                
                # Show status every minute
                if operation_count % 20 == 0:
                    remaining = (end_time - time.time()) / 60
                    print(f"   ⏳ {remaining:.1f} minutes remaining, {operation_count} operations completed")
                    self.show_system_status()
                
                # Wait before next operation
                time.sleep(random.uniform(1, 3))
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping continuous monitoring...")
        
        print(f"✅ Continuous monitoring completed ({operation_count} operations)")

def main():
    """Main test function"""
    print("🧠 Maran Advanced Monitoring System - Test Script")
    print("=" * 60)
    
    test = MonitoringTest()
    
    try:
        # Start monitoring
        test.start_monitoring()
        
        # Wait a moment for system to initialize
        time.sleep(2)
        
        # Show initial status
        test.show_system_status()
        
        # Run load test
        test.run_load_test(15)
        
        # Show status after load test
        test.show_system_status()
        
        # Ask user if they want continuous monitoring
        print("\n" + "="*60)
        print("🎯 Test Options:")
        print("1. Run continuous monitoring (5 minutes)")
        print("2. Exit and keep monitoring running")
        print("3. Exit and stop monitoring")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                test.run_continuous_monitoring(5)
                test.show_system_status()
            elif choice == "2":
                print("\n📊 Monitoring system continues running...")
                print("🌐 You can check metrics at: http://localhost:9090/metrics")
                print("⏰ System will continue monitoring until process is stopped")
                # Keep running
                try:
                    while True:
                        time.sleep(30)
                        test.show_system_status()
                except KeyboardInterrupt:
                    pass
            
        except (EOFError, KeyboardInterrupt):
            pass
        
        print("\n🛑 Stopping monitoring system...")
        test.monitoring_system.stop()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        try:
            test.monitoring_system.stop()
        except:
            pass
        raise

if __name__ == "__main__":
    main()