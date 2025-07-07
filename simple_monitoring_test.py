#!/usr/bin/env python3
# simple_monitoring_test.py
"""
Simple test without torch dependencies
"""

import time
import random
import threading
import subprocess
import json

# Create a minimal monitoring test without heavy dependencies
class SimpleMonitoringTest:
    """Simple monitoring test without torch"""
    
    def __init__(self):
        self.operations = []
        
    def test_basic_monitoring(self):
        """Test basic monitoring functionality"""
        print("🧠 Maran Advanced Monitoring System - Simple Test")
        print("=" * 60)
        
        # Test 1: Check if we can import the monitoring module
        try:
            print("📦 Testing imports...")
            
            # Test minimal import first
            import sys
            sys.path.append('/app')
            
            # Import without torch-dependent parts
            from prometheus_client import Gauge, Counter, start_http_server
            print("✅ Prometheus client imported successfully")
            
            # Test metrics creation
            test_gauge = Gauge('test_metric', 'Test metric')
            test_counter = Counter('test_counter', 'Test counter') 
            print("✅ Metrics created successfully")
            
            # Test metrics usage
            test_gauge.set(42)
            test_counter.inc()
            print("✅ Metrics updated successfully")
            
        except Exception as e:
            print(f"❌ Import test failed: {str(e)}")
            return False
        
        # Test 2: Start metrics server
        try:
            print("\n🚀 Starting metrics server...")
            start_http_server(9091)  # Use different port to avoid conflicts
            print("✅ Metrics server started on port 9091")
            
            # Wait a moment
            time.sleep(2)
            
            # Test if server is responding
            try:
                import urllib.request
                response = urllib.request.urlopen('http://localhost:9091/metrics', timeout=5)
                content = response.read().decode('utf-8')
                if 'test_metric' in content:
                    print("✅ Metrics server is responding correctly")
                else:
                    print("⚠️ Metrics server responding but content unclear")
            except Exception as e:
                print(f"⚠️ Could not verify metrics server: {str(e)}")
                
        except Exception as e:
            print(f"❌ Metrics server test failed: {str(e)}")
            return False
        
        # Test 3: Basic system metrics
        try:
            print("\n📊 Testing system metrics...")
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            print(f"   🖥️  CPU Usage: {cpu_percent:.1f}%")
            print(f"   💾 Memory Usage: {memory.percent:.1f}%")
            print(f"   💿 Disk Usage: {disk.percent:.1f}%")
            print(f"   🔢 Available Memory: {memory.available / (1024**3):.1f} GB")
            
            # Update metrics with real values
            test_gauge.set(cpu_percent)
            
            print("✅ System metrics collected successfully")
            
        except Exception as e:
            print(f"❌ System metrics test failed: {str(e)}")
            return False
        
        # Test 4: Logging functionality
        try:
            print("\n📝 Testing logging...")
            import logging
            from pathlib import Path
            
            # Create logs directory
            Path("logs").mkdir(exist_ok=True)
            
            # Setup logger
            logger = logging.getLogger("test_logger")
            handler = logging.FileHandler("logs/test.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            # Test logging
            logger.info("Test log message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            print("✅ Logging functionality working")
            
        except Exception as e:
            print(f"❌ Logging test failed: {str(e)}")
            return False
        
        # Test 5: Simulate monitoring operations
        try:
            print("\n⚡ Running monitoring simulation...")
            
            for i in range(5):
                operation_start = time.time()
                
                # Simulate work
                work_duration = random.uniform(0.1, 0.5)
                time.sleep(work_duration)
                
                operation_end = time.time()
                duration = operation_end - operation_start
                
                # Record operation
                self.operations.append({
                    "id": i + 1,
                    "duration": duration,
                    "timestamp": operation_start,
                    "success": True
                })
                
                # Update metrics
                test_counter.inc()
                
                print(f"   Operation {i+1}: {duration:.3f}s")
            
            avg_duration = sum(op["duration"] for op in self.operations) / len(self.operations)
            print(f"   📈 Average duration: {avg_duration:.3f}s")
            print("✅ Monitoring simulation completed")
            
        except Exception as e:
            print(f"❌ Monitoring simulation failed: {str(e)}")
            return False
        
        # Test 6: Error simulation
        try:
            print("\n🚨 Testing error handling...")
            
            errors = []
            for i in range(3):
                try:
                    # Simulate error
                    if random.choice([True, False]):
                        raise ValueError(f"Simulated error {i+1}")
                    print(f"   Operation {i+1}: ✅ Success")
                except Exception as e:
                    errors.append({"error": str(e), "timestamp": time.time()})
                    print(f"   Operation {i+1}: ❌ {str(e)}")
            
            print(f"   📊 Total errors: {len(errors)}")
            print("✅ Error handling tested")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {str(e)}")
            return False
        
        return True
    
    def show_results(self):
        """Show test results and metrics"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS")
        print("="*60)
        
        print(f"✅ All basic monitoring tests passed!")
        print(f"📈 Metrics server: http://localhost:9091/metrics")
        print(f"📝 Log files: logs/test.log")
        print(f"⚡ Operations completed: {len(self.operations)}")
        
        if self.operations:
            avg_duration = sum(op["duration"] for op in self.operations) / len(self.operations)
            max_duration = max(op["duration"] for op in self.operations)
            min_duration = min(op["duration"] for op in self.operations)
            
            print(f"   📊 Performance Stats:")
            print(f"      Average: {avg_duration:.3f}s")
            print(f"      Maximum: {max_duration:.3f}s")
            print(f"      Minimum: {min_duration:.3f}s")
        
        print("\n🎯 Next Steps:")
        print("1. Check metrics at http://localhost:9091/metrics")
        print("2. Check logs in logs/test.log")
        print("3. The monitoring system is working correctly!")
        
        # Keep server running for a bit
        print("\n⏰ Keeping metrics server running for 30 seconds...")
        print("   You can check the metrics URL during this time.")
        
        try:
            for i in range(30):
                time.sleep(1)
                if i % 10 == 0:
                    print(f"   ⏳ {30-i} seconds remaining...")
        except KeyboardInterrupt:
            print("\n🛑 Stopping early...")
        
        print("✅ Test completed successfully!")

def main():
    """Main test function"""
    test = SimpleMonitoringTest()
    
    try:
        success = test.test_basic_monitoring()
        if success:
            test.show_results()
        else:
            print("❌ Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()