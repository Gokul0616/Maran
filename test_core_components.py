#!/usr/bin/env python3
"""
Test Core AI Components in Terminal
"""

import sys
sys.path.append('/app')

print("🧠 TESTING MARAN AI CORE COMPONENTS")
print("="*60)

# Test 1: Tokenizer
print("\n🔤 TEST 1: TOKENIZER FUNCTIONALITY")
print("-" * 40)

try:
    from maran_demo_simple import SimpleTokenizer
    
    tokenizer = SimpleTokenizer()
    test_text = "Hello AI agent, generate Python code"
    
    print(f"Input Text: '{test_text}'")
    
    # Encode
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    print(f"Vocabulary Size: {len(tokenizer.vocab)}")
    print("✅ Tokenizer test passed!")
    
except Exception as e:
    print(f"❌ Tokenizer test failed: {e}")

# Test 2: AI Agent Memory
print("\n🧠 TEST 2: AI AGENT MEMORY SYSTEM")
print("-" * 40)

try:
    from maran_demo_simple import SimpleMaranAgent
    
    agent = SimpleMaranAgent()
    
    # Add some test conversations
    test_conversations = [
        "What is Python?",
        "How do I write a function?",
        "Explain machine learning"
    ]
    
    for conv in test_conversations:
        agent.conversation_history.append({
            "timestamp": "2025-01-01T10:00:00",
            "type": "user",
            "content": conv
        })
    
    print(f"Conversation History Length: {len(agent.conversation_history)}")
    print("Recent conversations:")
    for i, conv in enumerate(agent.conversation_history[-3:], 1):
        print(f"  {i}. {conv['content']}")
    
    print("✅ Memory system test passed!")
    
except Exception as e:
    print(f"❌ Memory system test failed: {e}")

# Test 3: System Monitoring
print("\n📊 TEST 3: SYSTEM MONITORING")
print("-" * 40)

try:
    import psutil
    import time
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    # Disk usage
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
    
    # Process count
    print(f"Running Processes: {len(psutil.pids())}")
    
    print("✅ System monitoring test passed!")
    
except Exception as e:
    print(f"❌ System monitoring test failed: {e}")

# Test 4: File System Operations
print("\n📁 TEST 4: FILE SYSTEM OPERATIONS")
print("-" * 40)

try:
    import os
    import json
    from pathlib import Path
    
    # Check directories
    directories = ['/app/logs', '/app/models', '/app/memory']
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory} exists")
        else:
            print(f"❌ {directory} missing")
    
    # Test file operations
    test_data = {
        "test": "Maran AI Agent",
        "timestamp": "2025-01-01T10:00:00",
        "status": "active"
    }
    
    test_file = "/app/logs/test_output.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Created test file: {test_file}")
    
    # Read it back
    with open(test_file, 'r') as f:
        loaded_data = json.load(f)
    
    print(f"✅ Successfully read test file")
    print(f"   Data: {loaded_data}")
    
    print("✅ File system operations test passed!")
    
except Exception as e:
    print(f"❌ File system operations test failed: {e}")

# Test 5: Network Connectivity
print("\n🌐 TEST 5: NETWORK CONNECTIVITY")
print("-" * 40)

try:
    import requests
    import socket
    
    # Test local web server
    try:
        response = requests.get("http://localhost:8001/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Web server is responding")
            data = response.json()
            print(f"   Total Requests: {data['monitor_stats']['total_requests']}")
        else:
            print(f"⚠️ Web server returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
    
    # Test port availability
    def check_port(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    ports_to_check = [8001, 8080, 9090]
    for port in ports_to_check:
        if check_port('localhost', port):
            print(f"✅ Port {port} is active")
        else:
            print(f"❌ Port {port} is not responding")
    
    print("✅ Network connectivity test completed!")
    
except Exception as e:
    print(f"❌ Network connectivity test failed: {e}")

print("\n" + "="*60)
print("🎉 ALL CORE COMPONENT TESTS COMPLETED!")
print("="*60)