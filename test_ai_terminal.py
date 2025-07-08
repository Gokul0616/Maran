#!/usr/bin/env python3
"""
Terminal-based AI Agent Tester
Test the Maran AI Agent capabilities directly from command line
"""

import asyncio
import json
import sys
import time
from datetime import datetime

# Import our AI agent from the demo
sys.path.append('/app')
from maran_demo_simple import SimpleMaranAgent

class TerminalAITester:
    def __init__(self):
        self.agent = SimpleMaranAgent()
        
    async def test_conversation(self, query):
        """Test AI conversation capabilities"""
        print(f"\n{'='*60}")
        print(f"🤖 MARAN AI AGENT - TERMINAL TEST")
        print(f"{'='*60}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"❓ Query: {query}")
        print(f"{'='*60}")
        
        # Process the query
        start_time = time.time()
        response_data = await self.agent.process_query(query)
        end_time = time.time()
        
        # Display the response
        print(f"🧠 AI RESPONSE:")
        print(f"   {response_data['response']}")
        print(f"\n📊 METADATA:")
        print(f"   • Model: {response_data['model_used']}")
        print(f"   • Confidence: {response_data['confidence']:.2%}")
        print(f"   • Response Time: {(end_time - start_time):.2f} seconds")
        
        # Display thought process
        if response_data.get('thoughts'):
            print(f"\n🧠 AI THOUGHT PROCESS:")
            for i, thought in enumerate(response_data['thoughts'], 1):
                print(f"   {i}. {thought['type'].upper()}: {thought['content']}")
                print(f"      Confidence: {thought['confidence']:.1%}")
        
        return response_data
    
    def show_system_status(self):
        """Show current system status"""
        status = self.agent.get_system_status()
        
        print(f"\n{'='*60}")
        print(f"📊 SYSTEM STATUS")
        print(f"{'='*60}")
        print(f"🔢 STATISTICS:")
        print(f"   • Total Requests: {status['monitor_stats']['total_requests']}")
        print(f"   • AI Generations: {status['monitor_stats']['total_generations']}")
        print(f"   • System Uptime: {status['monitor_stats']['uptime']} seconds")
        print(f"   • Avg Response Time: {status['monitor_stats']['avg_response_time']}s")
        
        print(f"\n💾 MEMORY:")
        print(f"   • Total Memories: {status['memory_stats']['total_memories']}")
        print(f"   • Recent Conversations: {status['memory_stats']['recent_conversations']}")
        
        print(f"\n🖥️ SYSTEM INFO:")
        print(f"   • CPU Cores: {status['system_info']['cpu_count']}")
        print(f"   • Available Memory: {status['system_info']['memory_available']}GB")
        print(f"   • GPU Available: {status['system_info']['gpu_available']}")
        
        print(f"\n🤖 MODEL INFO:")
        print(f"   • Model Type: {status['model_info']['model_type']}")
        print(f"   • Parameters: {status['model_info']['model_parameters']:,}")
        
        print(f"\n🛠️ ACTIVE TOOLS:")
        for tool, stats in status['tool_stats'].items():
            print(f"   • {tool.replace('_', ' ').title()}: {stats['status']} ({stats['calls']} calls)")

async def main():
    """Main testing function"""
    tester = TerminalAITester()
    
    # Show initial system status
    tester.show_system_status()
    
    # Test queries
    test_queries = [
        "Hello, what are your capabilities?",
        "Explain artificial intelligence in simple terms",
        "Generate a Python function to calculate fibonacci numbers",
        "What is the meaning of life?",
        "How does machine learning work?"
    ]
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING AI CAPABILITY TESTS")
    print(f"{'='*60}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}/{len(test_queries)}")
        await tester.test_conversation(query)
        
        if i < len(test_queries):
            print(f"\n⏳ Waiting 2 seconds before next test...")
            await asyncio.sleep(2)
    
    # Show final system status
    print(f"\n{'='*60}")
    print(f"✅ ALL TESTS COMPLETED")
    print(f"{'='*60}")
    tester.show_system_status()
    
    print(f"\n🎉 Maran AI Agent testing completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())