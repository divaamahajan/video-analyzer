#!/usr/bin/env python3
"""
Example Usage of Basic AI Agent
===============================

This script demonstrates how to use the BasicAIAgent class
for various AI tasks.
"""

import sys
import os
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ai_agent import BasicAIAgent, create_agent

# Load environment variables
load_dotenv()


def example_basic_usage():
    """Example of basic prompt-response usage"""
    print("🤖 Basic AI Agent Example")
    print("=" * 40)
    
    try:
        # Create agent
        agent = create_agent()
        
        # Example 1: Simple question
        print("\n📝 Example 1: Simple Question")
        response = agent.generate_response(
            prompt="What is machine learning?",
            system_prompt="You are a computer science expert. Explain concepts clearly and concisely."
        )
        print(f"Question: What is machine learning?")
        print(f"Answer: {response}")
        
        # Example 2: Creative writing
        print("\n📝 Example 2: Creative Writing")
        response = agent.generate_response(
            prompt="Write a short story about a robot learning to paint.",
            system_prompt="You are a creative writer. Write engaging and imaginative stories.",
            temperature=0.9  # Higher temperature for more creativity
        )
        print(f"Story: {response}")
        
        # Example 3: Code generation
        print("\n📝 Example 3: Code Generation")
        response = agent.generate_response(
            prompt="Write a Python function to calculate fibonacci numbers.",
            system_prompt="You are a Python programming expert. Write clean, efficient code with comments."
        )
        print(f"Code:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_chat_conversation():
    """Example of chat conversation with history"""
    print("\n\n💬 Chat Conversation Example")
    print("=" * 40)
    
    try:
        # Create agent
        agent = create_agent()
        
        # Initialize conversation
        conversation = []
        
        # Chat messages
        messages = [
            "Hello! I'm working on a video analysis project.",
            "What programming languages would you recommend for computer vision?",
            "I'm using Python with OpenCV. Is that a good choice?",
            "What are some best practices for video processing?"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n👤 User: {message}")
            
            result = agent.chat(
                message=message,
                system_prompt="You are a helpful software engineering mentor with expertise in computer vision and video processing.",
                conversation_history=conversation,
                temperature=0.7
            )
            
            print(f"🤖 AI: {result['response']}")
            
            # Update conversation history
            conversation = result['conversation_history']
    
    except Exception as e:
        print(f"❌ Error: {e}")


def example_custom_parameters():
    """Example with custom parameters"""
    print("\n\n⚙️ Custom Parameters Example")
    print("=" * 40)
    
    try:
        # Create agent with specific model
        agent = create_agent(model="gpt-3.5-turbo")
        
        # Show available models
        print("Available models:")
        models = agent.get_available_models()
        for model in models[:5]:  # Show first 5 models
            print(f"  - {model}")
        
        # Example with different parameters
        print(f"\nUsing model: {agent.model}")
        
        response = agent.generate_response(
            prompt="Explain quantum computing in simple terms.",
            system_prompt="You are a physics professor who explains complex topics simply.",
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=150    # Limit response length
        )
        
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_mode():
    """Interactive mode for testing the agent"""
    print("\n\n🎮 Interactive Mode")
    print("=" * 40)
    print("Type 'quit' to exit, 'clear' to clear conversation history")
    
    try:
        agent = create_agent()
        conversation = []
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation = []
                print("🧹 Conversation history cleared!")
                continue
            elif not user_input:
                continue
            
            try:
                result = agent.chat(
                    message=user_input,
                    system_prompt="You are a helpful AI assistant.",
                    conversation_history=conversation
                )
                
                print(f"🤖 AI: {result['response']}")
                conversation = result['conversation_history']
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function to run examples"""
    print("🚀 Basic AI Agent Examples")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return
    
    # Run examples
    example_basic_usage()
    example_chat_conversation()
    example_custom_parameters()
    
    # Ask if user wants interactive mode
    try:
        choice = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
