#!/usr/bin/env python3
"""
Basic AI Agent
==============

A simple AI agent that takes a prompt and system prompt as input
and returns text output using OpenAI's API.
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BasicAIAgent:
    """
    A basic AI agent that uses OpenAI's API to process prompts and return responses.
    
    Attributes:
        client: OpenAI client instance
        model: The model to use for generating responses (default: gpt-3.5-turbo)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the AI agent.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variables.
            model: The OpenAI model to use for generating responses.
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the given prompt and system prompt.
        
        Args:
            prompt: The user's input prompt
            system_prompt: The system prompt to set the AI's behavior
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens in the response
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            The AI's response as a string
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def chat(
        self, 
        message: str, 
        system_prompt: str = "You are a helpful AI assistant.",
        conversation_history: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat with the AI agent, maintaining conversation history.
        
        Args:
            message: The user's message
            system_prompt: The system prompt to set the AI's behavior
            conversation_history: Previous conversation messages (optional)
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            Dictionary containing the response and updated conversation history
        """
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            return {
                "response": ai_response,
                "conversation_history": conversation_history
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate chat response: {str(e)}")
    
    def get_available_models(self) -> list:
        """
        Get list of available OpenAI models.
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Warning: Could not fetch available models: {str(e)}")
            return []
    
    def set_model(self, model: str) -> None:
        """
        Change the model used by the agent.
        
        Args:
            model: The new model name to use
        """
        self.model = model
        print(f"Model changed to: {model}")


def create_agent(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> BasicAIAgent:
    """
    Factory function to create a BasicAIAgent instance.
    
    Args:
        api_key: OpenAI API key (optional)
        model: The model to use (default: gpt-3.5-turbo)
        
    Returns:
        BasicAIAgent instance
    """
    return BasicAIAgent(api_key=api_key, model=model)


# Example usage
if __name__ == "__main__":
    # Create an agent
    agent = create_agent()
    
    # Example 1: Simple prompt-response
    print("=== Example 1: Simple Prompt-Response ===")
    response = agent.generate_response(
        prompt="What is the capital of France?",
        system_prompt="You are a geography expert."
    )
    print(f"Response: {response}\n")
    
    # Example 2: Chat with conversation history
    print("=== Example 2: Chat with History ===")
    conversation = []
    
    # First message
    result1 = agent.chat(
        message="Hello! My name is Alice.",
        system_prompt="You are a friendly assistant.",
        conversation_history=conversation
    )
    print(f"AI: {result1['response']}")
    
    # Second message (with history)
    result2 = agent.chat(
        message="What's my name?",
        conversation_history=conversation
    )
    print(f"AI: {result2['response']}")
