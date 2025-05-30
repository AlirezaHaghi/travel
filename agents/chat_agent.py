import os
from langchain_core.messages import SystemMessage, HumanMessage
import json
from typing import Generator
from rich import print


class MockAIMessage:
    """Mock AI message for when API key is not available"""

    def __init__(self, content: str):
        self.content = content


class ChatAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the ChatAgent with specified model."""
        self.mock_mode = False

        try:
            # Check if OpenAI API key is available
            if not os.getenv("OPENAI_API_KEY"):
                print("[WARNING] No OPENAI_API_KEY found. Running in mock mode.")
                self.mock_mode = True
                self.model = None
            else:
                from langchain_openai import ChatOpenAI

                self.model = ChatOpenAI(model=model_name, temperature=0.7, streaming=True)
        except Exception as e:
            print(f"[WARNING] Could not initialize OpenAI model: {e}. Running in mock mode.")
            self.mock_mode = True
            self.model = None

        # Define required fields (these must be filled)
        self.required_fields = ["city", "days", "budget", "people", "kids", "health", "start_date"]
        # Define all fields, including optional ones
        self.all_fields = self.required_fields + ["hobbies", "specificRequirements"]
        self.conversation_history = []

    def _mock_response_generator(self, message: str):
        """Generate mock responses when API is not available"""
        mock_responses = [
            "Hello! I'm currently running in demo mode. Please tell me about your travel plans.",
            "That sounds interesting! Can you tell me more about your destination and travel dates?",
            "Great! What's your budget for this trip?",
            "Perfect! How many people will be traveling?",
            "Excellent! I'm gathering your information. This is a demo response since no API key is configured.",
        ]

        # Simple logic to return different responses
        response_text = mock_responses[len(self.conversation_history) % len(mock_responses)]

        # Yield the response as chunks to simulate streaming
        for word in response_text.split():
            yield MockAIMessage(word + " ")

    def _init_system_message(self):
        """Initialize system message for the conversation."""
        return SystemMessage(
            content="""
        You are a helpful travel assistant. Your job is to collect information about the user's travel plans.
        Be friendly, conversational, and help the user plan their trip. Collect all necessary information.
        Also pay attention to any specific requirements the traveler mentions, such as accessibility needs,
        food restrictions, special interests, or any constraints that might affect their trip.
        """
        )

    def collect_info(self, user_input: str, state: dict = None) -> dict:
        """Check for missing information and ask user questions to complete the required information."""
        if state is None:
            state = {}

        # Set default health status to "ok" if not provided
        if "health" not in state or not state.get("health"):
            state["health"] = "healthy"

        # Initialize conversation if it's empty
        if not self.conversation_history:
            self.conversation_history.append(self._init_system_message())

        # Merge new inputs into state
        if user_input and user_input.strip():
            new_info = self.extract_info_from_message(user_input)
            for field, value in new_info.items():
                if value:
                    state[field] = value
                    print(f"Updated state: {field} = {value}")  # Debug log

        # Add user input to conversation if not empty
        if user_input and user_input.strip():
            self.conversation_history.append(HumanMessage(content=user_input))

        if self.mock_mode:
            # Return mock response
            return {
                "stream": self._mock_response_generator(user_input),
                "missing_fields": [f for f in self.required_fields if not state.get(f)],
                "complete": len([f for f in self.required_fields if not state.get(f)]) == 0,
                "state": state.copy(),
            }

        # Get AI response based on current state and conversation history
        messages = self.conversation_history.copy()
        messages.append(
            SystemMessage(
                content=f"""
        Current state: {json.dumps(state, ensure_ascii=False)}
        Required fields: {json.dumps(self.required_fields, ensure_ascii=False)}
        Missing fields: {json.dumps([f for f in self.required_fields if not state.get(f)], ensure_ascii=False)}
        Please help the user complete the missing information in a natural way.
        Remember to acknowledge information that has already been provided.
        Tell the user that they can write "not decided" for the start date if they don't have a specific date in mind.
        Also pay attention to any specific requirements they mention and reflect these in your responses.
        """
            )
        )

        try:
            print(f"******************* Messages: {messages}")
            response = self.model.stream(messages)
            return {
                "stream": response,
                "missing_fields": [f for f in self.required_fields if not state.get(f)],
                "complete": len([f for f in self.required_fields if not state.get(f)]) == 0,
                "state": state.copy(),
            }
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return {
                "stream": None,
                "missing_fields": [f for f in self.required_fields if not state.get(f)],
                "complete": False,
                "state": state.copy(),
                "error": str(e),
            }

    def interact_with_user(self, message: str, state: dict = None) -> Generator:
        """Process user message and generate a streaming response."""
        if state is None:
            state = {}

        # Add user message to conversation
        self.conversation_history.append(HumanMessage(content=message))

        if self.mock_mode:
            return self._mock_response_generator(message)

        # Generate streaming response based on the conversation history
        try:
            return self.model.stream(self.conversation_history)
        except Exception as e:
            print(f"Error in interact_with_user: {e}")
            return None

    def extract_info_from_message(self, message: str) -> dict:
        """Use LLM to extract structured travel information from user message."""

        if self.mock_mode:
            # Simple keyword-based extraction for demo mode
            extracted = {}
            message_lower = message.lower()

            # Simple pattern matching for demo
            if "paris" in message_lower:
                extracted["city"] = "Paris"
            elif "london" in message_lower:
                extracted["city"] = "London"
            elif "new york" in message_lower:
                extracted["city"] = "New York"

            # Look for numbers that might be days or people
            import re

            numbers = re.findall(r'\d+', message)
            if numbers:
                if "day" in message_lower:
                    extracted["days"] = numbers[0]
                elif "people" in message_lower or "person" in message_lower:
                    extracted["people"] = numbers[0]

            # Look for budget keywords
            if "$" in message or "budget" in message_lower:
                budget_numbers = re.findall(r'\$?(\d+)', message)
                if budget_numbers:
                    extracted["budget"] = budget_numbers[0]

            return extracted

        system_prompt = f"""Extract the following travel information from the user's message and return JSON.
        Carefully analyze the message to understand both explicit and implicit information.
        
        For example, if the user says "without kids" or "no children", set "kids" to "no".
        If they mention "all adults", also set "kids" to "no".
        If they mention family with children, set "kids" to "yes".
        The people field should be an integer.
        Specifically, if the user gives a start date, set "start_date" in YYYY-MM-DD format string. Otherwise, set "start_date" to "not decided".
        Pay attention to negations and context. Don't just look for keywords, understand the meaning.
        
        IMPORTANT: Also extract any specific requirements, constraints, or special requests the user mentions. 
        This includes but is not limited to:
        - Accessibility needs (e.g., wheelchair access, limited mobility)
        - Food restrictions or dietary preferences
        - Special interests or experiences they want to have
        - Particular constraints (e.g., fear of heights, need quiet accommodations)
        - Any important preferences not covered by other fields
        
        Return the following JSON structure:
        {{
        {', '.join([f'"{field}": ""' for field in self.all_fields])}
        }}
        
        For required fields, if any information is missing or unclear, leave it as an empty string.
        For specificRequirements, capture any important preferences or constraints mentioned by the user.
        """

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]

        try:
            llm_response = self.model.invoke(messages)
            extracted_info = json.loads(llm_response.content)

            # Only update fields that have non-empty values
            filtered_info = {}
            for field in self.all_fields:
                value = extracted_info.get(field, "")
                if value:  # Only include non-empty values
                    filtered_info[field] = value

            return filtered_info
        except Exception as e:
            print("Error parsing LLM output:", e)
            return {}
