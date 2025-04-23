from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for Large Language Model clients."""

    # No abstract __init__ - let concrete classes handle initialization
    # and ensure they define self.messages

    @abstractmethod
    def get_models(self):
        pass

    @abstractmethod
    def add_message(self, role: str, content: str):
        """Adds a single message to the conversation history."""
        pass

    @abstractmethod
    def add_messages(self, messages: list):
        """Adds multiple messages to the conversation history."""
        pass

    @abstractmethod
    def get_messages(self) -> list:
        """Returns the current conversation history."""
        pass

    @abstractmethod
    def reset_messages(self):
        """Clears the conversation history."""
        pass

    @abstractmethod
    def generate_text(self, model_name: str, **kwargs) -> str:
        """Generates text using the specified model and parameters."""
        pass

    @abstractmethod
    def prompt_template(self, role: str, prompt: str):
        """Formats a prompt into the message structure expected by the specific API."""
        pass
