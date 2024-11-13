"""
title: Unified Pipe for Flowise, using LLM for Chat History Summarization
author: matthewh
version: 1.9
license: MIT
required_open_webui_version: 0.3.32
"""

from typing import Optional, Callable, Awaitable, Dict, Any, List, Union
import aiohttp
import json
import time
import asyncio
from pydantic import BaseModel, Field
import logging
import re

# Import the generate_chat_completions and get_task_model_id functions 
from open_webui.main import (
    generate_chat_completions,
    get_task_model_id,
)  # Ensure this import path is correct based on your project structure

from open_webui.utils.misc import get_last_user_message, pop_system_message


class Pipe:
    """
    Comprehensive Pipeline for orchestrating interactions with Flowise and LLM APIs.

    - Summarizes chat history (excluding the latest user message) using LLM.
    - Appends the latest user message to the summary.
    - Sends the combined prompt to Flowise for processing.
    - Emits periodic status updates, optionally using LLM-generated whimsical messages.
    - Emits the final Flowise response via the event emitter.

    Requirements:
    - Flowise service: https://github.com/FlowiseAI/Flowise

    Note:
    - Ensure Flowise is deployed on the specified endpoint to prevent conflicts.
    """

    class Valves(BaseModel):
        # Group 1: Flowise Connection and Authentication
        FLOWISE_API_ENDPOINT: str = Field(
            default="http://host.docker.internal:3030/",
            description="Base URL for the Flowise API endpoint.",
        )
        FLOWISE_USERNAME: Optional[str] = Field(
            default=None, description="Username for Flowise API authentication."
        )
        FLOWISE_PASSWORD: Optional[str] = Field(
            default=None, description="Password for Flowise API authentication."
        )
        FLOWISE_CHATFLOW_ID: str = Field(
            default="", description="Chatflow ID for the Flowise API."
        )

        # Group 2: Summarization Prompt
        SUMMARIZATION_PROMPT_PREFIX: str = Field(
            default="Summarize this conversation, retaining all specific technical details:",
            description="Prefix that identifies a summarization request.",
        )

        # Group 3: Message History and Prompt Configuration
        MAX_HISTORY: int = Field(
            default=5,
            description="Maximum number of previous messages (user and assistant combined) to include in the prompt sent to Flowise.",
        )

        # Group 4: Status Updates (including dynamic options)
        emit_interval: float = Field(
            default=1.0, description="Interval in seconds between status updates."
        )
        enable_status_updates: bool = Field(
            default=True, description="Enable or disable status updates."
        )
        enable_dynamic_status: bool = Field(
            default=False, description="Enable LLM for generating status messages."
        )
        status_generation_prompt: str = Field(
            default="Whimsically inform the user that their request is underway (in 5 words or less).",
            description="LLM prompt for generating whimsical status messages.",
        )
        status_system_prompt: str = Field(
            default="You are a creative and whimsical assistant.",
            description="LLM system prompt for generating status messages.",
        )
        enable_timer_suffix: bool = Field(
            default=True, description="Add elapsed time to status messages."
        )

        # Group 5: Debug and General Configuration
        request_timeout: int = Field(
            default=300, description="HTTP client timeout in seconds."
        )
        debug: bool = Field(
            default=True, description="Enable or disable debug logging."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.stop_emitter = asyncio.Event()
        self.chat_sessions = {}
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG if self.valves.debug else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.log.handlers:
            self.log.addHandler(handler)
        self.start_time: Optional[float] = None  # Initialize start_time
        self.task_model_id: Optional[str] = None  # To store the selected task model

    def log_debug(self, message: str):
        """Logs a debug message if debugging is enabled."""
        if self.valves.debug:
            self.log.debug(message)

    def clean_response_text(self, text: str) -> str:
        """
        Removes unnecessary surrounding quotes from the response.

        Handles cases where the response text may be wrapped in quotes.
        """
        self.log_debug(f"Original text before cleaning: {text!r}")
        # Remove outer quotes if present using regex
        pattern = r'^([\'"])(.*)\1$'
        match = re.match(pattern, text)
        if match:
            text = match.group(2)
            self.log_debug(f"Text after stripping quotes: {text!r}")
        cleaned_text = text.strip()
        self.log_debug(f"Final cleaned text: {cleaned_text!r}")
        return cleaned_text

    def _get_combined_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Combines user and assistant messages into a structured prompt.

        Example:
            User: Hi!
            Assistant: Hello! How can I help you today?
            User: Tell me a joke.
        """
        prompt_parts = [
            f"{message.get('role', 'user').capitalize()}: {message.get('content', '')}"
            for message in messages
        ]
        combined_prompt = "\n".join(prompt_parts)
        self.log_debug(f"Combined prompt:\n{combined_prompt}")
        return combined_prompt

    async def emit_periodic_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        interval: float,
    ):
        """Emit status updates periodically until the stop event is set."""
        try:
            while not self.stop_emitter.is_set():
                current_time = asyncio.get_event_loop().time()
                elapsed_time = (
                    current_time - self.start_time if self.start_time else 0.0
                )

                # Check if status updates are enabled
                if self.valves.enable_status_updates:
                    if self.valves.enable_dynamic_status:
                        # Generate a new status message using LLM
                        status_message = await self.generate_status_message(
                            task_model_id=self.task_model_id, elapsed_time=elapsed_time
                        )
                    else:
                        # Use a default status message
                        status_message = "Processing request..."

                        # Append elapsed time if enabled
                        if self.valves.enable_timer_suffix:
                            status_message = (
                                f"{status_message} (elapsed: {elapsed_time:.1f}s)"
                            )

                    # Emit the status message
                    await self.emit_status(
                        __event_emitter__,
                        status_message,
                        False,
                    )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self.log_debug("Periodic status emitter cancelled.")

    async def generate_status_message(
        self, task_model_id: str, elapsed_time: float
    ) -> str:
        """Use LLM to generate a whimsical status message."""
        prompt = self.valves.status_generation_prompt
        system_prompt = self.valves.status_system_prompt
        self.log_debug(f"LLM Generation Prompt for Status: {prompt}")
        self.log_debug(f"LLM System Prompt for Status: {system_prompt}")

        # Define the payload for the chat completion
        payload = {
            "model": task_model_id,  # Use the task_model_id instead of hardcoded "gpt-4"
            "messages": [
                {
                    "role": "system",
                    "content": self.valves.status_system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 20,  # Adjust as necessary
            "temperature": 0.8,  # Adjust for creativity
        }

        try:
            # Generate the chat completion
            response = await generate_chat_completions(form_data=payload)
            self.log_debug(f"LLM Response for Status: {response}")

            # Validate response structure
            if (
                "choices" in response
                and len(response["choices"]) > 0
                and "message" in response["choices"][0]
                and "content" in response["choices"][0]["message"]
            ):
                generated_message = response["choices"][0]["message"]["content"].strip()
                self.log_debug(
                    f"Generated Status Message Before Cleanup: {generated_message}"
                )

                # Clean up the message by removing leading/trailing quotes if both exist
                cleaned_message = self.clean_response_text(generated_message)
                self.log_debug(
                    f"Generated Status Message After Cleanup: {cleaned_message}"
                )

                # Append elapsed time if enabled
                if self.valves.enable_timer_suffix:
                    cleaned_message = (
                        f"{cleaned_message} (elapsed: {elapsed_time:.1f}s)"
                    )

                return cleaned_message
            else:
                self.log.error(
                    "Invalid response structure from LLM for status message."
                )
                # Fallback to a default message if response is invalid
                fallback_message = (
                    f"Processing your request... (elapsed: {elapsed_time:.1f}s)"
                )
                self.log_debug(f"Fallback Status Message: {fallback_message}")
                return fallback_message
        except Exception as e:
            self.log.error(f"Error generating status message: {e}")
            # Fallback to a default message in case of error
            fallback_message = (
                f"Processing your request... (elapsed: {elapsed_time:.1f}s)"
            )
            self.log_debug(f"Fallback Status Message after Error: {fallback_message}")
            return fallback_message

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        message: str,
        done: bool,
    ):
        """Emit status updates to the event emitter."""
        if __event_emitter__:
            event = {
                "type": "status",
                "data": {"description": message, "done": done},
            }
            self.log_debug(f"Emitting status event: {event}")
            await __event_emitter__(event)

    async def summarize_chat_history(
        self,
        messages: List[Dict[str, str]],
        __user__: Optional[dict],
    ) -> Union[Dict[str, str], Dict[str, str]]:
        """
        Summarizes the chat history excluding the latest user message using LLM,
        ensuring that specific technical details are retained.
        """
        self.log_debug("Starting chat history summarization using LLM.")

        if not messages:
            self.log_debug("No prior messages to summarize.")
            return {"summary": ""}  # Return an empty summary

        combined_prompt = self._get_combined_prompt(messages)
        summary_prompt = f"{self.valves.SUMMARIZATION_PROMPT_PREFIX}\n{combined_prompt}"
        payload = {
            "model": self.task_model_id,  # Use the task_model_id for summarization
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes conversations.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            "max_tokens": 150,  # Adjust as necessary for summary length
            "temperature": 0.5,  # Lower temperature for more factual summaries
        }

        try:
            # Generate the summary using LLM
            response = await generate_chat_completions(form_data=payload)
            self.log_debug(f"LLM Response for Summarization: {response}")

            # Validate response structure
            if (
                "choices" in response
                and len(response["choices"]) > 0
                and "message" in response["choices"][0]
                and "content" in response["choices"][0]["message"]
            ):
                generated_summary = response["choices"][0]["message"]["content"].strip()
                self.log_debug(f"Generated Summary Before Cleanup: {generated_summary}")

                # Clean up the summary
                cleaned_summary = self.clean_response_text(generated_summary)
                self.log_debug(f"Generated Summary After Cleanup: {cleaned_summary}")

                return {"summary": cleaned_summary}
            else:
                self.log.error("Invalid response structure from LLM for summarization.")
                # Fallback to a default message if response is invalid
                fallback_summary = "Summarization failed due to an invalid response."
                self.log_debug(f"Fallback Summary: {fallback_summary}")
                return {"error": fallback_summary}
        except Exception as e:
            error_message = f"Error during LLM summarization: {e}"
            self.log.error(error_message)
            return {"error": error_message}

    async def handle_flowise_request(
        self,
        question: str,
        __user__: Optional[dict],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
    ) -> Union[Dict[str, Any], Dict[str, str]]:
        """
        Handles regular requests by sending them to Flowise.
        """
        try:
            # Prepare the payload for Flowise
            payload = {"question": question}
            # Include chatId if it exists in the session
            user_id = (
                __user__.get("user_id", "default_user") if __user__ else "default_user"
            )
            chat_session = self.chat_sessions.get(user_id, {})
            chat_id = chat_session.get("chat_id")
            if chat_id:
                payload["chatId"] = chat_id

            self.log_debug(f"Payload for Flowise: {payload}")

            # Send the request to Flowise API
            endpoint = self.valves.FLOWISE_API_ENDPOINT.rstrip("/")
            url = f"{endpoint}/api/v1/prediction/{self.valves.FLOWISE_CHATFLOW_ID}"
            headers = {"Content-Type": "application/json"}

            # Handle authentication if provided
            auth = None
            if self.valves.FLOWISE_USERNAME and self.valves.FLOWISE_PASSWORD:
                auth = aiohttp.BasicAuth(
                    self.valves.FLOWISE_USERNAME, self.valves.FLOWISE_PASSWORD
                )
                self.log_debug("Flowise authentication enabled.")

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.valves.request_timeout),
                auth=auth,
            ) as session:
                self.log_debug(
                    f"Sending request to Flowise at {url} with payload: {payload}"
                )
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    self.log_debug(f"Flowise response status: {response.status}")
                    self.log_debug(f"Flowise response text: {response_text}")

                    if response.status != 200:
                        error_message = f"Error: Flowise API call failed with status {response.status}"
                        self.log_debug(error_message)
                        return {"error": error_message}

                    try:
                        data = json.loads(response_text)
                        self.log_debug(f"Parsed Flowise response data: {data}")
                    except json.JSONDecodeError:
                        error_message = "Error: Invalid JSON response from Flowise."
                        self.log_debug(error_message)
                        return {"error": error_message}

                    raw_text = data.get("text", "")
                    text = self.clean_response_text(raw_text)
                    new_chat_id = data.get("chatId", chat_id)

                    if not text:
                        error_message = "Error: Empty response from Flowise."
                        self.log_debug(error_message)
                        return {"error": error_message}

                    self.log_debug(f"Extracted text from Flowise: {text!r}")
                    self.log_debug(f"New chat ID from Flowise: {new_chat_id}")

                    # Update chat session
                    if user_id not in self.chat_sessions:
                        self.chat_sessions[user_id] = {"chat_id": None, "history": []}
                    self.chat_sessions[user_id]["chat_id"] = new_chat_id

                    # Append to chat history
                    self.chat_sessions[user_id]["history"].append(
                        {"role": "assistant", "content": text}
                    )
                    self.log_debug(
                        f"Updated chat history for user '{user_id}': {self.chat_sessions[user_id]['history']}"
                    )

                    # Emit the Flowise response via the event emitter
                    if __event_emitter__:
                        response_event = {
                            "type": "message",
                            "data": {"content": text},
                        }
                        self.log_debug(
                            f"Emitting Flowise response event: {response_event}"
                        )
                        await __event_emitter__(response_event)

                    return {"response": text}

        except Exception as e:
            error_message = f"Error during Flowise request handling: {e}"
            self.log.error(error_message)
            return {"error": error_message}

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Union[Dict[str, Any], Dict[str, str]]:
        """
        Main pipe method to handle the processing pipeline.

        Steps:
        1. Determine the task_model_id based on the request.
        2. Emit periodic status updates if enabled.
        3. Extract messages from the request body.
        4. Separate the latest user message.
        5. Summarize the chat history using LLM (excluding the latest message).
        6. Append the latest user message to the summary.
        7. Send the combined prompt to Flowise.
        8. Emit final status update upon completion.
        9. Emit the Flowise response via the event emitter.

        Args:
            body (dict): The request body containing chat messages and model info.
            __user__ (Optional[dict]): User information.
            __event_emitter__ (Optional[Callable[[dict], Awaitable[None]]]): Event emitter for status updates.

        Returns:
            Union[Dict[str, Any], Dict[str, str]]: The response from Flowise or an error message.
        """
        status_task = None
        try:
            # Record the start time
            self.start_time = asyncio.get_event_loop().time()

            # Determine the task_model_id using the provided model in the body
            default_model_id = body.get(
                "model", "default-model-id"
            )  # Replace with your actual default
            self.task_model_id = get_task_model_id(default_model_id)
            self.log_debug(f"Selected task_model_id: {self.task_model_id}")

            # Emit periodic status if enabled
            if callable(__event_emitter__) and self.valves.enable_status_updates:
                self.log_debug("Starting periodic status emitter...")
                self.stop_emitter.clear()
                status_task = asyncio.create_task(
                    self.emit_periodic_status(
                        __event_emitter__,
                        self.valves.emit_interval,
                    )
                )
            else:
                self.log_debug(
                    "No valid event emitter provided or status updates disabled. Skipping periodic status."
                )

            # Extract messages from the request body
            messages = body.get("messages", [])
            self.log_debug(f"Messages extracted: {messages}")

            if not messages:
                error_message = "Error: No messages found."
                self.log_debug(error_message)
                if __event_emitter__:
                    await self.emit_status(__event_emitter__, error_message, True)
                return {"error": error_message}

            # Extract the latest user message using get_last_user_message
            latest_user_message = get_last_user_message(messages)
            if not latest_user_message:
                error_message = "Error: No user message found."
                self.log_debug(error_message)
                if __event_emitter__:
                    await self.emit_status(__event_emitter__, error_message, True)
                return {"error": error_message}

            self.log_debug(f"Latest user message: {latest_user_message!r}")

            # Extract prior messages (all except the latest user message)
            prior_messages = self._get_prior_messages(messages, latest_user_message)
            self.log_debug(f"Prior messages for summarization: {prior_messages}")

            # Check if there are prior messages to summarize
            if prior_messages:
                # Summarize the chat history using LLM (excluding the latest message)
                summary_response = await self.summarize_chat_history(
                    prior_messages, __user__
                )
                self.log_debug(f"Summary response from LLM: {summary_response!r}")

                if "error" in summary_response:
                    error_message = summary_response["error"]
                    self.log_debug(f"Summarization error: {error_message}")
                    if __event_emitter__:
                        await self.emit_status(__event_emitter__, error_message, True)
                    return {"error": error_message}

                summary = summary_response.get("summary", "")
                self.log_debug(f"Summary obtained from LLM: {summary!r}")

                # Append the latest user message to the summary
                combined_prompt = f"{summary}\nUser: {latest_user_message}"
                self.log_debug(f"Combined prompt for Flowise:\n{combined_prompt}")
            else:
                self.log_debug(
                    "No prior messages to summarize. Using the latest user message as the prompt."
                )
                combined_prompt = f"User: {latest_user_message}"
                self.log_debug(f"Combined prompt for Flowise:\n{combined_prompt}")

            # Send the combined prompt to Flowise for further processing
            flowise_response = await self.handle_flowise_request(
                combined_prompt, __user__, __event_emitter__
            )

            # Stop the periodic emitter
            if status_task:
                self.stop_emitter.set()
                await status_task
                self.log_debug("Stopped periodic status emitter.")

            # Emit final status
            total_elapsed = asyncio.get_event_loop().time() - self.start_time
            completion_message = "Request processing completed."
            if self.valves.enable_timer_suffix:
                completion_message = (
                    f"{completion_message} (elapsed: {total_elapsed:.1f}s)"
                )
            await self.emit_status(__event_emitter__, completion_message, True)

            return flowise_response

        except Exception as e:
            error_message = f"Error during pipe execution: {e}"
            self.log.error(error_message)
            if __event_emitter__:
                await self.emit_status(__event_emitter__, error_message, True)
            return {"error": str(e)}

        finally:
            if status_task:
                self.stop_emitter.set()
                await status_task
                self.log_debug(
                    "Finalizing pipe execution and ensuring emitter is stopped."
                )

    def _get_prior_messages(
        self, messages: List[Dict[str, str]], latest_message: str
    ) -> List[Dict[str, str]]:
        """
        Retrieves all messages excluding the latest user message.
        """
        self.log_debug("Extracting prior messages excluding the latest user message.")
        prior_messages = []
        found_latest = False
        for message in reversed(messages):
            if (
                message.get("role") == "user"
                and message.get("content", "").strip() == latest_message
            ):
                found_latest = True
                continue
            if found_latest:
                prior_messages.insert(
                    0, message
                )  # Insert at the beginning to maintain order
        self.log_debug(f"Prior messages extracted: {prior_messages}")
        return prior_messages
