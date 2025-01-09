"""
title: Llama sdsds 
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: flowise
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from flowise import Flowise, PredictionData
from types import SimpleNamespace
import json
import time

class IMessage(BaseModel):
    role: str
    content: str

class Pipeline:
    class Valves(BaseModel):
        USE_PERMISSIVE_SAFETY: bool = False  # Example implementation

    def __init__(self):
        self.name = "Wikipedia Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe: {__name__}")
        start = time.time()

        processed_messages = []

        for message in messages:
            processed_content = (
                [item["text"] for item in message["content"] if item["type"] == "text"]
                if isinstance(message.get("content"), list)
                else message.get("content", "")
            )

            print(f"Message processed time: {time.time() - start:.2f} seconds")

            processed_messages.append(
                SimpleNamespace(
                    role="userMessage" if message["role"] == "user" else "apiMessage",
                    content=processed_content,
                )
            )

        if body.get("title"):
            print("Title Generation")
            return "Wikipedia Pipeline"

        client = Flowise(base_url="http://flowise:3000")

        completion = client.create_prediction(
            PredictionData(
                chatflowId="0e4eb362-1ef8-4e14-9bd2-410ae7b14ddd",
                question=user_message,
                history=processed_messages,
                streaming=True,
            )
        )

        for chunk in completion:
            parsed_chunk = json.loads(chunk)
            if parsed_chunk["event"] == "token" and parsed_chunk["data"]:
                yield str(parsed_chunk["data"])
