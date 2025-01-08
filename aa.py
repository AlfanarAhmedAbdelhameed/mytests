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
from schemas import OpenAIChatMessage
from flowise import Flowise, PredictionData
import requests
import os
import json


class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "wiki_pipeline"
        self.name = "Wikipedia Pipeline"

        #self.API_URL = "http://flowise:3000/api/v1/prediction/0e4eb362-1ef8-4e14-9bd2-410ae7b14ddd"

        # Initialize rate limits
        self.valves = self.Valves(**{"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")})

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass



    
    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        processed_messages = []

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"role": "userMessage", "content": item["text"]})
            else:
                processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        

        if body.get("title", False):
            print("Title Generation")
            return "Wikipedia Pipeline"
        else:

            #return "Wikipedia Pipeline2222222222222"
            print(body["user"]["email"])
            print(body)
            print(messages)
            #test_streaming()
            client = Flowise(base_url="http://flowise:3000")

            # Test streaming prediction
            completion = client.create_prediction(
                PredictionData(
                    chatflowId="0e4eb362-1ef8-4e14-9bd2-410ae7b14ddd",
                    question= user_message,
                    history = processed_messages
                    #chatId="ss",                    
                    streaming=True
                )
            )

            # Process and print each streamed chunk
            print("Streaming response:")
            print(str(self))
            for chunk in completion:
                # {event: "token", data: "hello"}
                parsed_chunk = json.loads(chunk)
                if (parsed_chunk['event'] == 'token' and parsed_chunk['data'] != ''):
                    #print(str(parsed_chunk['data']))
                    yield str(parsed_chunk['data'])
                        
            
        return ""
