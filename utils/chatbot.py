from typing import Optional, Sequence
from dataclasses import dataclass
from openai import OpenAI
import os

@dataclass
class DecodingArguments(object):
    # the same as the openai API: https://platform.openai.com/docs/api-reference/chat/create
    # Use OpenAI manner max_tokens mean the max_new_tokens in transformers!
    #  mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
    model: str = "gpt-4"
    frequency_penalty: float = 0.0
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 512
    n: int = 1
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Sequence[str]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    tools: Optional[Sequence[str]] = None
    no_repeat_ngram_size: Optional[int] = None
    image_detail: Optional[str] = "low"


class Chatbot:
    def __init__(self, api_key: str = None, decoding_args: DecodingArguments = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = self.client
        self.model_name = decoding_args.model if decoding_args else "gpt-4"

        self.decoding_args = decoding_args or DecodingArguments()
        self.system_prompt = (
            "You are a helpful, friendly AI assistant."
        )

    def _call_gpt_5_text_only(self, prompt: str):
        resp = self.model.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            max_output_tokens=self.decoding_args.max_tokens,
        )
        return resp.output_text 

    def send(self, messages: list[dict]) -> str:
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]

        return self._call_gpt_5_text_only(user_prompt)

    def ask(self, user_prompt: str, system_prompt: str = None) -> str:
        prompt = system_prompt or self.system_prompt
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.send(conversation)
