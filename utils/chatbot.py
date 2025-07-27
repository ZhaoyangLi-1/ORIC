from typing import Optional, Sequence


class DecodingArguments(object):
    # the same as the openai API: https://platform.openai.com/docs/api-reference/chat/create
    # Use OpenAI manner max_tokens mean the max_new_tokens in transformers!
    #  mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
    model: str = None
    frequency_penalty: float = 0.0  # [-2, 2]
    logit_bias: Optional[float] = None  # [-100, 100]
    logprobs: Optional[int] = None  # [0, 5]
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0.0  # [-2, 2]
    seed: Optional[int] = None
    stop: Optional[Sequence[str]] = None  # Up to 4 sequences
    temperature: float = 1.0  # [0, 2]
    top_p: float = 1.0
    stream: bool = False
    tools: Optional[Sequence[str]] = None
    no_repeat_ngram_size: Optional[int] = None
    image_detail: Optional[str] = "low"


class Chatbot:

    def __init__(
        self,
        api_key: str = None,
        decoding_args: DecodingArguments = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        assert self.api_key, (
            "API key must be provided via the api_key argument "
            "or the OPENAI_API_KEY environment variable."
        )
        
        if not self.api_key:
            raise ValueError(
                "API key must be provided via argument or OPENAI_API_KEY environment variable."
            )
        openai.api_key = self.api_key

        if decoding_args is None:
            decoding_args = DecodingArguments()
        self.decoding_args = decoding_args

        self.system_prompt = (
            "You are a helpful, friendly AI assistant. Provide clear, concise answers "
            "and ask follow-up questions only when needed for clarity."
        )

    def send(self, messages: list[dict]) -> str:
        response = openai.ChatCompletion.create(
            model=self.decoding_args.model,
            messages=messages,
            temperature=self.decoding_args.temperature,
            max_tokens=self.decoding_args.max_tokens,
            top_p=self.decoding_args.top_p,
            n=self.decoding_args.n,
            stop=self.decoding_args.stop,
            presence_penalty=self.decoding_args.presence_penalty,
            frequency_penalty=self.decoding_args.frequency_penalty,
        )
        return response.choices[0].message["content"]

    def ask(self, user_prompt: str, system_prompt: str = None) -> str:
        prompt = system_prompt or self.system_prompt
        conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.send(conversation)

if __name__ == "__main__":
    args = DecodingArguments(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=256,
        top_p=1.0,
        n=1,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    bot = Chatbot(decoding_args=args)
    my_prompt = "Write a short poem about spring."
    reply = bot.ask(user_prompt=my_prompt)
    print("Assistant response:")
    print(reply)
