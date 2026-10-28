"""Generation: question + reranked chunks -> answer, with the system prompt and the [generation] settings."""
import logging
from functools import cache
from pathlib import Path

from openai import OpenAI

from hospital_rag.config import settings

logger = logging.getLogger(__name__)


@cache
def get_llm():
    return OpenAI(base_url=settings.llm_url, api_key="EMPTY")  # llama-server started without --api-key


def load_system_prompt(config):
    """The text of the system prompt file. The runner stores it with the experiment: files change, results must not."""
    return Path(config["generation"]["system_prompt"]).read_text(encoding="utf-8").strip()


def format_passages(chunks):
    """Numbered extracts, best first: [1] ... [2] ... The numbers are what citations.txt asks the LLM to cite."""
    return "\n\n".join(f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks, 1))


def build_messages(question, chunks, config):
    """System prompt = how to answer; user message = the extracts, then the question last, closest to the answer."""
    return [
        {"role": "system", "content": load_system_prompt(config)},
        {"role": "user", "content": f"Extraits :\n\n{format_passages(chunks)}\n\nQuestion : {question}"},
    ]


def generate(question, chunks, config):
    """The LLM's answer, with token counts. No seed: at temperature > 0, repeated runs must be able to differ."""
    params = config["generation"]
    response = get_llm().chat.completions.create(
        model=params["model"],
        messages=build_messages(question, chunks, config),
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
        extra_body={"top_k": params["top_k"]},  # llama-server parameter, not part of the OpenAI API
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning("answer cut at max_tokens=%d: %s", params["max_tokens"], question)
    return {
        "answer": (choice.message.content or "").strip(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "finish_reason": choice.finish_reason,
    }
