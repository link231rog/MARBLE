import os

import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff
from marble.llms.usage import record_successful_completion


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    # litellm.set_verbose=True
    if "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    else:
        base_url = None
    extra_body = {}
    if "Qwen" in llm_model:
        # ponytail: disable Qwen3 thinking so content is populated and generation is fast
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    effective_reasoning_effort = reasoning_effort or os.environ.get(
        "MARBLE_REASONING_EFFORT"
    )
    if effective_reasoning_effort is not None:
        extra_body["reasoning_effort"] = effective_reasoning_effort
        extra_body["allowed_openai_params"] = ["reasoning_effort"]
    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
        timeout=300,
        **extra_body,
    )
    message_0: Message = completion.choices[0].message
    if message_0 is None:
        message_0 = Message(content="", role="assistant")
    elif message_0.content is None:
        message_0.content = getattr(message_0, "reasoning_content", None) or ""
    assert isinstance(message_0, Message)
    record_successful_completion(completion)
    return [message_0]
