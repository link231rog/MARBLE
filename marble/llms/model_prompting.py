import os
from collections.abc import Mapping

import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff
from marble.llms.usage import record_successful_completion


_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method in ("model_dump", "to_dict", "dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            dumped = converter()
            if isinstance(dumped, Mapping):
                return dict(dumped)
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def _normalize_zai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for raw_message in messages:
        message = _as_dict(raw_message)
        role = message.get("role")
        role = role if role in _MESSAGE_ROLES else "user"
        clean: Dict[str, Any] = {
            "role": role,
            "content": message.get("content") or "",
        }
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = []
            for raw_call in message["tool_calls"]:
                call = _as_dict(raw_call)
                function = _as_dict(call.get("function"))
                if function.get("name"):
                    tool_calls.append(
                        {
                            "id": str(call.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(function["name"]),
                                "arguments": str(function.get("arguments", "")),
                            },
                        }
                    )
            if tool_calls:
                clean["tool_calls"] = tool_calls
        if role == "tool" and message.get("tool_call_id") is not None:
            clean["tool_call_id"] = str(message["tool_call_id"])
        normalized.append(clean)
    return normalized


def _is_zai_request(llm_model: str, base_url: Optional[str]) -> bool:
    return "z.ai" in (base_url or "").lower() or (
        base_url is None and "glm-" in llm_model.lower()
    )


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Any],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    # litellm.set_verbose=True
    if base_url is None and llm_model.startswith("openai/"):
        base_url = os.environ.get("OPENAI_API_BASE")
    if base_url is None and "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    is_zai = _is_zai_request(llm_model, base_url)
    if is_zai:
        base_url = base_url or _ZAI_BASE_URL
    extra_body = {}
    if "Qwen" in llm_model:
        # ponytail: disable Qwen3 thinking so content is populated and generation is fast
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    effective_reasoning_effort = reasoning_effort or os.environ.get(
        "MARBLE_REASONING_EFFORT"
    )
    if effective_reasoning_effort is not None and not is_zai:
        extra_body["reasoning_effort"] = effective_reasoning_effort
        extra_body["allowed_openai_params"] = ["reasoning_effort"]
    completion_kwargs: Dict[str, Any] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if is_zai:
        api_key = os.environ.get("ZAI_API_KEY") or api_key
    if api_key:
        completion_kwargs["api_key"] = api_key
    input_messages = _normalize_zai_messages(messages) if is_zai else messages
    safe_messages = []
    for msg in input_messages:
        if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
            c = msg["content"]
            if len(c) > 60000:
                msg = dict(msg)
                msg["content"] = c[:20000] + "\n\n... [TRUNCATED DUE TO CONTEXT LIMIT] ...\n\n" + c[-40000:]
        safe_messages.append(msg)

    completion = litellm.completion(
        model=llm_model,
        messages=safe_messages,
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
        **completion_kwargs,
    )
    message_0: Message = completion.choices[0].message
    if message_0 is None:
        message_0 = Message(content="", role="assistant")
    elif message_0.content is None:
        message_0.content = getattr(message_0, "reasoning_content", None) or ""
    assert isinstance(message_0, Message)
    record_successful_completion(completion)
    return [message_0]
