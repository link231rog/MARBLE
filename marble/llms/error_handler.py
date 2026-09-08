import os
import random
import time
from functools import wraps

from beartype.typing import Any, Callable, List, Optional, Set, TypeVar, Union, cast
from pydantic import BaseModel

T = TypeVar("T", bound=Callable[..., Union[Optional[List[Any]], Set[str]]])


def _configured_wait_time(env_name: str, fallback: float) -> float:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return fallback
    try:
        wait_time = float(raw_value)
    except (ValueError, TypeError):
        return fallback
    return wait_time if wait_time >= 0 else fallback


def _configured_int(env_name: str, fallback: int) -> int:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return fallback
    try:
        val = int(raw_value)
    except (ValueError, TypeError):
        return fallback
    return val if val >= 0 else fallback


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if str(status_code) == "429":
        return True

    error_name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "ratelimit" in error_name
        or "rate limit" in message
        or "429" in message
        or "too many requests" in message
        or "rpm exhausted" in message
    )


def api_calling_error_exponential_backoff(
    retries: int = 5,
    base_wait_time: int = 1,
    rate_limit_base_wait_time: Optional[float] = None,
) -> Callable[[T], T]:
    """
    Decorator for applying exponential backoff to a function.
    :param retries: Maximum number of retries for general errors.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    :param rate_limit_base_wait_time: Optional base wait for 429 errors. When
        omitted, MARBLE_API_429_BASE_WAIT_TIME is used if set.
    :return: The wrapped function with exponential backoff applied.
    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[List[str]]:
            error_handler_mode = kwargs.get("mode", None)
            if error_handler_mode == "TEST":
                modified_retries = 1
                modified_base_wait_time = 1
                modified_rate_limit_base_wait_time = 1
            else:
                modified_retries = _configured_int("MARBLE_API_RETRIES", retries)
                modified_base_wait_time = base_wait_time
                modified_rate_limit_base_wait_time = rate_limit_base_wait_time
                if modified_rate_limit_base_wait_time is None:
                    modified_rate_limit_base_wait_time = _configured_wait_time(
                        "MARBLE_API_429_BASE_WAIT_TIME", 3.0
                    )

            attempts = 0
            last_exc: Optional[Exception] = None
            max_429_retries = _configured_int("MARBLE_API_429_MAX_RETRIES", 100)
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    is_429 = _is_rate_limit_error(e)
                    current_max = max_429_retries if is_429 else modified_retries
                    if attempts >= current_max:
                        break
                    retry_base_wait_time = (
                        modified_rate_limit_base_wait_time
                        if is_429
                        else modified_base_wait_time
                    )
                    raw_wait = retry_base_wait_time * (2**min(attempts, 6))
                    max_wait = float(os.environ.get("MARBLE_API_MAX_WAIT_TIME", "35.0"))
                    if is_429:
                        # Full jitter desynchronization: break lockstep retry storms across concurrent workers
                        upper = min(raw_wait, max_wait)
                        wait_time = random.uniform(retry_base_wait_time, max(upper, retry_base_wait_time)) + random.uniform(2.0, 8.0)
                        print(f"[429 RateLimit] Attempt {attempts + 1}/{current_max} failed: {e}")
                        print(f"[429 RateLimit] Backing off {wait_time:.1f}s before retrying (must retry, never fail on 429)...")
                    else:
                        wait_time = min(raw_wait, max_wait) + random.uniform(0.5, 2.5)
                        print(f"Attempt {attempts + 1}/{current_max} failed: {e}")
                        print(f"Waiting {wait_time:.1f} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            print(
                f"Failed to execute '{func.__name__}' after {attempts} retries."
            )
            if last_exc is not None:
                raise last_exc
            return None

        return cast(T, wrapper)

    return cast(Callable[[T], T], decorator)


TBaseModel = TypeVar("TBaseModel", bound=Callable[..., BaseModel])


def parsing_error_exponential_backoff(
    retries: int = 5, base_wait_time: int = 1
) -> Callable[[TBaseModel], TBaseModel]:
    """
    Decorator for retrying a function that returns a BaseModel with exponential backoff.
    :param retries: Maximum number of retries.
    :param base_wait_time: Base wait time in seconds for the exponential backoff.
    :return: The wrapped function with retry logic applied.
    """

    def decorator(func: TBaseModel) -> TBaseModel:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Optional[BaseModel]:
            attempts = 0
            while attempts < retries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    wait_time = base_wait_time * (2**attempts)
                    print(f"Attempt {attempts + 1} failed: {e}")
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    attempts += 1
            print(
                f"Failed to get valid input from {func.__name__} after {retries} retries."
            )
            return None

        return cast(TBaseModel, wrapper)

    return cast(Callable[[TBaseModel], TBaseModel], decorator)
