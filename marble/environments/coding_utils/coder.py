import os
import re
from typing import Any, Dict

from ruamel.yaml import YAML

from marble.llms.model_prompting import model_prompting


def resolve_coding_task(env, task_description: str = "", model_name: str = "") -> tuple[str, str, str]:
    """Dynamically resolve worker model name, full task description, and implementation requirements."""
    configured_model = None
    if hasattr(env, "config") and isinstance(env.config, dict):
        configured_model = env.config.get("llm")
    if not configured_model:
        configured_model = os.environ.get("MARBLE_WORKER_MODEL")
    if not model_name or model_name in ("gpt-3.5-turbo", "default"):
        model_name = configured_model or "openai/nvidia/nemotron-3-super-120b-a12b"

    full_task = None
    if hasattr(env, "config") and isinstance(env.config, dict):
        task_node = env.config.get("task")
        if isinstance(task_node, dict) and "content" in task_node:
            full_task = task_node["content"]
        elif "task_content" in env.config:
            full_task = env.config["task_content"]
    if not full_task and task_description and task_description.strip():
        full_task = task_description

    if not full_task:
        config_path = "marble/configs/coding_config/coding_config.yaml"
        if os.path.exists(config_path):
            yaml = YAML()
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.load(f)
            full_task = config.get("task", {}).get("content", "")
        else:
            full_task = ""

    req_start = "1. Implementation requirements:\n"
    req_end = "\n\n2. Project structure:"
    if full_task and req_start in full_task and req_end in full_task:
        start_idx = full_task.find(req_start) + len(req_start)
        end_idx = full_task.find(req_end)
        reqs = full_task[start_idx:end_idx].strip()
    else:
        reqs = full_task

    return model_name, full_task, reqs


def create_solution_handler(
    env, task_description: str = "", model_name: str = "", file_path: str = "solution.py", **kwargs
) -> Dict[str, Any]:
    """
    Creates solution.py file and generates content based on task description.

    The generated code will include inline comments explaining the file,
    and the final output will be enclosed in a Markdown code block with the language
    specified as python. Only the code within the code block (without the markdown markers)
    will be stored in the file.

    Args:
        env: The environment instance.
        task_description (str): Task description.
        model_name (str): Name of the LLM model to use.
        file_path (str): File path, defaults to solution.py.

    Returns:
        Dict[str, Any]: Result of the operation.
    """
    try:
        file_path = "solution.py"
        full_path = os.path.join(env.workspace_dir, file_path)

        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass

        model_name, full_task_description, requirements = resolve_coding_task(env, task_description, model_name)
        if not full_task_description:
            return {
                "success": False,
                "error-msg": "Config file not found or task description is empty",
            }

        os.makedirs(env.workspace_dir, exist_ok=True)

        system_prompt = (
            "You are a Python developer. Create a solution based on the following task description.\n"
            "Your code should be clean, well-documented, and follow Python best practices.\n"
            "Include explanations of the code and its functionality as inline comments within the code.\n"
            "Your final output must be enclosed in a markdown code block with the language specified as python.\n"
            "Ensure that nothing besides the code is inside the markdown code block.\n"
            f"Task Description:\n{full_task_description}\n\n"
            f"Implementation Requirements:\n{requirements}\n"
        )

        user_prompt = "Please write the complete Python code for this task."

        response = model_prompting(
            model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            return_num=1,
            max_token_num=8192,
            temperature=0.0,
        )[0]

        code_content = response.content.strip()

        code_block_match = re.search(r"```(?:python)?\s*(.*?)(?:```|$)", code_content, re.DOTALL)
        if code_block_match and code_block_match.group(1).strip():
            code_content = code_block_match.group(1).strip()
        else:
            code_content = re.sub(r"^```(?:python)?\s*", "", code_content)
            code_content = re.sub(r"\s*```$", "", code_content).strip()

        with open(full_path, "w") as file:
            file.write(code_content)

        return {
            "success": True,
            "message": f"Solution file created at {full_path}",
            "code": code_content,
        }

    except Exception as e:
        return {"success": False, "error-msg": str(e)}


#
# def revise_solution_handler(env, task_description: str, model_name: str, file_path: str = "solution.py") -> Dict[str, Any]:
#     """
#     Reads solution.py content and improves/modifies it based on task description.
#     If advices.json exists, incorporates the suggestions into the improvement process.
#
#     Args:
#         env: The environment instance.
#         task_description (str): Task description.
#         model_name (str): Name of the LLM model to use.
#         file_path (str): File path, defaults to solution.py.
#
#     Returns:
#         Dict[str, Any]: Result of the operation.
#     """
#     try:
#         full_path = os.path.join(env.workspace_dir, os.path.basename(file_path))
#         advice_path = os.path.join(env.workspace_dir, "advices.json")
#
#         # Create workspace directory if it doesn't exist
#         os.makedirs(env.workspace_dir, exist_ok=True)
#
#         # Create file if it doesn't exist
#         if not os.path.exists(full_path):
#             return create_solution_handler(env, task_description, model_name, file_path)
#
#         # Read existing code
#         with open(full_path, 'r') as file:
#             existing_code = file.read()
#
#         # Try to load suggestions from advices.json if it exists
#         suggestions = ""
#         if os.path.exists(advice_path):
#             try:
#                 with open(advice_path, 'r') as f:
#                     advice_data = json.load(f)
#                     if isinstance(advice_data, list) and len(advice_data) > 0:
#                         suggestions = advice_data[0].get("suggestions", "")
#             except (json.JSONDecodeError, KeyError):
#                 suggestions = ""
#
#         # Construct system prompt with suggestions if available
#         system_prompt = (
#             "You are a Python developer. Review and improve the existing code based on the task description.\n"
#             "Your improvements should maintain code clarity and follow Python best practices.\n"
#             "Include explanations of your modifications as inline comments within the code.\n"
#             "Your final output must be enclosed in a markdown code block with the language specified as python.\n"
#             "Ensure that only the code is within the code block.\n"
#             "At the very end of your code, include the following exact conclusion as a comment:\n"
#             "# The task description is: [repeat the full task description here]. Based on this task description, I have improved the solution.\n\n"
#             f"Task Description:\n{task_description}\n"
#             "\nExisting Code:\n"
#             f"{existing_code}\n"
#         )
#
#         if suggestions:
#             system_prompt += (
#                 "\nPrevious Code Review Suggestions:\n"
#                 f"{suggestions}\n"
#                 "\nPlease consider these suggestions while improving the code.\n"
#             )
#
#         user_prompt = "Please provide the improved version of this code, taking into account any previous suggestions if provided."
#
#         response = model_prompting(
#             model_name,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             return_num=1,
#             max_token_num=4096,
#             temperature=0.0
#         )[0]
#
#         improved_code = response.content
#
#         # 提取 ```python ... ``` 内的代码
#         code_block_match = re.search(r"```python(.*?)```", improved_code, re.DOTALL)
#         if code_block_match:
#             improved_code = code_block_match.group(1).strip()
#         else:
#             improved_code = improved_code.strip()
#
#         # 更新文件内容
#         with open(full_path, 'w') as file:
#             file.write(improved_code)
#
#         return {
#             "success": True,
#             "message": f"Solution file revised at {full_path}",
#             "original_code": existing_code,
#             "improved_code": improved_code,
#             "previous_suggestions": suggestions if suggestions else "No previous suggestions found"
#         }
#
#     except Exception as e:
#         return {"success": False, "error-msg": str(e)}


def register_coder_actions(env):
    """
    Register coding-related actions in the environment.
    """
    default_model = "openai/nvidia/nemotron-3-super-120b-a12b"
    if hasattr(env, "config") and isinstance(env.config, dict) and env.config.get("llm"):
        default_model = env.config["llm"]
    elif os.environ.get("MARBLE_WORKER_MODEL"):
        default_model = os.environ["MARBLE_WORKER_MODEL"]

    desc_solution = {
        "type": "function",
        "function": {
            "name": "create_solution",
            "description": "Creates solution.py file and generates content based on task description",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task (will be read from config file)",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Name of the LLM model to use",
                        "default": default_model,
                    },
                },
                "required": ["task_description"],
                "additionalProperties": False,
            },
        },
    }

    env.register_action(
        "create_solution",
        handler=lambda **kwargs: create_solution_handler(env, **kwargs),
        description=desc_solution,
    )

    desc_code = {
        "type": "function",
        "function": {
            "name": "create_code",
            "description": "Creates solution.py file and generates content based on task description (alias for create_solution)",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task (will be read from config file)",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Name of the LLM model to use",
                        "default": default_model,
                    },
                },
                "required": ["task_description"],
                "additionalProperties": False,
            },
        },
    }

    env.register_action(
        "create_code",
        handler=lambda **kwargs: create_solution_handler(env, **kwargs),
        description=desc_code,
    )

    # 如果需要，也可以类似地注册 revise_solution 动作（目前该函数为注释状态）
    # env.register_action(
    #     "revise_solution",
    #     handler=lambda **kwargs: revise_solution_handler(env, **kwargs),
    #     description={
    #         "type": "function",
    #         "function": {
    #             "name": "revise_solution",
    #             "description": "Revise existing solution file by improving/modifying code based on task description",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "task_description": {
    #                         "type": "string",
    #                         "description": "Description of the task to implement"
    #                     },
    #                     "model_name": {
    #                         "type": "string",
    #                         "description": "Name of the LLM model to use"
    #                     },
    #                     "file_path": {
    #                         "type": "string",
    #                         "description": "Path of the solution file to revise (optional, defaults to 'solution.py')"
    #                     }
    #                 },
    #                 "required": ["task_description", "model_name"],
    #                 "additionalProperties": False
    #             }
    #         }
    #     }
    # )
