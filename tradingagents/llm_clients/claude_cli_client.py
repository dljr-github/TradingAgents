import json
import os
import re
import subprocess
import uuid
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from .base_client import BaseLLMClient


def _format_tool_schema(tool: BaseTool) -> str:
    """Format a single tool's schema for inclusion in the prompt."""
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    params = []
    for name, info in props.items():
        param_type = info.get("type", "string")
        desc = info.get("description", "")
        req = " (required)" if name in required else ""
        params.append(f"    - {name}: {param_type}{req} — {desc}")

    param_block = "\n".join(params) if params else "    (no parameters)"
    description = tool.description or ""
    return f"- {tool.name}: {description}\n  Parameters:\n{param_block}"


def _build_tools_prompt(tools: Sequence[BaseTool]) -> str:
    """Build the tool instruction block to prepend to the prompt."""
    tool_descriptions = "\n".join(_format_tool_schema(t) for t in tools)
    return f"""You have access to the following tools:

{tool_descriptions}

When you want to call a tool, output EXACTLY this format on its own line:
TOOL_CALL: {{"name": "tool_name", "arguments": {{"param1": "value1"}}}}

For multiple tool calls, use multiple TOOL_CALL: lines:
TOOL_CALL: {{"name": "tool1", "arguments": {{"param1": "value1"}}}}
TOOL_CALL: {{"name": "tool2", "arguments": {{"param2": "value2"}}}}

If you do NOT need to call any tools, respond with plain text as normal (no TOOL_CALL prefix).
IMPORTANT: Do NOT wrap tool call JSON in markdown code blocks. Each TOOL_CALL must be raw JSON on a single line."""


def _extract_tool_calls(text: str) -> tuple[Optional[List[dict]], str]:
    """Extract TOOL_CALL: lines from Claude's response text.

    Returns (tool_calls_list_or_None, remaining_text).
    """
    tool_calls = []
    non_tool_lines = []

    for line in text.split("\n"):
        stripped_line = line.strip()
        if stripped_line.startswith("TOOL_CALL:"):
            json_str = stripped_line[len("TOOL_CALL:"):].strip()
            # Strip markdown code fences if wrapped
            fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", json_str, re.DOTALL)
            if fence_match:
                json_str = fence_match.group(1)
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append(data)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            # Failed to parse — treat as regular text
            non_tool_lines.append(line)
        else:
            non_tool_lines.append(line)

    remaining = "\n".join(non_tool_lines).strip()
    return (tool_calls if tool_calls else None, remaining)


class ClaudeCLIChatModel(BaseChatModel):
    """LangChain ChatModel that shells out to the Claude CLI."""

    cli_path: str = os.path.expanduser("~/.local/bin/claude")
    model_name: Optional[str] = None
    timeout: int = 300
    tools: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ClaudeCLIChatModel":
        """Return a new instance with tools bound."""
        new = self.model_copy()
        new.tools = list(tools)
        return new

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)

        cmd = [self.cli_path, "--print", "--output-format", "text"]
        if self.model_name:
            cmd.extend(["--model", self.model_name])
        cmd.extend(["-p", prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Claude CLI timed out after {self.timeout}s"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Claude CLI not found at {self.cli_path}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI exited with code {result.returncode}: {result.stderr}"
            )

        text = result.stdout.strip()

        if self.tools:
            tool_calls_data, remaining_text = _extract_tool_calls(text)
            if tool_calls_data:
                tool_calls = []
                for tc in tool_calls_data:
                    tool_calls.append({
                        "id": str(uuid.uuid4()),
                        "name": tc["name"],
                        "args": tc.get("arguments", {}),
                    })
                msg = AIMessage(content=remaining_text, tool_calls=tool_calls)
                return ChatResult(generations=[ChatGeneration(message=msg)])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        parts = []

        if self.tools:
            parts.append(_build_tools_prompt(self.tools))

        for msg in messages:
            role = msg.type
            if role == "human":
                parts.append(f"Human: {msg.content}")
            elif role == "ai":
                if msg.content:
                    parts.append(f"Assistant: {msg.content}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tc_lines = []
                    for tc in msg.tool_calls:
                        tc_json = json.dumps({"name": tc["name"], "arguments": tc["args"]})
                        tc_lines.append(f"TOOL_CALL: {tc_json}")
                    parts.append("Assistant: " + "\n".join(tc_lines))
            elif role == "tool":
                parts.append(f"Tool ({msg.name}): {msg.content}")
            elif role == "system":
                parts.append(f"System: {msg.content}")
            else:
                parts.append(f"{role}: {msg.content}")
        return "\n\n".join(parts)


class ClaudeCLIClient(BaseLLMClient):
    """Client that uses the Claude CLI instead of an API key."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ClaudeCLIChatModel instance."""
        cli_path = self.kwargs.get(
            "claude_cli_path",
            os.path.expanduser("~/.local/bin/claude"),
        )
        timeout = self.kwargs.get("claude_cli_timeout", 300)

        return ClaudeCLIChatModel(
            cli_path=cli_path,
            model_name=self.model,
            timeout=timeout,
        )

    def validate_model(self) -> bool:
        """Claude CLI uses whatever model the user has configured; always valid."""
        return True
