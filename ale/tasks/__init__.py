from .base import StepResult, Task
from .code_fix import CodeFix
from .guess_number import GuessNumber
from .text_nav import TextNav
from .tool_math import ToolMath

REGISTRY: dict[str, type[Task]] = {
    cls.name: cls for cls in (GuessNumber, TextNav, CodeFix, ToolMath)
}


def make(name: str, seed: int = 0) -> Task:
    if name not in REGISTRY:
        raise KeyError(f"Unknown task: {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name](seed=seed)


__all__ = ["REGISTRY", "StepResult", "Task", "CodeFix", "GuessNumber", "TextNav", "ToolMath", "make"]
