from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[Document]: ...
