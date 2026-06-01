import uuid
from typing import Dict, Optional


class EntityMapper:
    """One-shot entity mapper scoped to a single document.

    The mapping is created and discarded per document — no persistent
    state survives across calls, preventing cross-document reidentification.
    """

    def __init__(self) -> None:
        self._mapping: Dict[str, str] = {}
        self._salt: str = uuid.uuid4().hex[:8]

    def map(self, original: str) -> str:
        if original not in self._mapping:
            anon = f"ENTITY-{self._salt}-{len(self._mapping) + 1}"
            self._mapping[original] = anon
        return self._mapping[original]

    def reset(self) -> None:
        self._mapping.clear()
        self._salt = uuid.uuid4().hex[:8]
