from __future__ import annotations

import time
from collections import deque
from threading import Condition
from typing import Any


class MockCaptureBackend:
    """Deterministisches Capture-Backend ohne angeschlossene Hardware."""

    def __init__(
        self,
        unique_id: str,
        *,
        fail_on_open: bool = False,
    ) -> None:
        self.unique_id = unique_id
        self.fail_on_open = fail_on_open
        self._is_open = False
        self._frames: deque[Any] = deque()
        self._condition = Condition()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if self.fail_on_open:
            raise RuntimeError(
                f"Mock-Backend {self.unique_id} konnte nicht geöffnet werden."
            )

        with self._condition:
            self._is_open = True
            self._condition.notify_all()

    def push_frame(self, frame: Any) -> None:
        with self._condition:
            self._frames.append(frame)
            self._condition.notify_all()

    def read(self, timeout: float | None = None) -> tuple[bool, Any]:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout darf nicht negativ sein.")

        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            if not self._is_open:
                return False, None

            while not self._frames:
                if timeout == 0:
                    return False, None

                remaining = (
                    None
                    if deadline is None
                    else deadline - time.monotonic()
                )

                if remaining is not None and remaining <= 0:
                    return False, None

                self._condition.wait(timeout=remaining)

                if not self._is_open:
                    return False, None

            return True, self._frames.popleft()

    def release(self) -> None:
        with self._condition:
            self._is_open = False
            self._condition.notify_all()
