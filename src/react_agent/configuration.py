"""에이전트의 구성 가능한 매개변수를 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """에이전트의 구성입니다."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "에이전트의 상호작용에 사용할 시스템 프롬프트. "
            "이 프롬프트는 에이전트의 문맥과 동작을 설정합니다."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "에이전트의 주요 상호작용에 사용할 언어 모델의 이름. "
            "provider/model-name 형식이어야 합니다."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "각 검색 쿼리에 대해 반환할 최대 검색 결과 수."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """RunnableConfig 객체에서 Configuration 인스턴스를 생성합니다."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
