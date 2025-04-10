"""에이전트를 위한 상태 구조를 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """에이전트의 입력 상태를 정의하며, 외부 세계와의 더 좁은 인터페이스를 나타냅니다.

    이 클래스는 초기 상태와 들어오는 데이터의 구조를 정의하는 데 사용됩니다.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    에이전트의 주요 실행 상태를 추적하는 메시지들입니다.

    일반적으로 다음과 같은 패턴으로 누적됩니다:
    1. HumanMessage - 사용자 입력
    2. AIMessage와 .tool_calls - 정보 수집을 위해 에이전트가 사용할 도구를 선택함
    3. ToolMessage(들) - 실행된 도구로부터의 응답(또는 오류)
    4. .tool_calls가 없는 AIMessage - 에이전트가 사용자에게 비구조화된 형식으로 응답
    5. HumanMessage - 사용자가 다음 대화 차례로 응답

    필요에 따라 2-5단계가 반복될 수 있습니다.

    `add_messages` 주석은 새 메시지가 기존 메시지와 병합되도록 보장하며,
    동일한 ID를 가진 메시지가 제공되지 않는 한 ID로 업데이트하여 "추가 전용" 상태를 유지합니다.
    """


@dataclass
class State(InputState):
    """InputState를 추가 속성으로 확장하여 에이전트의 완전한 상태를 나타냅니다.

    이 클래스는 에이전트의 수명 주기 전체에 걸쳐 필요한 모든 정보를 저장하는 데 사용할 수 있습니다.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    현재 단계가 그래프가 오류를 발생시키기 전의 마지막 단계인지 여부를 나타냅니다.

    이는 사용자 코드가 아닌 상태 머신에 의해 제어되는 '관리된' 변수입니다.
    단계 수가 recursion_limit - 1에 도달하면 'True'로 설정됩니다.
    """

    # 필요에 따라 여기에 추가 속성을 추가할 수 있습니다.
    # 일반적인 예시:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
