from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.nodes import call_model
from react_agent.state import InputState, State
from react_agent.tools import TOOLS

# 새 그래프 정의

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# 순환할 두 개의 노드 정의
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# 진입점을 `call_model`로 설정
# 이는 이 노드가 가장 먼저 호출된다는 것을 의미합니다
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """모델의 출력을 기반으로 다음 노드를 결정합니다.

    이 함수는 모델의 마지막 메시지가 도구 호출을 포함하는지 확인합니다.

    매개변수:
        state (State): 대화의 현재 상태.

    반환:
        str: 호출할 다음 노드의 이름("__end__" 또는 "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # 도구 호출이 없으면 완료합니다
    if not last_message.tool_calls:
        return "__end__"
    # 그렇지 않으면 요청된 작업을 실행합니다
    return "tools"


# `call_model` 후의 다음 단계를 결정하기 위한 조건부 에지 추가
builder.add_conditional_edges(
    "call_model",
    # call_model 실행이 완료된 후, 다음 노드(들)는 
    # route_model_output의 출력을 기반으로 예약됩니다
    route_model_output,
)

# `tools`에서 `call_model`로 일반 에지 추가
# 이는 순환을 생성합니다: 도구를 사용한 후에는 항상 모델로 돌아갑니다
builder.add_edge("tools", "call_model")

# 빌더를 실행 가능한 그래프로 컴파일
# 상태 업데이트를 위한 중단 포인트를 추가하여 사용자 정의할 수 있습니다
graph = builder.compile(
    interrupt_before=[],  # 호출되기 전에 상태를 업데이트하기 위해 여기에 노드 이름 추가
    interrupt_after=[],  # 호출된 후에 상태를 업데이트하기 위해 여기에 노드 이름 추가
)
graph.name = "ReAct Agent"  # 이것은 LangSmith에서의 이름을 사용자 정의합니다
