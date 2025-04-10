"""사용자 정의 추론 및 행동 에이전트를 정의합니다.

도구 호출 기능을 지원하는 채팅 모델과 함께 작동합니다.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# 모델을 호출하는 함수 정의
async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """우리 "에이전트"를 구동하는 LLM을 호출합니다.

    이 함수는 프롬프트를 준비하고, 모델을 초기화하며, 응답을 처리합니다.

    매개변수:
        state (State): 대화의 현재 상태.
        config (RunnableConfig): 모델 실행을 위한 구성.

    반환:
        dict: 모델의 응답 메시지를 포함하는 사전.
    """
    configuration = Configuration.from_runnable_config(config)

    # 도구 바인딩으로 모델을 초기화합니다. 여기서 모델을 변경하거나 더 많은 도구를 추가할 수 있습니다.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # 시스템 프롬프트를 포맷합니다. 에이전트의 동작을 변경하려면 이곳을 사용자 정의하세요.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # 모델의 응답 가져오기
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # 마지막 단계이고 모델이 여전히 도구를 사용하고자 할 때 처리
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # 모델의 응답을 기존 메시지에 추가될 리스트로 반환
    return {"messages": [response]}