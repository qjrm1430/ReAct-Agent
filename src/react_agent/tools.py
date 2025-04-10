"""이 모듈은 웹 스크래핑 및 검색 기능을 위한 예제 도구를 제공합니다.

기본 Tavily 검색 함수(예제로)를 포함합니다.

이러한 도구들은 시작하기 위한 무료 예제로 제공됩니다. 운영 환경에서는
다양한 요구사항에 맞게 더 강력하고 특화된 도구를 구현하는 것을 고려하세요.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """
    일반 웹 결과를 검색합니다.

    이 함수는 Tavily 검색 엔진을 사용하여 검색을 수행합니다. 이 엔진은 포괄적이고 정확하며 신뢰할 수 있는 결과를 제공하도록 설계되었습니다. 특히 현재 이벤트에 대한 질문에 대답하는 데 유용합니다.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search]
