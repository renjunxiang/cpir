"""
创造类（将要素组成新整体或重组为新模型 / 体系）

任务分解
"""

from typing_extensions import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import get_runtime
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock


class DecomposeInput(BaseModel):
    query: str = Field(description="任务指令")
    messages: Annotated[list, InjectedState("messages")] = Field(description="历史上下文")


@tool(
    "任务分解",
    description="如果当前任务过于复杂，需要拆解为若干个相对简单的子任务。",
    args_schema=DecomposeInput,
)
def decompose(query: str, messages: Annotated[list, InjectedState("messages")], context: RunnableConfig) -> str:
    """
    如果当前任务过于复杂，需要拆解为若干个相对简单的子任务。
    """
    prompt = f"""
    你是一个任务分解模块，你的职责是将当前任务指令拆解为若干个相对简单的子任务，拆解的子任务数不宜超过5个。
    文档是：{get_runtime(context).context.get("document", "")}
    任务指令是：{query}。
    上下文是：{messages}
    输出格式为：子任务1, 子任务2, ...
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=任务分解")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=任务分解\n{response}")

    return response
