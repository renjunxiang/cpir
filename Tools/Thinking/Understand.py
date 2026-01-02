"""
理解类（从口头、书面和图像等交流形式的教学信息中建构意义）

任务解释
任务举例
"""

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock


class InterpretingInput(BaseModel):
    query: str = Field(description="任务指令")


@tool(
    "任务解释",
    description="如果任务指令比较抽象，解释成具体的任务内容",
    args_schema=InterpretingInput,
)
def interpreting(query: str) -> str:
    """
    理解任务指令，并给出自己的理解。
    """
    prompt = f"""
    你是一个任务解释模块，你的任务是把相对抽象的任务指令，解释成具体的任务内容。
    任务指令是：{query}。
    输出格式为：解释内容
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=任务解释")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=任务解释\n{response}")

    return response


class ExemplifyingInput(BaseModel):
    query: str = Field(description="任务指令")


@tool(
    "任务举例",
    description="如果当前任务指令内包含抽象的概念，需要给出相似的任务样例来帮助理解。",
    args_schema=ExemplifyingInput,
)
def exemplifying(query: str) -> str:
    """
    针对任务指令，给出相似的任务样例。
    """
    prompt = f"""
    你是一个任务举例模块，你的任务是针对任务指令内包含的抽象概念，给出相似的任务样例。
    任务指令是：{query}。
    输出格式为：样例1, 样例2, ...
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=任务举例")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=任务举例\n{response}")

    return response
