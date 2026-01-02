"""
记忆 / 回忆类（从长时记忆中提取相关的知识）

要素识别
上下文回忆



"""

from typing_extensions import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock


class RecognizingInput(BaseModel):
    query: str = Field(description="任务指令")


@tool(
    "要素识别",
    description="当任务指令中存在影响任务完成的关键信息要素（如时间、主体、规则、约束条件）时，需要通过识别工具从指令中提取出这些核心要素信息。",
    args_schema=RecognizingInput,
)
def recognizing(query: str) -> str:
    """
    任务指令中存在影响任务完成的关键信息要素（如时间、主体、规则、约束条件）时，需要通过识别工具从指令中提取出这些核心要素信息。
    """
    prompt = f"""
    你是一个知识要素识别模块，你的任务是从任务指令中提取出任务的关键知识要素。
    任务指令是：{query}。
    输出格式为：[要素1, 要素2, ...]
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=要素识别")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=要素识别\n{response}")

    return response


class RecallingInput(BaseModel):
    query: str = Field(description="任务指令")
    messages: Annotated[list, InjectedState("messages")] = Field(description="历史上下文")


@tool(
    "上下文回忆",
    description="当任务需要调用长时记忆来推进时，需要通过回忆工具从历史上下文调取与当前任务相关的内容。",
    args_schema=RecallingInput,
)
def recalling(query: str, messages: Annotated[list, InjectedState("messages")]) -> str:
    """
    当任务需要调用长时记忆来推进时，需要通过回忆工具从历史上下文调取与当前任务相关的内容。
    """
    prompt = f"""
    你是一个上下文回忆模块，你的任务是从历史上下文中，调取与当前任务相关的内容。
    任务指令是：{query}。
    上下文是：{messages}
    输出格式为：信息1, 信息2, ...
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=上下文回忆")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=上下文回忆\n{response}")

    return response
