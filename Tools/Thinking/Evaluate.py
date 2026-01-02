"""
评价类（基于准则和标准作出判断）

信息检查
方案评论
"""
from typing_extensions import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock


class CheckingInput(BaseModel):
    query: str = Field(description="任务指令")
    messages: Annotated[list, InjectedState("messages")] = Field(description="历史上下文")


@tool(
    "信息检查",
    description="当任务需要验证结论 / 结果是否符合既定准则、数据或事实（如核对计算结果准确性、校验结论与证据的匹配度）时，需要通过检查工具完成一致性验证。",
    args_schema=CheckingInput,
)
def checking(query: str, messages: list) -> str:
    """
    检查历史上下文中的执行过程是否符合既定准则、数据或事实。
    """
    prompt = f"""
    你是一个信息检查模块，你的任务是检查历史上下文中的执行过程，是否符合既定准则、数据或事实。
    任务指令是：{query}。
    历史上下文：{messages}
    输出格式为：检查结果
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=信息检查")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=信息检查\n{response}")

    return response


class CritiquingInput(BaseModel):
    query: str = Field(description="任务指令")
    messages: Annotated[list, InjectedState("messages")] = Field(description="历史上下文")


@tool(
    "方案评论",
    description="当任务需要基于既定标准评判方案 / 方法的优劣、可行性或合理性（如对比解决问题的两种思路、评估方案的落地价值）时，需要通过评论工具给出评判结论与依据。",
    args_schema=CritiquingInput,
)
def critiquing(query: str, messages: list) -> str:
    """
    针对任务指令，给出相似的任务样例。
    """
    prompt = f"""
    你是一个方案评论模块，你需要根据任务指令，对比上下文已经设计的方案，给出给出评判结论与依据。
    任务指令是：{query}。
    输出格式为：方案1怎么样, 方案2怎么样...
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=方案评论")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=方案评论\n{response}")

    return response
