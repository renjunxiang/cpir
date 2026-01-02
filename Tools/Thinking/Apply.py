"""
应用类（在给定的情景中执行或使用程序）

直接作答
"""

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock


class AnswerInput(BaseModel):
    """问题的输入"""

    query: str = Field(description="用户指令")


@tool(
    "直接作答",
    description="当任务的上下文信息足够时，利用大模型本身的能力进行作答，并结束任务。",
    args_schema=AnswerInput,
)
def direct_answer(query: str) -> str:
    """
    当任务的上下文信息足够时，利用大模型本身的能力进行作答，并结束任务。
    """
    prompt = f"""
    根据任务指令，直接回答问题。
    任务指令是：{query}。
    输出任务答案。
    """
    with lock:
        # 流式输出
        chunks  = []
        print(f"\n💡认知工具=直接作答")
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        response = "".join(chunks)

        # # 完整输出
        # response = llm.invoke(prompt).content
        # print(f"\n💡认知工具=直接作答\n{response}")

    return response


class RecognizingInput(BaseModel):
    query: str = Field(description="任务指令")
