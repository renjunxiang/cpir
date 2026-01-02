"""
规则搜索

审核规则搜索
"""

import json
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import get_runtime
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock

with open("Tools/Search/rules.json", "r", encoding="utf-8") as f:
    rules_all = json.load(f)


class RulesSearchInput(BaseModel):
    """审核规则查询的输入"""
    query: str = Field(description="用户的任务指令", default="")


@tool(
    "审核规则查询",
    description="根据文档，查找适用于文档的消保审核规则。",
    args_schema=RulesSearchInput,
)
def rules_search(query: str, context: RunnableConfig) -> str: #简单的模拟规则检索
    """
    根据文档，查找适用于文档的消保审核规则。
    """

    print(f"\n🔎检索工具=审核规则查询")
    rules = (
        rules_all  # 当规则库很大的时候，要引入检索引擎初筛，不能全部丢个大模型去匹配。
    )

    document = get_runtime(context).context.get("document", "")
    prompt = f"""
    请根据你自己的理解，判断哪些消保审核规则适用于文档。
    任务指令为：{query}。
    文档内容为：{document}。
    规则清单为：{rules}。其中fileName为审核规则的来源，rule为审核规则的内容。
    请结合任务指令，返回你认为适用于文档的审核规则内容，最多不超过5条。
    """
    result=""
    with lock:
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            result += chunk.content

    return result
