"""
术语搜索

保险术语定义查询
消保术语定义查询
"""

import json
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import get_runtime
from pydantic import BaseModel, Field
from Config.LLM_Client import llm, lock

with open("Tools/Search/insurance_terms.json", "r", encoding="utf-8") as f:
    insurance_terms = json.load(f)


class InsuranceSearchInput(BaseModel):
    """保险术语查询的输入"""

    terms: list = Field(description="保险术语列表", default=[])


@tool(
    "保险术语查询",
    description="输入保险术语列表，输出每个术语的定义。",
    args_schema=InsuranceSearchInput,
)
def insurance_terms_search(terms: list, context: RunnableConfig) -> str:
    """
    输入保险术语列表，输出每个术语的定义。
    """

    print(f"\n🔎检索工具=保险术语查询")

    result = ""
    words = []
    with lock:
        # 检索工具调用中识别到的保险术语
        # print("\n从任务指令中检索术语")
        for term in terms:
            if term in insurance_terms:
                one_term = f"{term}：{insurance_terms[term]}。"
                print(one_term)
                result += one_term
            else:
                words.append(term)
                # print(f"{term}：未找到定义")

        # 检索待审核文档是否存在关键词
        # print("\n从文档中检索术语")
        document = get_runtime(context).context.get("document", "")
        if document != "":
            for term in insurance_terms:
                if term in document and term not in result:
                    one_term = f"{term}：{insurance_terms[term]}。"
                    print(one_term)
                    result += one_term

        # 从LLM中获取术语定义
        print("\n知识库未查到的术语，LLM自己的定义如下：")
        prompt = f"""
        请根据你自己的理解，给出这些术语的定义，输出形式为：术语1：术语1的定义。术语2：术语2的定义...
        术语清单为：{words}
        """
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            result += chunk.content

    return result


with open("Tools/Search/consumer_protection_terms.json", "r", encoding="utf-8") as f:
    consumer_protection_terms = json.load(f)


class ConsumerProtectionSearchInput(BaseModel):
    """消保术语查询的输入"""

    terms: list = Field(description="消保术语列表", default=[])


@tool(
    "消保术语查询",
    description="输入消保术语列表，输出每个术语的定义。",
    args_schema=ConsumerProtectionSearchInput,
)
def consumer_protection_terms_search(terms: list, context: RunnableConfig) -> dict:
    """
    输入消保术语列表，输出每个术语的定义。
    """

    print(f"\n🔎检索工具=消保术语查询")

    result = ""
    words = []
    with lock:
        # 检索工具调用中识别到的消保术语
        print("\n从任务指令中检索术语")
        for term in terms:
            if term in consumer_protection_terms:
                one_term = f"{term}：{consumer_protection_terms[term]}。"
                print(one_term)
                result += one_term
            else:
                words.append(term)
                # print(f"{term}：未找到定义")

        # 检索待审核文档是否存在关键词
        print("\n从文档中检索术语")
        document = get_runtime(context).context.get("document", "")
        if document != "":
            for term in consumer_protection_terms:
                if term in document and term not in result:
                    one_term = f"{term}：{consumer_protection_terms[term]}。"
                    print(one_term)
                    result += one_term

        # 从LLM中获取术语定义
        print("\n知识库未查到的术语，LLM自己的定义如下：")
        prompt = f"""
        请根据你自己的理解，给出这些术语的定义，输出形式为：术语1：术语1的定义。术语2：术语2的定义...
        术语清单为：{words}
        """
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            result += chunk.content

    return result
