from langchain_openai import ChatOpenAI
from threading import Lock

lock = Lock()

# 请修改
llm_url = "your_llm_url"
api_key = "your_api_key"

# qwen2.5用这个
llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",  # 非思考模型试试水
    openai_api_key=api_key,
    openai_api_base=llm_url,
)

# # quen3，流式输出必须设置"enable_thinking":True
# llm = ChatOpenAI(
# #     # model="qwen-plus",  # 目前也是qwen3
#     model="qwen3-32b",  # 小模型试试水
#     openai_api_key=api_key,
#     openai_api_base=llm_url,
#     extra_body={"enable_thinking": True},
# )

