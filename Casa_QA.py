import os, json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime
from Config.LLM_Client import llm
from Tools.Thinking import (
    recalling,
    recognizing,
    interpreting,
    exemplifying,
    direct_answer,
    decompose,
    checking,
    critiquing,
)
from Tools.Search import (
    insurance_terms_search,  # 保险术语查询
    consumer_protection_terms_search,  # 消保术语查询
    rules_search,  # 审核规则查询
)

# 载入工具
tools = [
    recalling,  # 上下文回忆
    recognizing,  # 要素识别
    interpreting,  # 任务解释
    exemplifying,  # 任务举例
    decompose,  # 任务分解
    checking,  # 信息检查
    critiquing,  # 方案评论
    # direct_answer,  # 直接作答
    insurance_terms_search,  # 保险术语查询
    consumer_protection_terms_search,  # 消保术语查询
    rules_search,  # 审核规则查询
]
llm_with_tools = llm.bind_tools(tools)


# 固化信息
class ContextSchema(TypedDict):
    """
    不更改的信息，包括上传的文档信息
    """

    document: str = ""


# 状态信息
class State(TypedDict):
    """
    保持更新的状态定义
    """

    messages: Annotated[list, add_messages]
    memory: list
    query: str
    query_rewrite: str
    response: str = "很抱歉，该问题目前无法回答。"
    n_tools: int
    n_loop: int = 0


# 拒答判断
def reject(state: State):
    """
    判断是否拒绝回答，更新在状态的"reject"字段
    1. 返回“回答”：目标明确且不违规
    2. 返回“拒答”：目标不明确或违规
    """
    # print(f"\n=====我们看看进入到reject的状态是啥样：=====\n{state}\n")
    query = state["query"]
    messages = state["messages"]
    prompt = f"请综合历史上下文，判断当前问题是否存在色情、暴力等安全风险，如果不存在安全风险则返回“回答”，否则返回“拒答”。用户当前的问题是：{query}。历史上下文为：{messages}"
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # 流式输出
    response = ""
    for chunk in llm.stream(prompt):
        response += chunk.content

    # # 直接输出
    # response = llm.invoke(messages).content
    messages.pop()  # 剔除中间判断信息

    # 过程记录
    state["memory"].extend(
        [
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]
    )
    if "回答" in response:
        return "回答"
    else:
        state["response"] = "很抱歉，该问题目前无法回答。"
    return "拒答"


def planing(state: State) -> State:
    """
    计划下一步的操作
    """
    message = state["messages"]
    query = state["query"]
    ctx = get_runtime(ContextSchema)
    document = ctx.context.get("document", "")

    # 存在审核文档是审核任务
    if document != "":
        prompt = f"""
        用户上传了一份待审核的宣传文档内容，我们会通过任务规划、行动和反思的闭环来完成文档审核任务，只需要聚焦文档就可以，不要发散到文档内提及的其他文档。
        你是其中的任务规划模块，需要根据用户的指令和上下文轨迹，通过调用各种认知工具和信息搜索工具，来规划下一步的行动。相关指示如下：
        1.宣传文档内容是：{document}。用户问题是：{query}。上下文轨迹是：{message}。
        2.认知工具清单：上下文回忆、要素识别、任务解释、任务举例、任务分解、信息检查、方案评论。他们是帮助你在复杂任务下做好思考的，如果你认为当前任务很简单，可以不调用认知工具。
        3.信息搜索工具清单：保险术语查询、消保术语查询、审核规则查询。他们是帮助你获取额外的信息，这些信息在文档和用户提问中没有直接给出，如果你认为不需要额外信息也能回答，可以不调用搜索工具。
        4.请给出你认为后续执行的行动是什么，将交给行动模块去具体执行，工具仅局限在目前提及的认知工具和信息搜索工具。
        5.同一个工具不能被连续调用，除非上一轮执行失败。
        6.如果你认为当前已经执行完毕用户的任务指令，则无需再调用工具，直接返回最终答案即可。
        """
    # 不存在审核文档就当成问答任务
    else:
        prompt = f"""
        给定用户的任务指令，我们会通过任务规划、行动和反思闭环来完成用户的任务。
        你是其中的任务规划模块，需要根据用户的指令和上下文轨迹，通过调用各种认知工具和信息搜索工具，来规划下一步的行动。相关指示如下：
        1.用户问题是：{query}。上下文轨迹是：{message}。
        2.请给出你认为后续执行的行动是什么，将交给行动模块去具体执行。
        3.认知工具清单：上下文回忆、要素识别、任务解释、任务举例、任务分解、信息检查、方案评论。他们是帮助你在复杂任务下做好思考的，如果你认为当前任务很简单，可以不调用认知工具。
        4.信息搜索工具清单：保险术语查询、消保术语查询、审核规则查询。他们是帮助你获取额外的信息，这些信息用户提问中没有直接给出，如果你认为不需要额外信息也能回答，可以不调用搜索工具。
        5.工具调用会消耗大量时间，如果一个工具多次执行失败或者效果不佳，请更换工具避免陷入循环。
        6.如果你认为当前已经执行完毕用户的任务指令，则无需再调用工具，直接返回最终答案即可。
        """

    # 如果上一个工具调用是直接回答，就结束了，不再规划。
    messages = state["messages"]
    if len(messages) >= 2:
        if messages[-2].name == "直接作答" and "有效" in messages[-1].content:
            print(f"\n\n👍任务完成")
            return {"response": messages[-2].content, "n_tools": 0}

    # 流式输出
    gen = llm_with_tools.stream(prompt)
    response = None
    print("\n")
    for chunk in gen:
        if response is None:
            response = chunk
        else:
            response = response + chunk

        if not response.tool_calls:
            print(chunk.content, end="", flush=True)

    # # 直接输出
    # response = llm_with_tools.invoke(prompt)

    # 根据是否有工具调用来判断任务结束
    if response.tool_calls:
        print(f"\n\n👉规划下一步：工具调用\n{response.tool_calls}")
        return {"messages": response, "n_tools": len(response.tool_calls)}
    else:
        print(f"\n\n👍任务完成")
        return {"messages": response, "response": response.content, "n_tools": 0}


def should_use_tool(state: State) -> str:
    n_tools = state["n_tools"]
    if n_tools > 0:
        return "tools"
    else:
        return END


def verify_tool_call(state: State) -> State:
    """
    验证工具调用是否正确
    """
    query = state["query"]
    n_tools = state["n_tools"]
    message = state["messages"]
    for tool_response in message[-n_tools:]:
        state["memory"].append(tool_response)

    demo = {
        "工具调用": ["上下文回忆"],
        "结论": "有效或无效",
        "反思": "正确执行了规划模块的行动要求；但任务尚未完成，仍需后续的信息获取动作来真正满足用户查询需求。",
    }
    prompt = f"""
    给定用户的指令，我们会通过任务规划、行动和反思闭环来完成用户的任务。。
    你是其中的验证反思模块，需要判断工具的执行结果是否有效推进了任务解答。相关指示如下：
    1. 当前的用户问题是：{query}。历史执行轨迹是：{message[:-n_tools]}。行动结果为：{message[-n_tools :]}。
    2. 请给出你对工具调用结果的分析，判断是否满足规划模块的符合预期，返回的参考样例为{demo}，请保持输出的精炼简洁。
    """
    # 流式输出
    chunks = []
    print(f"\n\n👀验证反思结果")
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        chunks.append(chunk.content)

    response = "".join(chunks)
    state["memory"].append({"verify_tool_call": response})

    # # 完整输出
    # response = llm.invoke(prompt)
    # state["memory"].append({"verify_tool_call": response.content})
    # print(f"\n👉验证反思结果\n{response.content}")

    return {"messages": AIMessage(content=response), "n_loop": state["n_loop"] + 1}


def break_loop(state: State) -> State:
    """
    验证次数超出最大次数时，跳出循环
    """
    if state["n_loop"] >= 100:
        return END
    else:
        return "continue"


graph = StateGraph(State)
graph.add_node("规划", planing)
tool_node = ToolNode(tools=tools)
graph.add_node("工具调用", tool_node)
graph.add_node("行动验证", verify_tool_call)


graph.add_conditional_edges(START, reject, {"拒答": END, "回答": "规划"})
graph.add_conditional_edges("规划", should_use_tool, {"tools": "工具调用", END: END})
graph.add_edge("工具调用", "行动验证")
graph.add_conditional_edges("行动验证", break_loop, {"continue": "规划", END: END})
# 编译静态图
app = graph.compile()

# # 粗略可视化
# app.get_graph().print_ascii()

# # 保存静态图
# if not os.path.exists("./output"):
#     os.makedirs("./output")
# png_data = app.get_graph(xray=True).draw_mermaid_png()
# with open("./output/graph.png", "wb") as f:
#     f.write(png_data)

if __name__ == "__main__":
    """
    文案参考
    分红险：家庭财富的最优保障选择\n\n既要稳稳的收益，又要满满的安心？呱呱分红险必选！比存款利息高、比基金更靠谱，下有 3% 保证收益托底，上有年年分红稳赚超额收益，经济再波动也能躺赢！不管是养老规划还是子女教育，都是不二之选～

    指令参考
    百万医疗险的保证续保措辞有什么要注意的？
    请帮我指出文档中哪些内容违反消保合规问题，分别违反了什么规定？
    """

    state = {
        "query": "",
        "messages": [],
        "memory": [],
        "n_tools": 0,
        "n_loop": 0,
        "response": "回答完毕",
    }

    print("\n🤖 机器人：你好呀！")
    while True:
        # 获取用户文档
        document = input("\n📝 请输入待审核文档（如果没有请直接回车）: ")
        context = {
            "document": document,
        }
        # 获取用户输入
        user_input = input("\n👤 你: ")

        # 如果输入 exit 就结束
        if user_input.lower() == "exit":
            print("\n🤖 机器人: 再见！")
            break
        else:
            user_input = user_input

        # 更新agent状态，替换为本轮用户查询
        state["query"] = user_input
        state["response"] = ""
        # print(state)
        state = app.invoke(
            state,
            context=context,
            config={"recursion_limit": 100},  # 因为这里多步思考可能会很多轮
        )
        response = state["response"]
        print(f"\n🤖 机器人：{response}")
