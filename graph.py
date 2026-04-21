from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .tools import researcher_tool_node, fact_checker_tool_node
from langgraph.checkpoint.memory import MemmorySaver

from .state import AgentState
from .agents import supervisor_agent, researcher_agent, fact_checker_agent, writer_agent, route_from_supervisor, researcher_should_use_tools, fact_checker_should_use_tools


workflow = StateGraph(AgentState)
workflow.add_node("supervisor",supervisor_agent)
workflow.add_node("researcher",         researcher_agent)
workflow.add_node("researcher_tools",   researcher_tool_node)
workflow.add_node("fact_checker",       fact_checker_agent)
workflow.add_node("fact_checker_tools", fact_checker_tool_node)
workflow.add_node("writer",           writer_agent)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor",route_from_supervisor,{
        "researcher": "researcher",
        "fact_checker": "fact_checker",
        "writer": "writer",
        "__end__": END
})

workflow.add_conditional_edges("researcher", researcher_should_use_tools, {
    "researcher_tools": "researcher_tools",
    "supervisor": "supervisor"
})

workflow.add_conditional_edges("fact_checker", fact_checker_should_use_tools, {
    "fact_checker_tools": "fact_checker_tools",
    "supervisor": "supervisor"
})

workflow.add_edge("researcher_tools", "researcher")
workflow.add_edge("fact_checker_tools", "fact_checker")
workflow.add_edge("writer", END)
memory = MemmorySaver()


app = workflow.compile(checkpointer=memory)