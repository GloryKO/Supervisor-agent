from typing import Literal
from state import AgentState
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from. tools import researcher_tools, fact_checker_tools
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"]   = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm= init_chat_model("groq:llama-3.1-8b-instant")

supervisor_llm= llm
researcher_lllm = llm.bind_tools(researcher_tools)
fact_checker_llm = llm.bind_tools(fact_checker_tools)
writer_llm = llm

SUPERVISOR_SYSTEM = """
You are a workflow supervisor managing a research pipeline.
Your job is to read the conversation so far and decide which agent should act next.
 
Available agents:
  researcher   — searches the web for information
  fact_checker — verifies specific claims found by the researcher  
  writer       — writes the final report for the user
 
How to decide:
 
  If no research has been done yet:
    → researcher
 
  If researcher just finished, read their findings carefully:
    → If the findings are clear, specific, and well-sourced with no contradictions:
         writer   (fact checking not needed — go straight to writing)
    → If the findings contain vague claims, conflicting information, or uncertain sources:
         fact_checker   (we need to verify before writing)
 
  If fact_checker just finished, read their verdict carefully:
    → If claims are VERIFIED:
         writer   (we can now write with confidence)
    → If claims are DISPUTED or UNVERIFIED:
         researcher   (we need to dig deeper with a more specific search)
 
  If writer just finished:
    → FINISH
 
Respond with ONLY one word: researcher | fact_checker | writer | FINISH
No explanation. No punctuation. Just the single word.
"""

def supervisor_agent(state:AgentState)-> AgentState:
    """
    Reads the full conversation and outputs the next agent name.
    This is a real decision — different conversations will produce different paths.
    """
    sys_msg = SystemMessage(content=SUPERVISOR_SYSTEM)
    messages = [sys_msg] + state["messages"]
    response = supervisor_llm.invoke(messages)
    decision = response.content.strip().lower()
    if decision not in ["researcher", "fact_checker", "writer", "finish"]:
        decision = "writer"  # default to writer if supervisor is confused
    current_cycle = state.get("current_cycle", 0)
    if decision == "researcher":
        current_cycle += 1
    if current_cycle >= 3:
        decision = "writer"  # avoid infinite loops, force writing after 3 cycles
    return {"next_agent": decision, "current_cycle": current_cycle, "messages":[]}

def researcher_agent(state:AgentState)-> AgentState:
    """
    First pass: broad search.
    On re-entry (cycle_count > 1): the supervisor sent us back because
    fact_checker found disputed claims, so we search more specifically.
    """
    cycle = state.get("current_cycle", 1)
    if cycle <= 1:
        instruction = (
            "You are a research assistant. "
            "Use search_web to find comprehensive information on the user's request. "
            "When done, clearly state your findings. "
            "Be explicit if any claims feel uncertain or if sources conflict — "
            "the supervisor uses this to decide whether fact-checking is needed."
        )
    else:
         instruction = (
            "You are a research assistant. "
            "The fact-checker found disputed or unverified claims in the previous research. "
            "Use search_web to dig deeper with MORE SPECIFIC queries to resolve those disputes. "
            "Focus on finding authoritative, recent sources. "
            "Clearly state whether you found clarifying evidence."
        )
    system_msg = SystemMessage(content=instruction)
    response = researcher_lllm.invoke([system_msg] + state["messages"])
    return {"messages":[response]}

 
def fact_checker_agent(state: AgentState) -> AgentState:
    """
    Extracts the key claims from the research and verifies them one by one.
    The verdict it writes ("VERIFIED" / "DISPUTED") directly drives the
    supervisor's next routing decision.
    """
    system_msg = SystemMessage(
        content=(
            "You are a fact-checker. "
            "Review the researcher's findings in the conversation. "
            "Identify the 2-3 most important specific claims. "
            "Use verify_claim to check each one. "
            "After checking, write a clear verdict summary:\n"
            "  - List each claim and whether it is VERIFIED, UNVERIFIED, or DISPUTED\n"
            "  - End with an overall verdict: OVERALL: VERIFIED or OVERALL: DISPUTED\n"
            "The supervisor reads this verdict to decide the next step."
        )
    )
    response = fact_checker_llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

def writer_agent(state: AgentState) -> AgentState:
    """
    Writes the final report. By this point the supervisor has ensured
    the research is either verified or as good as it's going to get.
    """
    system_msg = SystemMessage(
        content=(
            "You are a technical writer. "
            "Using the research and fact-checking results in the conversation, "
            "write a clear, professional final report. Structure it as:\n\n"
            "  ## Overview\n"
            "  ## Key Findings  (only include verified or uncontested claims)\n"
            "  ## Conclusion\n\n"
            "If any claims were disputed and not resolved, note them as unconfirmed."
        )
    )
    response = writer_llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

def researcher_should_use_tools(state:AgentState)->Literal["researcher, supervisor"]:
    last = state["messages"][-1]
    if hasattr(last,"tool_calls") and last.tool_calls:
        return "researcher_tools"
    return "supervisor"

def fact_checker_should_use_tools(state:AgentState)->Literal["fact_checker, supervisor"]:
    last = state["messages"][-1]
    if hasattr(last,"tool_calls") and last.tool_calls:
        return "fact_checker"
    return "supervisor"

def route_from_supervisor(state:AgentState)-> Literal["researcher", "fact_checker", "writer", "__end__"]:
    destination = state["next_agent"]
    return {
        "researcher": "researcher",
        "fact_checker": "fact_checker",
        "writer": "writer",
        "finish": "__end__"
    }.get(destination, "writer")  # default to writer if something goes wrong
