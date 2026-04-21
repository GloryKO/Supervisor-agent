from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.base import tool


@tool
def search_web(query: str) -> TavilySearchResults:
    """Search the web for recent information.
    Use when you need current facts, news, or data.
    """
    
    search= TavilySearchResults(max_results=3)
    return str(search.invoke(query))
 
@tool
def verify_claim(claim: str) -> str:
    """
    Independently verify a specific claim by searching the web for
    corroborating or contradicting evidence.
    Use this to fact-check a statement before it reaches the final report.
    Returns supporting evidence and a VERIFIED / UNVERIFIED / DISPUTED label.
    """
    search = TavilySearchResults(max_results=3)
    results = search.invoke(f"verify: {claim}")
    results_text = str(results)
 
    # Simple heuristic verdict based on result quality
    if not results_text or len(results_text) < 100:
        verdict = "UNVERIFIED — insufficient evidence found"
    elif any(word in results_text.lower() for word in ["false", "misleading", "debunked", "incorrect"]):
        verdict = "DISPUTED — contradicting evidence found"
    else:
        verdict = "VERIFIED — supporting evidence found"
 
    return f"Claim: {claim}\nVerdict: {verdict}\n\nEvidence:\n{results_text[:800]}"

researcher_tools   = [search_web]
fact_checker_tools = [verify_claim]
researcher_tool_node   = ToolNode(researcher_tools)
fact_checker_tool_node = ToolNode(fact_checker_tools)