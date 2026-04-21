
from langgraph.graph import  MessagesState
 

class AgentState(MessagesState):
    """
    next_agent: the supervisor writes a value here to control routing.
    MessagesState already provides: messages: Annotated[list[BaseMessage], add_messages]
    """
    next_agent: str
    current_cycle: int