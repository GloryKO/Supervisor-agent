import os
from dotenv import load_dotenv
from graph import app

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "truly-dynamic-demo"}}
 
    result = app.invoke(
        {
            "messages":    [("user", "What are the latest developments in AI agents in 2025?")],
            "cycle_count": 0,
        },
        config=config,
    )
 
    print("\n=== Final Report ===")
    print(result["messages"][-1].content)
