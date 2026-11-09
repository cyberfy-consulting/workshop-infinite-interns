import logging
import os
import sqlite3
import pandas as pd
import time, textwrap
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Callable
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt 
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt

from IPython.display import HTML, display
from utils.restaurant_layout import restaurant_html

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

DB_PATH = os.path.join(os.path.dirname(__file__), "booking.db")

#restaurant opening hours
OPENING_TIME = "18:00"
CLOSING_TIME = "22:00"   
# days of weather returned, starting at start_date
DAYS = 3

# WMO weather codes used by Open-Meteo
WEATHER_FROM_CODE = {
    0: "Clear sky", 
    1: "Mainly clear", 
    2: "Partly cloudy", 
    3: "Overcast",
    45: "Fog", 
    48: "Depositing rime fog",
    51: "Light drizzle", 
    53: "Moderate drizzle", 
    55: "Dense drizzle",
    56: "Light freezing drizzle", 
    57: "Dense freezing drizzle",
    61: "Slight rain", 
    63: "Moderate rain", 
    65: "Heavy rain",
    66: "Light freezing rain", 
    67: "Heavy freezing rain",
    71: "Slight snow fall", 
    73: "Moderate snow fall", 
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 
    81: "Moderate rain showers", 
    82: "Violent rain showers",
    85: "Slight snow showers", 
    86: "Heavy snow showers",
    95: "Thunderstorm", 
    96: "Thunderstorm with slight hail", 
    99: "Thunderstorm with heavy hail",
}

_previous_ids = None   # ids at the first listing = initial setup, never highlighted. Only new reservations with each call are highlighted

def list_reservations_df():
    global _previous_ids
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT reservation_id, name, reservation_time, party_size, outside, waterfront, shade, table_id FROM reservations",
        conn
    )
    conn.close()
    try:
        df["reservation_time"] = pd.to_datetime(df["reservation_time"], format='mixed') #parses mixed time formats and returns consistent datetime
        #return integers 0,1 as booleans
        df["outside"] = df["outside"].astype(bool) 
        df["waterfront"] = df["waterfront"].fillna(0).astype(bool)
        df["shade"] = df["shade"].fillna(0).astype(bool) 
    except (ValueError, TypeError):
        pass # data isn't readable as dates 
    
    ids = set(df["reservation_id"])
    if _previous_ids is None:
        _previous_ids = ids    # first call: nothing is highlighted with initial reservation list
    
    #reservations added since the first listing
    new_ids = ids - _previous_ids      
    _previous_ids = ids # the next call compares against this listing

    # yellow background for every new reservation
    styled = df.style.apply(
        lambda row: ["background-color: #98c3f7" if row["reservation_id"] in new_ids else ""] * len(row),
        axis=1,
    ).hide(axis="index").format({"reservation_time": "{:%Y-%m-%d %H:%M}"})  # drops row-number index, no seconds in reservation_time
    display(styled)

    #floor plan: booked tables red with "guest, time" underneath, new reservations blue
    with_table = df.dropna(subset=["table_id"])
    booked = dict(zip(with_table["table_id"],
                      with_table["name"] + ", " + with_table["reservation_time"].dt.strftime("%H:%M")))
    new = set(df.loc[df["reservation_id"].isin(new_ids), "table_id"].dropna())
    return HTML(restaurant_html(booked=booked, new=new))



def pretty_print_chunk(log: dict):
    """
    Nicely prints a single agent log dict of the form:
      {
        "agent_name": {
          "messages": [HumanMessage(...), AIMessage(...), ToolMessage(...), ...]
        }
      }
    """
    # ---- handle interrupts first ----
    if "__interrupt__" in log:
        # the value is typically a 1‐tuple containing an Interrupt instance
        intr = log["__interrupt__"]
        if isinstance(intr, tuple):
            intr = intr[0]
        print("🔔 Interrupt received:")
        for idx, entry in enumerate(intr.value, start=1):
            ar     = entry.get("action_request", {})
            config = entry.get("config", {})
            desc   = entry.get("description", "")
            print(f" {idx}. Action: {ar.get('action')}  args={ar.get('args')}")
            print(f"    Description: {desc}")
            print(f"    Config:      {config}")
        print()
        return   # nothing more to do

    for agent_name, agent_data in log.items():
        print(f"Agent: {agent_name}")
        messages = (agent_data or {}).get("messages", [])
        if not messages:
            print("  (no messages)\n")
            continue

        for idx, msg in enumerate(messages, start=1):
            # Message type (class name)
            msg_type = msg.__class__.__name__
            print(f" {idx}. {msg_type}")
            # Core fields
            content = getattr(msg, 'content', '')
            print("    Content:")
            for line in content.splitlines():
                    print(f"      {line}")

            # Additional kwargs (e.g. tool_calls embedded)
            additional = getattr(msg, "additional_kwargs", None)
            if additional:
                # If tool_calls exist under additional_kwargs, print them
                tcs = additional.get("tool_calls") or []
                if tcs:
                    print("    Tool calls:")
                    for tc in tcs:
                        name = tc.get("name") or tc.get("function", {}).get("name")
                        args = tc.get("args") or tc.get("function", {}).get("arguments")
                        print(f"      • {name} args={args}")

            # Also check for top‐level .tool_calls list
            top_tcs = getattr(msg, "tool_calls", [])
            if top_tcs:
                print("    Tool calls (top‐level):")
                for tc in top_tcs:
                    name = tc.get("name")
                    args = tc.get("args")
                    print(f"      • {name} args={args}")

            print()  # blank line between messages
        print()  # blank line between agents



def pretty_print(messages: List[BaseMessage]) -> None:
    """
    Print a sequence of LangChain messages in a concise, human-readable form,
    omitting all IDs, token counts, and other metadata.
    """
    for msg in messages:
        role = type(msg).__name__
        # Map class names to simpler labels
        if isinstance(msg, SystemMessage):
            label = "[System]"
        elif isinstance(msg, HumanMessage):
            label = "[User]"
        elif isinstance(msg, AIMessage):
            # If the AI called tools, list their names
            tools = getattr(msg, "tool_calls", None) or []
            if tools:
                tool_names = ", ".join(call["name"] for call in tools)
                label = f"[AI → {tool_names}]"
            else:
                label = "[AI]"
        elif isinstance(msg, ToolMessage):
            label = f"[Tool: {msg.name}]"
        else:
            label = f"[{role}]"

        # Print the label and the actual content
        print(f"{label} {msg.content}\n")


def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review.""" 
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(  
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    async def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }
        response = interrupt([request])[0]  
        # approve the tool call
        if response["type"] == "accept":
            tool_response = await tool.ainvoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = await tool.ainvoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt