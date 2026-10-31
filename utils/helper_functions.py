import logging
import sqlite3
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from geopy.geocoders import Nominatim
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

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

def get_current_time(years_back: int = 0) -> str:
    """
    Get the current time formatted nicely for display.
    years_back shifts the date into the past (e.g. 10 = same day, 10 years ago),
    keeping the weekday correct for that date.
    """
    now = datetime.now(ZoneInfo("Europe/Zurich"))
    if years_back:
        try:
            now = now.replace(year=now.year - years_back)
        except ValueError:  # Feb 29 doesn't exist in the target year
            now = now.replace(year=now.year - years_back, day=28)
    formatted = now.strftime("%A, %B %d, %Y at %H:%M")
    return f"It is currently {formatted}."

# Returns (latitude, longitude). Nominatim supports max 1 request per second
def get_geolocation(location: str):
    app = Nominatim(user_agent="ai_agents")
    raw = app.geocode(location).raw
    return float(raw["lat"]), float(raw["lon"])

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

_initial_ids = None   # ids at the first listing = initial setup, never highlighted

def list_reservations_df():
    global _initial_ids
    conn = sqlite3.connect("utils/booking.db")
    df = pd.read_sql(
        "SELECT id, name, reservation_time, party_size, outside FROM reservations",
        conn
    )
    conn.close()
    if not df.empty and not "Eve" in df["name"].values:
        df["reservation_time"] = pd.to_datetime(df["reservation_time"], format='mixed')
        df["outside"] = df["outside"].astype(bool)

    if _initial_ids is None:
        _initial_ids = set(df["id"])    # first call: nothing is highlighted

    # green background for every reservation that wasn't there at the first listing
    return df.style.apply(
        lambda row: ["background-color: #c6efce" if row["id"] not in _initial_ids else ""] * len(row),
        axis=1,
    )

def list_customers_df():
    conn = sqlite3.connect("utils/loyalty.db")
    df = pd.read_sql(
        "SELECT name, address, loyalty_points FROM customers",
        conn
    )
    conn.close()
    if not df.empty:
        df["loyalty_points"] = df["loyalty_points"].astype(int)
    return df

def restore_booking_db():
    """
    Restore the booking database to its original state
    """
    conn = sqlite3.connect("utils/booking.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM reservations")  # Clear existing data
    # Reinsert original data
    cur.executemany(
        "INSERT INTO reservations (name, reservation_time, party_size, outside) VALUES (?, ?, ?, ?)",
        [
            ("Alice", "2026-06-27 18:30", 2, False),
            ("Bob",   "2026-06-27 19:00", 4, True),
            ("Carol", "2026-06-28 12:30", 3, False),
            ("Dave",  "2026-06-26 20:00", 5, True),
        ]
    )
    conn.commit()
    conn.close()

def restore_loyalty_db():
    """
    Restore the loyalty database to its original state
    """
    conn = sqlite3.connect("utils/loyalty.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM customers")  # Clear existing data
    # Reinsert original data
    cur.executemany(
        "INSERT INTO customers (name, address, loyalty_points) VALUES (?, ?, ?)",
        [
            ("Max Mustermann", "Musterstrasse 1, Musterstadt", 100),
            ("John Doe", "Doe Street 2, Example City", 150),
        ]
    )
    conn.commit()
    conn.close()



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