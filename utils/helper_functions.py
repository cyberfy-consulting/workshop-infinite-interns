import logging
import os
import sqlite3
from typing import List

import pandas as pd
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain.agents.middleware import HumanInTheLoopMiddleware
from IPython.display import HTML, display
from utils.restaurant_layout import restaurant_html

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

DB_PATH = os.path.join(os.path.dirname(__file__), "booking.db")

#restaurant reservation hours
OPENING_TIME = "18:00"
CLOSING_TIME = "22:00"   
# days of weather returned, starting at start_date
FORECASTED_DAYS = 3

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
    
    is_dt = pd.api.types.is_datetime64_any_dtype(df["reservation_time"])
    # yellow background for every new reservation
    styled = df.style.apply(
        lambda row: ["background-color: #98c3f7" if row["reservation_id"] in new_ids else ""] * len(row),
        axis=1,
    ).hide(axis="index")
    if is_dt:
        styled = styled.format({"reservation_time": "{:%Y-%m-%d %H:%M}"})  # drops row-number index, no seconds in reservation_time
    display(styled)

    #floor plan: booked tables red with "guest, time" underneath, new reservations blue
    with_table = df.dropna(subset=["table_id"])
    times = with_table["reservation_time"].dt.strftime("%H:%M") if is_dt else with_table["reservation_time"].astype(str)
    booked = dict(zip(with_table["table_id"], with_table["name"] + ", " + times))
    new = set(df.loc[df["reservation_id"].isin(new_ids), "table_id"].dropna())
    return HTML(restaurant_html(booked=booked, new=new))



def pretty_print_chunk(log: dict):
    if "__interrupt__" in log:
        print("🔔 Interrupt received:")
        interrupts = log["__interrupt__"]
        if not isinstance(interrupts, (tuple, list)):
            interrupts = (interrupts,)
        for intr in interrupts:
            value = intr.value
            for idx, (action, config) in enumerate(
                zip(value["action_requests"], value["review_configs"], strict=True),
                start=1,
            ):
                print(f" {idx}. Action: {action.get('name')}  args={action.get('args')}")
                print(f"    Description: {action.get('description', '')}")
                print(f"    Decisions:   {config.get('allowed_decisions', [])}")
        print()
        return

    for agent_name, agent_data in log.items():
        print(f"Agent: {agent_name}")
        messages = (agent_data or {}).get("messages", [])
        if not messages:
            print("  (no messages)\n")
            continue

        pretty_print(messages)  # blank line between agents



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


HUMAN_IN_THE_LOOP_MIDDLEWARE = HumanInTheLoopMiddleware(
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "respond"],
            "description": "Please review the tool call",
        }
    }
)