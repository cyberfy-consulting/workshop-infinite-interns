import logging
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain.agents.middleware import HumanInTheLoopMiddleware
from IPython.display import HTML, display
from utils.booking_system_db import CLOSING_TIME, OPENING_TIME, TODAY, booking_connection
from utils.visualize_restaurant import restaurant_html

LOG_LEVEL_NAME = "WARNING"

for logger_name in ("httpx", "httpcore", "openai", "azure"):
    logging.getLogger(logger_name).setLevel(LOG_LEVEL_NAME)


def weekday(date: str) -> str:
    return datetime.strptime(date, "%Y-%m-%d").strftime("%A")


def month(date: str) -> int:
    return datetime.strptime(date, "%Y-%m-%d").month


def month_to_season(month: int) -> str:
    seasons = ("winter", "winter", "spring", "spring", "spring", "summer",
               "summer", "summer", "autumn", "autumn", "autumn", "winter")
    return seasons[(int(month) - 1) % 12]


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

def _http_get_weather_data(
    start_date: date,
    latitude: float,
    longitude: float,
    days: int = FORECASTED_DAYS,
) -> Mapping[str, Any]:
    end_date = start_date + timedelta(days=days - 1)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "weather_code,temperature_2m",
        "timezone": "auto",
    }

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    try:
        response = session.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        raise ConnectionAbortedError(
            "Weather lookup failed: the weather service did not respond. Please try again."
        ) from exc
    if not response.ok:
        raise ConnectionAbortedError(
            f"Weather lookup failed ({response.status_code}): only past dates are available."
        )

    return response.json()


def pretty_weather_response(data: Mapping[str, Any]) -> str:
    hourly = data["hourly"]
    lines = []
    current_day: date | None = None

    for time_str, code, temp in zip(
        hourly["time"], hourly["weather_code"], hourly["temperature_2m"]
    ):
        stamp = datetime.fromisoformat(time_str)
        if not (OPENING_TIME <= f"{stamp:%H:%M}" <= CLOSING_TIME):
            continue
        if stamp.date() != current_day:
            current_day = stamp.date()
            lines.extend(
                [
                    f"Weather for {stamp:%A, %Y-%m-%d}:",
                    "\tHourly forecasts:",
                ]
            )
        description = WEATHER_FROM_CODE.get(code, f"code{code}")
        lines.append(f"\t\t{stamp:%H:%M}: {temp}°C, {description}")

    return "\n".join(lines)


HUMAN_IN_THE_LOOP_MIDDLEWARE = HumanInTheLoopMiddleware(
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "respond"],
            "description": "Please review the tool call",
        }
    }
)

_previous_ids = None


def list_reservations_df(reservation_date: date | str | None = TODAY):
    global _previous_ids
    query = (
        "SELECT reservation_id, name, reservation_time, party_size, outside, waterfront, shade, table_id "
        "FROM reservations"
    )
    params = ()
    if reservation_date is not None:
        query += " WHERE date(reservation_time) = ?"
        params = (str(reservation_date),)

    with booking_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df["reservation_time"] = df["reservation_time"].astype(str)
    for column in ("outside", "waterfront", "shade"):
        df[column] = df[column].fillna(0).replace({0: False, 1: True})
    
    ids = set(df["reservation_id"])
    if _previous_ids is None:
        _previous_ids = ids  # Seed reservations are not highlighted on the first listing.
    
    # Reservations added since the previous listing.
    new_ids = ids - _previous_ids
    _previous_ids = ids
    
    # yellow background for every new reservation
    styled = df.style.apply(
        lambda row: ["background-color: #98c3f7" if row["reservation_id"] in new_ids else ""] * len(row),
        axis=1,
    ).hide(axis="index")
    display(styled)

    #floor plan: booked tables red with "guest, time" underneath, new reservations blue
    with_table = df.dropna(subset=["table_id"])
    raw_times = with_table["reservation_time"]
    time_parts = raw_times.str.rsplit(" ", n=1).str[-1]
    times = time_parts.str.slice(0, 5).where(time_parts.str.contains(":"), raw_times)
    booked = dict(zip(with_table["table_id"], with_table["name"] + ", " + times))
    new = set(df.loc[df["reservation_id"].isin(new_ids), "table_id"].dropna())
    return HTML(restaurant_html(booked=booked, new=new))



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