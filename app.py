import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import calendar
from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field
import plotly.express as px


# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough


# --- Configuration ---
# Ensure your GROQ_API_KEY is set as an environment variable
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY environment variable not set. Please set it and rerun the app."
    )
    st.stop()

DATA_FILE = "oee_data.xlsx"


# --- Data Loading (Cached) ---
@st.cache_data
def load_oee_data(file_path: str):
    """Loads OEE data from an Excel file."""
    if not os.path.exists(file_path):
        st.error(
            f"Data file '{file_path}' not found. Please run the data generation script."
        )
        return None
    try:
        df = pd.read_excel(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        df["Year"] = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["Week"] = df["Timestamp"].dt.isocalendar().week
        df["Year_Week"] = df["Timestamp"].dt.strftime("%Y-W%W")
        df["Year_Month"] = df["Timestamp"].dt.strftime("%Y-%m")

        st.success(
            f"Successfully loaded data from '{file_path}'. Data shape: {df.shape}"
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# --- Data Filtering, Aggregation, and Calculation ---


def filter_data(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    selected_devices: List[str],
    selected_locations: List[str],
) -> pd.DataFrame:
    """Filters DataFrame based on date range, devices, and locations."""
    df_filtered = df.copy()

    # Date filtering
    df_filtered = df_filtered[
        (df_filtered["Date"] >= start_date) & (df_filtered["Date"] <= end_date)
    ]

    # Device filtering
    if selected_devices:  # Filter only if devices are selected
        df_filtered = df_filtered[df_filtered["Device_ID"].isin(selected_devices)]

    # Location filtering
    if selected_locations:  # Filter only if locations are selected
        df_filtered = df_filtered[df_filtered["Location"].isin(selected_locations)]

    return df_filtered


def aggregate_oee_data(df_subset: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
    """Aggregates data by the specified time level and calculates OEE metrics."""
    if df_subset.empty:
        return pd.DataFrame()

    if aggregation_level == "Daily":
        group_col = "Date"
        sort_order = "Date"
    elif aggregation_level == "Weekly":
        group_col = "Year_Week"  # Group by Year-Week string
        sort_order = "Year_Week"
    elif aggregation_level == "Monthly":
        group_col = "Year_Month"  # Group by Year-Month string
        sort_order = "Year_Month"
    else:
        raise ValueError(f"Invalid aggregation level: {aggregation_level}")

    # Aggregate raw metrics
    agg_data = (
        df_subset.groupby(group_col)
        .agg(
            total_scheduled_time_hrs=("Scheduled_Run_Time_Hrs", "sum"),
            total_downtime_hrs=("Downtime_Hrs", "sum"),
            total_units_produced=("Total_Units_Produced", "sum"),
            total_good_units_produced=("Good_Units_Produced", "sum"),
            avg_ideal_cycle_time_sec=("Ideal_Cycle_Time_Sec", "mean"),
        )
        .reset_index()
    )

    # Calculate OEE components for each aggregated period
    # Availability
    agg_data["running_time_hrs"] = (
        agg_data["total_scheduled_time_hrs"] - agg_data["total_downtime_hrs"]
    )
    agg_data["Availability (%)"] = (
        agg_data["running_time_hrs"] / agg_data["total_scheduled_time_hrs"]
    ) * 100.0
    agg_data["Availability (%)"] = (
        agg_data["Availability (%)"].fillna(0).replace([np.inf, -np.inf], 0)
    )

    # Performance
    agg_data["ideal_run_time_hrs"] = (
        agg_data["total_units_produced"] * agg_data["avg_ideal_cycle_time_sec"]
    ) / 3600.0
    agg_data["Performance (%)"] = (
        agg_data["ideal_run_time_hrs"] / agg_data["running_time_hrs"]
    ) * 100.0
    agg_data["Performance (%)"] = (
        agg_data["Performance (%)"].fillna(0).replace([np.inf, -np.inf], 0)
    )
    agg_data["Performance (%)"] = agg_data["Performance (%)"].clip(upper=100.0)

    # Quality
    agg_data["Quality (%)"] = (
        agg_data["total_good_units_produced"] / agg_data["total_units_produced"]
    ) * 100.0
    agg_data["Quality (%)"] = (
        agg_data["Quality (%)"].fillna(0).replace([np.inf, -np.inf], 0)
    )

    # OEE
    agg_data["OEE (%)"] = (
        (agg_data["Availability (%)"] / 100.0)
        * (agg_data["Performance (%)"] / 100.0)
        * (agg_data["Quality (%)"] / 100.0)
        * 100.0
    )

    # Drop intermediate columns if not needed for plotting
    agg_data = agg_data.drop(columns=["running_time_hrs", "ideal_run_time_hrs"])

    # Sort by the time column
    agg_data = agg_data.sort_values(by=sort_order)

    return agg_data


def calculate_oee(df_subset: pd.DataFrame) -> dict:
    """
    Calculates OEE and its components (Availability, Performance, Quality)
    from a DataFrame subset, aggregating data within the subset.
    Handles potential division by zero by returning 0 for components.
    """
    if df_subset.empty:
        return {"error": "No data available for the selected filters."}

    # Aggregate key metrics over the subset
    total_scheduled_time_hrs = df_subset["Scheduled_Run_Time_Hrs"].sum()
    total_downtime_hrs = df_subset["Downtime_Hrs"].sum()
    total_units_produced = df_subset["Total_Units_Produced"].sum()
    total_good_units_produced = df_subset["Good_Units_Produced"].sum()

    # Use the average Ideal_Cycle_Time_Sec for the subset
    avg_ideal_cycle_time_sec = df_subset["Ideal_Cycle_Time_Sec"].mean()

    # Availability
    running_time_hrs = total_scheduled_time_hrs - total_downtime_hrs
    availability = 0.0
    if total_scheduled_time_hrs > 0:
        availability = (running_time_hrs / total_scheduled_time_hrs) * 100.0

    # Performance
    ideal_run_time_hrs = 0.0
    if pd.notna(avg_ideal_cycle_time_sec) and avg_ideal_cycle_time_sec > 0:
        ideal_run_time_hrs = (total_units_produced * avg_ideal_cycle_time_sec) / 3600.0

    performance = 0.0
    if running_time_hrs > 0:
        performance = (ideal_run_time_hrs / running_time_hrs) * 100.0
    performance = min(performance, 100.0)

    # Quality
    quality = 0.0
    if total_units_produced > 0:
        quality = (total_good_units_produced / total_units_produced) * 100.0

    # OEE
    oee = (availability / 100.0) * (performance / 100.0) * (quality / 100.0) * 100.0

    return {
        "availability": round(availability, 2),
        "performance": round(performance, 2),
        "quality": round(quality, 2),
        "oee": round(oee, 2),
        "total_scheduled_time_hrs": round(total_scheduled_time_hrs, 2),
        "total_downtime_hrs": round(total_downtime_hrs, 2),
        "running_time_hrs": round(running_time_hrs, 2),
        "total_units_produced": int(total_units_produced),
        "total_good_units_produced": int(total_good_units_produced),
        "avg_ideal_cycle_time_sec": (
            round(avg_ideal_cycle_time_sec, 2)
            if pd.notna(avg_ideal_cycle_time_sec)
            else None
        ),
    }


# --- Conversational AI Tool Definition ---


class OEEQueryInput(BaseModel):
    device_id: Optional[str] = Field(
        None,
        description="Specific Device ID to filter by. If not provided, data for all devices matching other filters will be used.",
    )
    location: Optional[str] = Field(
        None,
        description="Specific Location to filter by. If not provided, data for all locations matching other filters will be used.",
    )
    month: Optional[str] = Field(
        None, description="Month to filter by (e.g., 'January', 'Feb', '03')."
    )
    year: Optional[int] = Field(
        None,
        description="Year to filter by (e.g., 2024, 2025). Must be a 4-digit year.",
    )


@tool("calculate_oee_chat_query", args_schema=OEEQueryInput)
def calculate_oee_chat_query_tool(
    device_id: Optional[str] = None,
    location: Optional[str] = None,
    month: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """
    Calculates the Overall Equipment Efficiency (OEE) and its components
    (Availability, Performance, Quality) for packaging devices based on filters
    extracted from a user's chat query. This tool processes explicit requests
    from the chat interface, independent of the UI filters set for plotting.

    Filters can include a specific Device ID, Location, Month (by name or number), and Year.
    If no filters are provided in the query, calculates OEE for the entire dataset.
    Filters are combined using AND logic.

    Provide the Device ID (e.g. 'D02'), Location (e.g. 'Chennai'), Month (e.g., 'January', 'March', '6'), and Year
    (e.g., 2024) based on the user's request in the chat.
    Month can be the full name, abbreviation, or number (1-12).
    Year must be a 4-digit number.

    Example calls:
    calculate_oee_chat_query(device_id='D02', month='January', year=2024)
    calculate_oee_chat_query(location='chennai', year=2025)
    calculate_oee_chat_query(month='July', year=2024)
    calculate_oee_chat_query(device_id='D05') # Calculates OEE for D05 across all time/locations
    calculate_oee_chat_query() # Calculates overall OEE
    """
    # Access the globally loaded DataFrame (cached by Streamlit)
    df_filtered = st.session_state.oee_df.copy()

    # Apply filters from the chat query parameters
    if device_id:
        # Use exact match for clarity from chat queries
        df_filtered = df_filtered[df_filtered["Device_ID"] == device_id]
        if df_filtered.empty:
            return f"No data found for Device ID '{device_id}'."

    if location:
        # Use exact match for clarity from chat queries
        df_filtered = df_filtered[df_filtered["Location"] == location]
        if df_filtered.empty:
            return f"No data found for Location '{location}' matching other criteria."

    if month:
        try:
            if month.isdigit():
                month_num = int(month)
                if not 1 <= month_num <= 12:
                    return f"Invalid month number provided in query: {month}. Please provide a number between 1 and 12."
            else:
                month_abbr_to_num = {
                    name.lower(): num
                    for num, name in enumerate(calendar.month_abbr)
                    if num > 0
                }
                month_name_to_num = {
                    name.lower(): num
                    for num, name in enumerate(calendar.month_name)
                    if num > 0
                }
                month_lower = month.lower()
                if month_lower in month_name_to_num:
                    month_num = month_name_to_num[month_lower]
                elif month_lower in month_abbr_to_num:
                    month_num = month_abbr_to_num[month_lower]
                else:
                    return f"Could not understand the month '{month}' from the query. Please use a full name (e.g., January), abbreviation (e.g., Jan), or number (e.g., 1)."

            df_filtered = df_filtered[df_filtered["Month"] == month_num]
            if df_filtered.empty:
                return f"No data found for the selected month ({month}) matching other criteria."

        except ValueError:
            return f"Could not parse the month '{month}' from the query. Please use a valid month name or number."

    if year:
        if (
            not isinstance(year, int) or year < 1000 or year > 3000
        ):  # Basic year validation
            return f"Invalid year provided in query: {year}. Please provide a 4-digit year."
        df_filtered = df_filtered[df_filtered["Year"] == year]
        if df_filtered.empty:
            return f"No data found for the year {year} matching other criteria."

    # Calculate OEE on the filtered data
    oee_results = calculate_oee(df_filtered)

    if "error" in oee_results:
        return oee_results["error"]

    # Format the response string for the LLM to present
    filter_summary = []
    if device_id:
        filter_summary.append(f"Device ID '{device_id}'")
    if location:
        filter_summary.append(f"Location '{location}'")
    if month:
        filter_summary.append(f"Month '{month}'")
    if year:
        filter_summary.append(f"Year {year}")

    if not filter_summary:
        filter_text = "the entire dataset"
    else:
        filter_text = "for " + ", ".join(filter_summary)

    response_string = f"""
Calculation successful {filter_text}.
Overall Equipment Effectiveness (OEE): {oee_results['oee']:.2f}%
Availability: {oee_results['availability']:.2f}%
Performance: {oee_results['performance']:.2f}%
Quality: {oee_results['quality']:.2f}%
Details: Scheduled Time={oee_results['total_scheduled_time_hrs']} hrs, Downtime={oee_results['total_downtime_hrs']} hrs, Total Units Produced={oee_results['total_units_produced']}, Good Units Produced={oee_results['total_good_units_produced']}, Average Ideal Cycle Time={oee_results['avg_ideal_cycle_time_sec']} sec.
"""
    return response_string.strip()


# --- LangChain Agent Setup (Cached) ---


@st.cache_resource
def initialize_agent(oee_df: pd.DataFrame):
    """Initializes the LangChain agent."""
    st.spinner("Initializing AI agent...")

    available_devices_list = sorted(oee_df["Device_ID"].unique().tolist())
    available_locations_list = sorted(oee_df["Location"].unique().tolist())
    available_years_list = sorted(oee_df["Year"].unique().tolist())
    available_years_range = (
        f"{min(available_years_list)} to {max(available_years_list)}"
        if available_years_list
        else "No data years available."
    )

    tools = [calculate_oee_chat_query_tool]

    system_prompt = f"""You are an AI assistant specialized in analyzing manufacturing data, specifically Overall Equipment Efficiency (OEE) for packaging devices.
The user is interacting with a Streamlit application that has two main sections:
1.  **Data Exploration & Plotting:** Where they can set filters using sidebar widgets (date range, device, location) and view plots based on the filtered data.
2.  **Conversational AI:** This chat interface, where you operate.

Your role in this chat interface is to answer specific questions about OEE based on explicit filters mentioned in the *user's chat query*. You use the `calculate_oee_chat_query` tool for this. The calculations you perform using the tool are based on the filters *from the chat*, and are independent of the filters the user might have set in the "Data Exploration & Plotting" section.

When a user asks for OEE in the chat, carefully extract the requested Device ID, Location, Month, and Year *from their message*.
Then, use the `calculate_oee_chat_query` tool with the extracted parameters.
If the user doesn't specify a filter in the chat message, use the tool without that filter.
If no filters are specified in the chat message at all, calculate OEE for the entire dataset using the tool.

After calling the tool, the tool will return a structured string with the calculation results or an error message.
Based on the tool's output, present the results in a clear, conversational manner, summarizing the calculated OEE, Availability, Performance, and Quality percentages.
If the tool's output indicates an error or no data, inform the user appropriately.

You can also briefly mention that they can use the "Data Exploration & Plotting" tab to visualize trends over time using interactive filters.

Available data covers the period from {available_years_range}.
Available Device IDs include (examples): {', '.join(available_devices_list[:10])}
Available Locations include: {', '.join(available_locations_list)}

Do not invent data or calculations. Only use the information returned by the `calculate_oee_chat_query` tool.
Maintain a helpful and professional tone.
"""

    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Setup chat history using Streamlit session state
    chat_history = ChatMessageHistory()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        else:
            chat_history.add_ai_message(msg["content"])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    agent_with_chat_history = (
        RunnablePassthrough.assign(chat_history=lambda x: chat_history.messages)
        | agent_executor
    )

    return (
        agent_with_chat_history,
        chat_history,
    )


# --- Streamlit App UI ---

st.title("📦 Packaging OEE Dashboard & Chatbot")

# --- Load data and initialize state ---
if "oee_df" not in st.session_state:
    st.session_state.oee_df = load_oee_data(DATA_FILE)
    if st.session_state.oee_df is None:
        st.stop()

# Get available filter options from the data
available_devices = sorted(st.session_state.oee_df["Device_ID"].unique().tolist())
available_locations = sorted(st.session_state.oee_df["Location"].unique().tolist())
min_date = st.session_state.oee_df["Date"].min()
max_date = st.session_state.oee_df["Date"].max()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can help you analyze packaging OEE data.",
        }
    ]


agent_with_chat_history, chat_history_obj = initialize_agent(st.session_state.oee_df)


# --- Sidebar Filters ---
st.sidebar.header("Data Filters")

# Date Range Slider
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD",
)

# Ensure date_range always has two elements
if len(date_range) == 2:
    selected_start_date, selected_end_date = date_range
elif len(date_range) == 1:
    selected_start_date = date_range[0]
    selected_end_date = max_date
else:
    selected_start_date = min_date
    selected_end_date = max_date


# Device Multiselect
selected_devices = st.sidebar.multiselect(
    "Select Device(s)",
    options=available_devices,
    default=[],  # Default to show all devices if none selected
)

# Location Multiselect
selected_locations = st.sidebar.multiselect(
    "Select Location(s)",
    options=available_locations,
    default=[],  # Default to show all locations if none selected
)

# --- Apply Filters and Aggregate for Display/Plotting ---
df_filtered_ui = filter_data(
    st.session_state.oee_df,
    selected_start_date,
    selected_end_date,
    selected_devices,
    selected_locations,
)

# --- Main Content Area (Tabs) ---
tab1, tab2 = st.tabs(["📊 Data Exploration & Plotting", "🤖 Conversational AI"])

with tab1:
    st.header("Data Exploration & Plotting")

    st.write("Use the filters in the sidebar to select the data you want to explore.")

    if df_filtered_ui.empty:
        st.warning("No data matches the selected filters.")
    else:
        st.write(f"Showing data for {len(df_filtered_ui)} rows matching filters.")

        # Select Metric to Plot
        metric_to_plot = st.selectbox(
            "Select Metric to Plot",
            options=[
                "OEE (%)",
                "Availability (%)",
                "Performance (%)",
                "Quality (%)",
                "total_scheduled_time_hrs",
                "total_downtime_hrs",
                "total_units_produced",
                "total_good_units_produced",
            ],
            index=0,  # Default to OEE
        )

        # Select Aggregation Level
        aggregation_level = st.radio(
            "Aggregate data by",
            options=["Daily", "Weekly", "Monthly"],
            index=2,  # Default to Monthly
            horizontal=True,
        )

        df_aggregated = aggregate_oee_data(df_filtered_ui, aggregation_level)

        if df_aggregated.empty or metric_to_plot not in df_aggregated.columns:
            st.warning(
                f"No aggregated data or '{metric_to_plot}' column available for plotting with current filters/aggregation."
            )
        else:
            st.subheader(f"{metric_to_plot} Trend ({aggregation_level} Aggregation)")

            if aggregation_level == "Daily":
                x_axis_col = "Date"

                df_aggregated[x_axis_col] = pd.to_datetime(df_aggregated[x_axis_col])
            elif aggregation_level == "Weekly":
                x_axis_col = "Year_Week"
            elif aggregation_level == "Monthly":
                x_axis_col = "Year_Month"
            else:
                x_axis_col = df_aggregated.columns[0]

            fig = px.line(
                df_aggregated,
                x=x_axis_col,
                y=metric_to_plot,
                title=f"{metric_to_plot} Over Time",
                labels={x_axis_col: aggregation_level},
                markers=True,
            )

            fig.update_layout(xaxis_title=aggregation_level, yaxis_title=metric_to_plot)
            fig.update_xaxes(tickangle=45)

            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox(f"Show Aggregated Data Table ({aggregation_level})"):
            st.subheader(f"Aggregated Data ({aggregation_level})")
            st.dataframe(df_aggregated)

        if st.checkbox("Show Raw Filtered Data Table"):
            st.subheader("Raw Filtered Data")
            st.dataframe(df_filtered_ui)


with tab2:
    st.header("Conversational AI")
    st.write(
        """
    Ask me specific questions about OEE metrics using filters like Device ID, Location, Month, or Year.
    Example: "What is the OEE for D01 in January 2024?" or "Tell me the Availability for Bangalore in 2025".
    The analysis here is based on the filters you provide in the chat query, independent of the sidebar filters.
    """
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "Ask about OEE (e.g., 'What's the OEE for D03 in February 2025?')"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_history_obj.add_user_message(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Invoke the agent with the latest input and the full chat history
                # The agent_with_chat_history runnable handles injecting chat_history into the prompt
                response = agent_with_chat_history.invoke({"input": prompt})
                assistant_response = response["output"]

            st.markdown(assistant_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            chat_history_obj.add_ai_message(assistant_response)
