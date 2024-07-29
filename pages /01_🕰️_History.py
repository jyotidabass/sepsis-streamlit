import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer

import asyncio
from aiocache import cached, Cache

import pandas as pd

from utils.navigation import navigation
from utils.footer import footer

from config import HISTORY_FILE

# Set page configuration
st.set_page_config(
    page_title='History Page',
    page_icon='🕰️',
    layout="wide",
    initial_sidebar_state='auto'
)


# @st.cache_data(show_spinner="Getting history of predictions...")
@cached(ttl=10, cache=Cache.MEMORY, namespace='streamlit_savedataset')
async def get_history_data():
    try:
        df_history = pd.read_csv(HISTORY_FILE, index_col=0)
        df_history['time_of_prediction'] = [timestamps[0]
                                            for timestamps in df_history['time_of_prediction'].str.split('.')[0:]]

        df_history['time_of_prediction'] = pd.to_datetime(
            df_history['time_of_prediction'])
    except Exception as e:
        df_history = None

    return df_history


async def main():
    st.title("Prediction History 🕰️")

    # Navigation
    await navigation()

    df_history = await get_history_data()

    if df_history is not None:
        df_history_explorer = dataframe_explorer(df_history, case=False)

        st.dataframe(df_history_explorer)
    else:
        st.info(
            "There is no history file yet. Make a prediction.", icon='ℹ️')

    # Add footer
    await footer()


if __name__ == "__main__":
    asyncio.run(main())
