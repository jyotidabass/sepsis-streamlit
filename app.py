import os
import time
import httpx
import string
import random
import datetime as dt
from dotenv import load_dotenv

import streamlit as st
import extra_streamlit_components as stx

import asyncio
from aiocache import cached, Cache

import pandas as pd
from typing import Optional, Callable

from config import ENV_PATH, BEST_MODELS, TEST_FILE, TEST_FILE_URL, HISTORY_FILE, markdown_table_all

from utils.navigation import navigation
from utils.footer import footer
from utils.janitor import Janitor


# Load ENV
load_dotenv(ENV_PATH)  # API_URL

# Set page configuration
st.set_page_config(
    page_title="Homepage",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state='auto'
)


@cached(ttl=10, cache=Cache.MEMORY, namespace='streamlit_savedataset')
# @st.cache_data(show_spinner="Saving datasets...") # Streamlit cache is yet to support async functions
async def save_dataset(df: pd.DataFrame, filepath, csv=True) -> None:
    async def save(df: pd.DataFrame, file):
        return df.to_csv(file, index=False) if csv else df.to_excel(file, index=False)

    async def read(file):
        return pd.read_csv(file) if csv else pd.read_excel(file)

    async def same_dfs(df: pd.DataFrame, df2: pd.DataFrame):
        return df.equals(df2)

    if not os.path.isfile(filepath):  # Save if file does not exists
        await save(df, filepath)
    else:  # Save if data are not same
        df_old = await read(filepath)
        if not await same_dfs(df, df_old):
            await save(df, filepath)


@cached(ttl=10, cache=Cache.MEMORY, namespace='streamlit_testdata')
async def get_test_data():
    try:
        df_test_raw = pd.read_csv(TEST_FILE_URL)
        await save_dataset(df_test_raw, TEST_FILE, csv=True)
    except Exception:
        df_test_raw = pd.read_csv(TEST_FILE)

    # Some house keeping, clean df
    df_test = df_test_raw.copy()
    janitor = Janitor()
    df_test = janitor.clean_dataframe(df_test)  # Cleaned

    return df_test_raw, df_test


# Function for selecting models
async def select_model() -> str:
    col1, _ = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            'Select a model', options=BEST_MODELS, key='selected_model')

    return selected_model


async def endpoint(model: str) -> str:
    api_url = os.getenv("API_URL")
    model_endpoint = f"{api_url}={model}"
    return model_endpoint


# Function for making prediction
async def make_prediction(model_endpoint) -> Optional[pd.DataFrame]:

    test_data = await get_test_data()
    _, df_test = test_data

    df: pd.DataFrame = None
    search_patient = st.session_state.get('search_patient', False)
    search_patient_id = st.session_state.get('search_patient_id', False)
    manual_patient_id = st.session_state.get('manual_patient_id', False)
    if isinstance(search_patient_id, str) and search_patient_id:  # And not empty string
        search_patient_id = [search_patient_id]
    if search_patient and search_patient_id:  # Search Form df and a patient was selected
        mask = df_test['id'].isin(search_patient_id)
        df_form = df_test[mask]
        df = df_form.copy()
    elif not (search_patient or search_patient_id) and manual_patient_id:  # Manual form df
        columns = ['manual_patient_id', 'prg', 'pl', 'pr', 'sk',
                   'ts', 'm11', 'bd2', 'age', 'insurance']
        data = {c: [st.session_state.get(c)] for c in columns}
        data['insurance'] = [1 if i == 'Yes' else 0 for i in data['insurance']]

        # Make a DataFrame
        df = pd.DataFrame(data).rename(
            columns={'manual_patient_id': 'id'})
        columns_int = ['prg', 'pl', 'pr', 'sk', 'ts', 'age']
        columns_float = ['m11', 'bd2']

        df[columns_int] = df[columns_int].astype(int)
        df[columns_float] = df[columns_float].astype(float)
    else:  # Form did not send a patient
        message = 'You must choose valid patient(s) from the select box.'
        icon = '😞'
        st.toast(message, icon=icon)
        st.warning(message, icon=icon)

    if df is not None:
        try:
            # JSON data
            data = df.to_dict(orient='list')

            # Send POST request with JSON data using the json parameter
            async with httpx.AsyncClient() as client:
                response = await client.post(model_endpoint, json=data, timeout=30)
                response.raise_for_status()  # Ensure we catch any HTTP errors

            if (response.status_code == 200):
                pred_prob = (response.json()['result'])
                prediction = pred_prob['prediction'][0]
                probability = pred_prob['probability'][0]

                # Store results in session state
                st.session_state['prediction'] = prediction
                st.session_state['probability'] = probability
                df['prediction'] = prediction
                df['probability (%)'] = probability
                df['time_of_prediction'] = pd.Timestamp(dt.datetime.now())
                df['model_used'] = st.session_state['selected_model']

                df.to_csv(HISTORY_FILE, mode='a',
                          header=not os.path.isfile(HISTORY_FILE))
        except Exception as e:
            st.error(f'😞 Unable to connect to the API server. {e}')

    return df


async def convert_string(df: pd.DataFrame, string: str) -> str:
    return string.upper() if all(col.isupper() for col in df.columns) else string


async def make_predictions(model_endpoint, df_uploaded=None, df_uploaded_clean=None) -> Optional[pd.DataFrame]:

    df: pd.DataFrame = None
    search_patient = st.session_state.get('search_patient', False)
    patient_id_bulk = st.session_state.get('patient_id_bulk', False)
    upload_bulk_predict = st.session_state.get('upload_bulk_predict', False)
    if search_patient and patient_id_bulk:  # Search Form df and a patient was selected
        _, df_test = await get_test_data()
        mask = df_test['id'].isin(patient_id_bulk)
        df_bulk: pd.DataFrame = df_test[mask]
        df = df_bulk.copy()

    elif not (search_patient or patient_id_bulk) and upload_bulk_predict:  # Upload widget df
        df = df_uploaded_clean.copy()
    else:  # Form did not send a patient
        message = 'You must choose valid patient(s) from the select box.'
        icon = '😞'
        st.toast(message, icon=icon)
        st.warning(message, icon=icon)

    if df is not None:  # df should be set by form input or upload widget
        try:
            # JSON data
            data = df.to_dict(orient='list')

            # Send POST request with JSON data using the json parameter
            async with httpx.AsyncClient() as client:
                response = await client.post(model_endpoint, json=data, timeout=30)
                response.raise_for_status()  # Ensure we catch any HTTP errors

            if (response.status_code == 200):
                pred_prob = (response.json()['result'])
                predictions = pred_prob['prediction']
                probabilities = pred_prob['probability']

                # Add columns sepsis, probability, time, and model used to uploaded df and form df

                async def add_columns(df):
                    df[await convert_string(df, 'sepsis')] = predictions
                    df[await convert_string(df, 'probability_(%)')] = probabilities
                    df[await convert_string(df, 'time_of_prediction')
                       ] = pd.Timestamp(dt.datetime.now())
                    df[await convert_string(df, 'model_used')
                       ] = st.session_state['selected_model']

                    return df

                # Form df if search patient is true or df from Uploaded data
                if search_patient:
                    df = await add_columns(df)

                    df.to_csv(HISTORY_FILE, mode='a', header=not os.path.isfile(
                        HISTORY_FILE))  # Save only known patients

                else:
                    df = await add_columns(df_uploaded)  # Raw, No cleaning

                # Store df with prediction results in session state
                st.session_state['bulk_prediction_df'] = df
        except Exception as e:
            st.error(f'😞 Unable to connect to the API server. {e}')

    return df


def on_click(func: Callable, model_endpoint: str):
    async def handle_click():
        await func(model_endpoint)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(handle_click())
    loop.close()


async def search_patient_form(model_endpoint: str) -> None:
    test_data = await get_test_data()
    _, df_test = test_data

    patient_ids = df_test['id'].unique().tolist()+['']
    if st.session_state['sidebar'] == 'single_prediction':
        with st.form('search_patient_id_form'):
            col1, _ = st.columns(2)
            with col1:
                st.write('#### Patient ID 🤒')
                st.selectbox(
                    'Search a patient', options=patient_ids, index=len(patient_ids)-1, key='search_patient_id')
            st.form_submit_button('Predict', type='primary', on_click=on_click, kwargs=dict(
                func=make_prediction, model_endpoint=model_endpoint))
    else:
        with st.form('search_patient_id_bulk_form'):
            col1, _ = st.columns(2)
            with col1:
                st.write('#### Patient ID 🤒')
                st.multiselect(
                    'Search a patient', options=patient_ids, default=None, key='patient_id_bulk')
            st.form_submit_button('Predict', type='primary', on_click=on_click, kwargs=dict(
                func=make_predictions, model_endpoint=model_endpoint))


async def gen_random_patient_id() -> str:
    numbers = ''.join(random.choices(string.digits, k=6))
    letters = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"ICU{numbers}-gen-{letters}"


async def manual_patient_form(model_endpoint) -> None:
    with st.form('manual_patient_form'):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write('### Patient Demographics 🛌')
            st.text_input(
                'ID', value=await gen_random_patient_id(), key='manual_patient_id')
            st.number_input('Age: patients age (years)', min_value=0,
                            max_value=100, step=1, key='age')
            st.selectbox('Insurance: If a patient holds a valid insurance card', options=[
                'Yes', 'No'], key='insurance')

        with col2:
            st.write('### Vital Signs 🩺')
            st.number_input('BMI (weight in kg/(height in m)^2', min_value=10.0,
                            format="%.2f", step=1.00, key='m11')
            st.number_input(
                'Blood Pressure (mm Hg)', min_value=10.0, format="%.2f", step=1.00, key='pr')
            st.number_input(
                'PRG (plasma glucose)', min_value=10.0, format="%.2f", step=1.00, key='prg')

        with col3:
            st.write('### Blood Work 💉')
            st.number_input(
                'PL: Blood Work Result-1 (mu U/ml)', min_value=10.0, format="%.2f", step=1.00, key='pl')
            st.number_input(
                'SK: Blood Work Result 2 (mm)', min_value=10.0, format="%.2f", step=1.00, key='sk')
            st.number_input(
                'TS: Blood Work Result-3 (mu U/ml)', min_value=10.0, format="%.2f", step=1.00, key='ts')
            st.number_input(
                'BD2: Blood Work Result-4 (mu U/ml)', min_value=10.0, format="%.2f", step=1.00, key='bd2')

        st.form_submit_button('Predict', type='primary', on_click=on_click, kwargs=dict(
            func=make_prediction, model_endpoint=model_endpoint))


async def do_single_prediction(model_endpoint: str) -> None:
    if st.session_state.get('search_patient', False):
        await search_patient_form(model_endpoint)
    else:
        await manual_patient_form(model_endpoint)


async def show_prediction() -> None:
    final_prediction = st.session_state.get('prediction', None)
    final_probability = st.session_state.get('probability', None)

    if final_prediction is None:
        st.markdown('#### Prediction will show below! 🔬')
        st.divider()
    else:
        st.markdown('#### Prediction! 🔬')
        st.divider()
        if final_prediction.lower() == 'positive':
            st.toast("Sepsis alert!", icon='🦠')
            message = f"It is **{final_probability:.2f} %** likely that the patient will develop **sepsis.**"
            st.warning(message, icon='😞')
            time.sleep(5)
            st.toast(message)
        else:
            st.toast("Continous monitoring", icon='🔬')
            message = f"The patient will **not** develop sepsis with a likelihood of **{final_probability:.2f}%**."
            st.success(message, icon='😊')
            time.sleep(5)
            st.toast(message)

    # Set prediction and probability to None
    st.session_state['prediction'] = None
    st.session_state['probability'] = None


# @st.cache_data(show_spinner=False) Caching results from async functions buggy
async def convert_df(df: pd.DataFrame):
    return df.to_csv(index=False)


async def bulk_upload_widget(model_endpoint: str) -> None:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel File", type=['csv', 'xls', 'xlsx'])

    uploaded = uploaded_file is not None

    upload_bulk_predict = st.button('Predict', type='primary',
                                    help='Upload a csv/excel file to make predictions', disabled=not uploaded, key='upload_bulk_predict')
    df = None
    if upload_bulk_predict and uploaded:
        df_test_raw, _ = await get_test_data()
        # Uploadfile is a "file-like" object is accepted
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                df = pd.read_excel(uploaded_file)

            df_columns = set(df.columns)
            df_test_columns = set(df_test_raw.columns)
            df_schema = df.dtypes
            df_test_schema = df_test_raw.dtypes

            if df_columns != df_test_columns or not df_schema.equals(df_test_schema):
                df = None
                raise Exception
            else:
                # Clean dataframe
                janitor = Janitor()
                df_clean = janitor.clean_dataframe(df)

                df = await make_predictions(
                    model_endpoint, df_uploaded=df, df_uploaded_clean=df_clean)

        except Exception:
            st.subheader('Data template')
            data_template = df_test_raw[:3]
            st.dataframe(data_template)
            csv = await convert_df(data_template)
            message_1 = 'Upload a valid csv or excel file.'
            message_2 = f"{message_1.split('.')[0]} with the columns and schema of the above data template."
            icon = '😞'
            st.toast(message_1, icon=icon)

            st.download_button(
                label='Download template',
                data=csv,
                file_name='Data template.csv',
                mime="text/csv",
                type='secondary',
                key='download-data-template'
            )
            st.info('Download the above template for use as a baseline structure.')

            # Display explander to show the data dictionary
            with st.expander("Expand to see the data dictionary", icon="💡"):
                st.subheader("Data dictionary")
                st.markdown(markdown_table_all)
            st.warning(message_2, icon=icon)

    return df


async def do_bulk_prediction(model_endpoint: str) -> None:
    if st.session_state.get('search_patient', False):
        await search_patient_form(model_endpoint)
    else:
        # File uploader
        await bulk_upload_widget(model_endpoint)


async def show_bulk_predictions(df: pd.DataFrame) -> None:
    if df is not None:
        st.subheader("Bulk predictions 🔮", divider=True)
        st.dataframe(df.astype(str))

        csv = await convert_df(df)
        message = 'The predictions are ready for download.'
        icon = '⬇️'
        st.toast(message, icon=icon)
        st.info(message, icon=icon)
        st.download_button(
            label='Download predictions',
            data=csv,
            file_name='Bulk prediction.csv',
            mime="text/csv",
            type='secondary',
            key='download-bulk-prediction'
        )

        # Set bulk prediction df to None
        st.session_state['bulk_prediction_df'] = None


async def sidebar(sidebar_type: str) -> st.sidebar:
    return st.session_state.update({'sidebar': sidebar_type})


async def main():
    st.title("🤖 Predict Sepsis 🦠")

    # Navigation
    await navigation()

    st.sidebar.toggle("Looking for a patient?", value=st.session_state.get(
        'search_patient', False), key='search_patient')

    selected_model = await select_model()
    model_endpoint = await endpoint(selected_model)

    selected_predict_tab = st.session_state.get('selected_predict_tab')
    default = 1 if selected_predict_tab is None else selected_predict_tab

    with st.spinner('A little house keeping...'):
        time.sleep(st.session_state.get('sleep', 1.5))
        chosen_id = stx.tab_bar(data=[
            stx.TabBarItemData(id=1, title='🔬 Predict', description=''),
            stx.TabBarItemData(id=2, title='🔮 Bulk predict',
                               description=''),
        ], default=default)
        st.session_state['sleep'] = 0

    if chosen_id == '1':
        await sidebar('single_prediction')
        await do_single_prediction(model_endpoint)
        await show_prediction()

    elif chosen_id == '2':
        await sidebar('bulk_prediction')
        df_with_predictions = await do_bulk_prediction(model_endpoint)
        if df_with_predictions is None:
            df_with_predictions = st.session_state.get(
                'bulk_prediction_df', None)
        await show_bulk_predictions(df_with_predictions)

    # Add footer
    await footer()


if __name__ == "__main__":
    asyncio.run(main())
