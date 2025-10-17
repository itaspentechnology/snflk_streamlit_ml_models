import uuid

import streamlit as st
from snowflake.snowpark import Session


class Callbacks:
    @staticmethod
    def add_step():
        st.session_state["preprocessing_steps"].append(str(uuid.uuid4()))

    @staticmethod
    def remove_step(step_id):
        if step_id in st.session_state["preprocessing_steps"]:
            st.session_state["preprocessing_steps"].remove(step_id)

    @staticmethod
    def eda_mode_switch(mode: bool):
        st.session_state["eda_mode"] = st.session_state[mode]

    @staticmethod
    def set_dataset(session: Session, db: str, schema: str, table_key: str) -> None:
        try:
            # Get the actual table name from session state
            table = st.session_state.get(table_key)
            if not table:
                st.error(f"Table not found in session state key: {table_key}")
                return
                
            # Properly quote identifiers for Snowflake
            db_quoted = f'"{db}"'
            schema_quoted = f'"{schema}"'
            table_quoted = f'"{table}"'
            fully_qualified_name = f"{db_quoted}.{schema_quoted}.{table_quoted}"
            
            # Store the fully qualified table name
            st.session_state["full_qualified_table_nm"] = fully_qualified_name
            
            # Create the dataset table reference
            st.session_state["dataset"] = session.table(fully_qualified_name)
            
            # Clear any previous column selections when dataset changes
            st.session_state["selected_features"] = []
            st.session_state["selected_target"] = None
            
            st.success(f"✅ Dataset loaded: {fully_qualified_name}")
            
        except Exception as e:
            st.error(f"❌ Failed to set dataset: {str(e)}")
            st.error(f"Parameters: db={db}, schema={schema}, table_key={table_key}")
            st.session_state["dataset"] = None

    @staticmethod
    def set_workflow(workflow_id: int):
        if workflow_id == 0:
            st.cache_data.clear()
        st.session_state["workflow"] = workflow_id

    @staticmethod
    def set_timeseries_seq(seq_id: int):
        # Manages workflow state for timeseries page
        st.session_state["timeseries_deploy_sequence"] = seq_id
