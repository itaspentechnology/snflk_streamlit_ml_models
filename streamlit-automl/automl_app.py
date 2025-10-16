from pathlib import Path

import streamlit as st
st.write("Streamlit version:", st.__version__)
from callbacks import Callbacks
from ml_modeling import AutoMLModeling
from ml_ops import ModelReg
from utils import initialize_session_state, set_png_as_page_bg

st.set_page_config(
    layout="wide",
    page_title="Snowflake Auto ML",
    page_icon="❄️",
    initial_sidebar_state=st.session_state["sidebar_state"],
)

st.write("Streamlit version:", st.__version__)
st.write("✅ App loaded successfully")

if "sidebar_state" not in st.session_state:
    st.session_state["sidebar_state"] = "collapsed"

initialize_session_state()

with open(Path(__file__).parent / "styles" / "css_bootstrap.html", "r") as r:
    styles = r.read()
    st.markdown(styles, unsafe_allow_html=True)
    if "css_styles" not in st.session_state:
        st.session_state["css_styles"] = styles

st.markdown(
    set_png_as_page_bg(Path(__file__).parent / "resources" / "background.png"),
    unsafe_allow_html=True,
)

with st.container(height=135, border=False):
    st.title("ML Sidekick")
    st.caption("A no-code application for leveraging the snowflake-ml-python package")
    if st.session_state["workflow"] == 0:
        with st.container(border=False, height=51):
            with st.popover("Create Project", use_container_width=True):
                st.button(
                    "ML Model",
                    use_container_width=True,
                    on_click=Callbacks.set_workflow,
                    args=[1],
                )
    else:
        with st.container(border=False, height=49):
            st.button("←", on_click=Callbacks.set_workflow, args=[0])


if st.session_state["logged_in"]:
    session = st.session_state["session"]

    # Workflow 0 = Registry
    if st.session_state["workflow"] == 0:
        model_reg = ModelReg(session=session)
        model_reg.render_registry()

        # Trigger model test
        if st.button("Run Model Test"):
            st.session_state.show_model_test = True

        # Show model test UI
        if st.session_state.get("show_model_test", False):
            with st.expander("Model Test", expanded=True):
                if st.session_state.get("dataset"):
                    tbl_name = f"{st.session_state.get('aml_mpa.sel_db','-')}.{st.session_state.get('aml_mpa.sel_schema','-')}.{st.session_state.get('aml_mpa.sel_table','-')}"
                    model_reg.call_test_models(
                        df=st.session_state["dataset"],
                        tbl_name=tbl_name,
                    )
                else:
                    st.warning("You must select a dataset before running model tests.")

    # Workflow 1 = ML Builder
    if st.session_state["workflow"] == 1:
        AutoMLModeling(session=session).render_ml_builder()

