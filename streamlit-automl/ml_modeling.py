from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from callbacks import Callbacks
from code_exporter import create_notebook
from common import get_col_types
from histograms import AutoHistogram, show_histogram_dialog
from model_metrics import ModelMetrics
from preprocessing import AutoPreProcessor
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
)
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session
from streamlit.components.v1 import html
from utils import get_databases, get_feature_importance_df, get_schemas, get_tables

AVATAR_PATH = str(Path(__file__).parent / "resources" / "Snowflake_ICON_Chat.png")


def set_state(state: int):
    if "recorded_steps" not in st.session_state:
        st.session_state["recorded_steps"] = []
        
    st.session_state["app_state"] = state
    if state not in st.session_state["recorded_steps"]:
        st.session_state["recorded_steps"].append(state)

    st.write(f"🔄 App state set to: {state}")
    st.write(f"✅ Recorded steps: {st.session_state['recorded_steps']}")

def create_metric_card(label, value):
    return f"""
             <span class="property_container">
                <span class="property_title">{label}</span>
                <span class="property_pill_current">{value}</span>
            </span>
                """


class TopMenu:
    def __init__(self) -> None:
        header_menu_c = st.container(border=False, height=60)
        header_menu = header_menu_c.columns(3)
        header_menu[0].button(
            "Select Dataset",
            key="btn_select",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 0 else "secondary",
            disabled=0 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[0],
        )
        header_menu[1].button(
            "Pre-Processing",
            key="btn_preprocess",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 1 else "secondary",
            disabled=1 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[1],
        )
        header_menu[2].button(
            "Modeling",
            key="btn_modeling",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 2 else "secondary",
            disabled=2 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[2],
        )


class AutoMLModeling:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.step_data = {}

    def render_ml_builder(self):
        with st.container(height=101, border=False):
            if st.button("🔍"):
                if st.session_state.get("dataset"):
                    st.session_state.show_histogram = True
                else:
                    st.toast("You must select a dataset before.")

        if st.session_state.get("show_histogram", False):
            with st.expander("Exploratory Data Analysis", expanded=True):
                AutoHistogram(
                    df=st.session_state["dataset"],
                    name=f"{st.session_state.get('aml_mpa.sel_db','-')}.{st.session_state.get('aml_mpa.sel_schema','-')}.{st.session_state.get('aml_mpa.sel_table','-')}",
                ).render_grid()


        TopMenu()
        if st.session_state["app_state"] > -1:
            dataset_chat = st.chat_message(
                name="assistant",
                avatar=AVATAR_PATH,
            )
            with dataset_chat:
                st.write("Let's begin by selecting a source dataset.")
                with st.popover(
                    "Dataset Selection",
                    disabled=not (st.session_state["app_state"] == 0),
                    use_container_width=True,
                ):
                    context_menu_cols = st.columns((1, 2))
                    databases = get_databases(self.session)
                    db = context_menu_cols[0].selectbox(
                        "Source Database",
                        index=None,
                        options=databases,
                        placeholder="Select a database",
                        key="aml_mpa.sel_db",
                    )
                    if db:
                        st.session_state["context"]["database"] = db
                        st.session_state["context"]["schemas"] = get_schemas(
                            self.session, db
                        )
                    else:
                        st.session_state["context"]["schemas"] = []

                    schema = context_menu_cols[0].selectbox(
                        "Source Schema",
                        st.session_state["context"].get("schemas", []),
                        index=None,
                        placeholder="Select a schema",
                        key="aml_mpa.sel_schema",
                    )

                    if schema:
                        st.session_state["context"]["tables"] = get_tables(
                            self.session, db, schema
                        )
                    else:
                        st.session_state["context"]["tables"] = []

                    table = context_menu_cols[0].selectbox(
                        "Source Table",
                        st.session_state["context"].get("tables", []),
                        index=None,
                        placeholder="Select a table",
                        key="aml_mpa.sel_table",
                        on_change=Callbacks.set_dataset,
                        args=[self.session, db, schema, "aml_mpa.sel_table"],
                    )
                    if all([db, schema, table]):
                        context_menu_cols[1].dataframe(
                            st.session_state["dataset"].limit(5),
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.container(border=False, height=10)
                        dataset_chat_cols = dataset_chat.columns(3)
                        dataset_chat_cols[2].button(
                            "Next",
                            use_container_width=True,
                            type="primary",
                            on_click=set_state,
                            args=[1],
                            disabled=not (st.session_state["app_state"] == 0),
                        )

        if 1 in st.session_state["recorded_steps"] and st.session_state.get("dataset") is not None:
            preproc_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with preproc_chat:
                st.write("Now, let's pre-process the source dataset.")
                st.info(
                    "Click the :mag: icon above to dig deeper into your data. If you have null values, add a **SimpleImputer** step. If you have string features, add a **OneHotEncoder** step. "
                )
                
                # Validate dataset exists and is accessible
                try:
                    if st.session_state["dataset"] is None:
                        st.error("❌ No dataset loaded. Please go back to step 1 and select a dataset.")
                        if st.button("← Back to Dataset Selection", key="back_to_dataset"):
                            set_state(0)
                            st.rerun()
                        return
                        
                    # Test dataset accessibility
                    dataset_columns = st.session_state["dataset"].columns
                    if not dataset_columns:
                        st.error("❌ Dataset has no columns or is not accessible.")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Dataset access error: {str(e)}")
                    st.error("This might be due to:")
                    st.error("- Snowflake connection issues")
                    st.error("- Table permissions")
                    st.error("- Invalid table reference")
                    
                    recovery_cols = st.columns(2)
                    if recovery_cols[0].button("🔄 Retry Dataset Access", key="retry_dataset"):
                        st.rerun()
                    if recovery_cols[1].button("← Back to Dataset Selection", key="back_to_dataset_error"):
                        set_state(0)
                        st.rerun()
                    return
            preproc_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with preproc_chat:
                st.write("Now, let's pre-process the source dataset.")
                st.info(
                    "Click the :mag: icon above to dig deeper into your data. If you have null values, add a **SimpleImputer** step. If you have string features, add a **OneHotEncoder** step. "
                )
                with st.expander(
                    "Pre-Processing Options",
                    expanded=st.session_state["app_state"] == 1,
                ):
                    st.header("Preprocessing Options")
                    st.caption(":red[*] required fields")
                    
                    # Enhanced Debug: Show current dataset state with recovery options
                    debug_enabled = st.checkbox("Show Debug Info", key="debug_cols")
                    if debug_enabled:
                        st.write(f"🔍 Dataset columns: {st.session_state['dataset'].columns}")
                        st.write(f"🔍 Dataset shape: {st.session_state['dataset'].count()} rows")
                        
                        # Add refresh button for dataset state
                        if st.button("🔄 Refresh Dataset State", key="refresh_dataset"):
                            try:
                                # Force refresh the dataset from the source
                                db = st.session_state.get('aml_mpa.sel_db')
                                schema = st.session_state.get('aml_mpa.sel_schema') 
                                table = st.session_state.get('aml_mpa.sel_table')
                                if all([db, schema, table]):
                                    st.session_state["dataset"] = self.session.table(f"{db}.{schema}.{table}")
                                    st.success("✅ Dataset state refreshed successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Cannot refresh - missing table information")
                            except Exception as e:
                                st.error(f"❌ Failed to refresh dataset: {str(e)}")
                    
                    # Store current columns for validation
                    current_dataset_columns = st.session_state["dataset"].columns
                    
                    feature_cols = st.multiselect(
                        "Select the feature columns.:red[*]",
                        options=current_dataset_columns,
                        key="selected_features"
                    )
                    target_col = st.selectbox(
                        "Select the target column.:red[*]",
                        current_dataset_columns,
                        index=None,
                        key="selected_target"
                    )
                    
                    # Enhanced Debug: Show what was selected with validation
                    if debug_enabled and (feature_cols or target_col):
                        st.write(f"🔍 Selected features: {feature_cols} (type: {type(feature_cols)})")
                        st.write(f"🔍 Selected target: {target_col} (type: {type(target_col)})")
                        
                        # Real-time validation
                        if feature_cols:
                            invalid_features = [col for col in feature_cols if col not in current_dataset_columns]
                            if invalid_features:
                                st.error(f"❌ Invalid feature columns detected: {invalid_features}")
                            else:
                                st.success(f"✅ All {len(feature_cols)} feature columns are valid")
                        
                        if target_col and target_col not in current_dataset_columns:
                            st.error(f"❌ Invalid target column: {target_col}")
                        elif target_col:
                            st.success(f"✅ Target column '{target_col}' is valid")
                    
                    # Show current selections summary
                    if feature_cols or target_col:
                        with st.expander("📋 Current Column Selections", expanded=False):
                            if feature_cols:
                                st.write(f"**Selected Features ({len(feature_cols)}):** {', '.join(feature_cols)}")
                            else:
                                st.write("**Selected Features:** None")
                            
                            if target_col:
                                st.write(f"**Selected Target:** {target_col}")
                            else:
                                st.write("**Selected Target:** None")
                            
                            # Show session state values for comparison
                            if debug_enabled:
                                st.write("**Session State Values:**")
                                st.write(f"- selected_features: {st.session_state.get('selected_features', 'Not set')}")
                                st.write(f"- selected_target: {st.session_state.get('selected_target', 'Not set')}")
                    
                    # Ensure selections are saved to session state immediately
                    if feature_cols != st.session_state.get("selected_features"):
                        st.session_state["selected_features"] = feature_cols
                    if target_col != st.session_state.get("selected_target"):
                        st.session_state["selected_target"] = target_col

                    if feature_cols and target_col:
                        # Enhanced validation with automatic recovery
                        try:
                            # Step 1: Refresh and validate current dataset state
                            current_columns = st.session_state["dataset"].columns
                            
                            # Debug output for dynamic selection
                            if st.session_state.get("debug_cols", False):
                                st.write(f"🔍 Current dataset columns: {current_columns}")
                                st.write(f"🔍 Feature columns from UI: {feature_cols}")
                                st.write(f"🔍 Target column from UI: {target_col}")
                            
                            # Step 2: Comprehensive column validation with auto-correction
                            validated_features = []
                            feature_errors = []
                            
                            for i, col in enumerate(feature_cols):
                                if col in current_columns:
                                    validated_features.append(col)
                                else:
                                    feature_errors.append(col)
                                    if st.session_state.get("debug_cols", False):
                                        st.error(f"❌ Feature column #{i+1} '{col}' not found!")
                                        st.write(f"   Column repr: {repr(col)}")
                                        # Find closest matches
                                        matches = [c for c in current_columns if col.lower() in c.lower() or c.lower() in col.lower()]
                                        if matches:
                                            st.write(f"   Closest matches: {matches}")
                                            # Auto-suggest replacement
                                            if len(matches) == 1:
                                                if st.button(f"🔧 Replace '{col}' with '{matches[0]}'?", key=f"fix_feature_{i}"):
                                                    # Update the feature selection in session state
                                                    new_features = feature_cols.copy()
                                                    new_features[i] = matches[0]
                                                    st.session_state["selected_features"] = new_features
                                                    st.rerun()
                            
                            # Target column validation with auto-correction
                            target_valid = target_col in current_columns
                            if not target_valid and st.session_state.get("debug_cols", False):
                                st.error(f"❌ Target column '{target_col}' not found!")
                                st.write(f"   Column repr: {repr(target_col)}")
                                # Find closest matches for target
                                target_matches = [c for c in current_columns if target_col.lower() in c.lower() or c.lower() in target_col.lower()]
                                if target_matches:
                                    st.write(f"   Closest matches: {target_matches}")
                                    if len(target_matches) == 1:
                                        if st.button(f"🔧 Replace target '{target_col}' with '{target_matches[0]}'?", key="fix_target"):
                                            st.session_state["selected_target"] = target_matches[0]
                                            st.rerun()
                            
                            # Step 3: Proceed only if validation passes or show recovery options
                            if feature_errors or not target_valid:
                                st.error("🚫 Column validation failed. Please fix the issues above or try these recovery options:")
                                
                                recovery_cols = st.columns(3)
                                
                                # Option 1: Clear selections
                                if recovery_cols[0].button("🗑️ Clear All Selections", key="clear_selections"):
                                    st.session_state["selected_features"] = []
                                    st.session_state["selected_target"] = None
                                    st.rerun()
                                
                                # Option 2: Reset to available columns only
                                if recovery_cols[1].button("🔄 Reset to Valid Columns", key="reset_valid"):
                                    # Keep only valid features
                                    valid_features = [col for col in feature_cols if col in current_columns]
                                    st.session_state["selected_features"] = valid_features
                                    # Reset target if invalid
                                    if target_col not in current_columns:
                                        st.session_state["selected_target"] = None
                                    st.rerun()
                                
                                # Option 3: Refresh dataset and retry
                                if recovery_cols[2].button("🔄 Refresh Dataset & Retry", key="refresh_retry"):
                                    try:
                                        db = st.session_state.get('aml_mpa.sel_db')
                                        schema = st.session_state.get('aml_mpa.sel_schema') 
                                        table = st.session_state.get('aml_mpa.sel_table')
                                        if all([db, schema, table]):
                                            st.session_state["dataset"] = self.session.table(f"{db}.{schema}.{table}")
                                            st.success("✅ Dataset refreshed!")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to refresh: {str(e)}")
                                
                                st.stop()  # Stop processing until issues are resolved
                            
                            # Step 4: All validations passed - proceed with processing
                            if st.session_state.get("debug_cols", False):
                                st.success(f"✅ All validations passed!")
                                st.write(f"🔍 Using features: {validated_features}")
                                st.write(f"🔍 Using target: {target_col}")
                            
                            # Save validated selections to session state for use in modeling
                            st.session_state["selected_features"] = validated_features
                            st.session_state["selected_target"] = target_col
                            
                            # Use validated columns for selection
                            selected_columns = validated_features + [target_col]
                            t_sub = st.session_state["dataset"].select(selected_columns)
                            
                        except Exception as e:
                            st.error("🚨 **Critical Error During Column Processing**")
                            st.error(f"**Error:** {str(e)}")
                            st.error(f"**Type:** {type(e).__name__}")
                            
                            with st.expander("🔍 Debugging Information", expanded=True):
                                st.write(f"**Feature columns from UI:** {feature_cols}")
                                st.write(f"**Target column from UI:** {target_col}")
                                st.write(f"**Dataset columns:** {st.session_state['dataset'].columns}")
                                st.write(f"**Dataset type:** {type(st.session_state['dataset'])}")
                                
                                # Show session state for debugging
                                st.write("**Session State Keys:**")
                                relevant_keys = [k for k in st.session_state.keys() if 'sel_' in k or 'dataset' in k]
                                for key in relevant_keys:
                                    st.write(f"  - {key}: {st.session_state.get(key)}")
                            
                            # Recovery options
                            st.error("**Recovery Options:**")
                            recovery_error_cols = st.columns(2)
                            
                            if recovery_error_cols[0].button("🔄 Reload Application", key="reload_app"):
                                # Clear problematic session state
                                for key in ['selected_features', 'selected_target', 'dataset']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()
                            
                            if recovery_error_cols[1].button("📋 Copy Error Details", key="copy_error"):
                                error_details = f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}
Feature Columns: {feature_cols}
Target Column: {target_col}
Dataset Columns: {st.session_state['dataset'].columns}
                                """
                                st.code(error_details)
                            
                            st.stop()
                        cat_cols = get_col_types(t_sub, "string")
                        num_cols = get_col_types(t_sub, "numeric")

                        if target_col in cat_cols:
                            cat_cols.remove(target_col)
                        if target_col in num_cols:
                            num_cols.remove(target_col)

                        preprocessor_options = [
                            "SimpleImputer (numeric)",
                            "SimpleImputer (categorical)",
                            "OneHotEncoder",
                            "StandardScaler",
                            "MinMaxScaler",
                            "MaxAbsScaler",
                        ]
                        steps_container = st.container(border=False)
                        st.divider()
                        st.button(
                            "Add Step",
                            on_click=Callbacks.add_step,
                            use_container_width=True,
                            type="primary",
                        )

                        st.container(border=False, height=25)
                        processors_map = {
                            "SI": SimpleImputer,
                            "OHE": OneHotEncoder,
                            "SS": StandardScaler,
                            "MMS": MinMaxScaler,
                            "MAS": MaxAbsScaler,
                        }
                        if bool(st.session_state["preprocessing_steps"]):
                            steps_container.divider()
                        for steps in st.session_state["preprocessing_steps"]:
                            with steps_container:
                                definition = AutoPreProcessor(
                                    id=steps,
                                    preprocessor_options=preprocessor_options,
                                    num_cols=num_cols,
                                    cat_cols=cat_cols,
                                )
                                if definition.step_return:
                                    self.step_data[steps] = definition.step_return

                        pprocessing_steps = []

                        for seq, data in enumerate(self.step_data.values()):
                            c_step = processors_map.get(data.get("preprocess_type"))
                            if data.get("is_valid"):
                                pprocessing_steps.append(
                                    (data.get("title"), c_step(**data.get("kw")))
                                )

                        progress_cont = st.empty()
                        if (
                            len(pprocessing_steps)
                            == len(st.session_state["preprocessing_steps"])
                            and len(st.session_state["preprocessing_steps"]) > 0
                        ):
                            pproc_btn = st.button(
                                "Generate Preview",
                                use_container_width=True,
                            )

                            if pproc_btn:
                                prproc_prg = progress_cont.progress(
                                    value=0, text="Pre-Processing Dataset"
                                )
                                with prproc_prg:
                                    transform_pipeline = Pipeline(
                                        steps=pprocessing_steps
                                    )
                                    prproc_prg.progress(
                                        33, "Pipeline Transform Updated"
                                    )
                                    transform_pipeline.fit(st.session_state["dataset"])
                                    prproc_prg.progress(66, "Pipeline Fit Completed")
                                    st.session_state["processed_df"] = (
                                        transform_pipeline.transform(
                                            st.session_state["dataset"]
                                        )
                                    )
                                    prproc_prg.progress(
                                        100,
                                        "Pipeline Transform Completed - Preview Available",
                                    )
                                    progress_cont.empty()
                                    st.session_state["pipeline_run"] = True

                            preproc_preview = st.popover(
                                "Preview", use_container_width=True
                            )
                            if (
                                st.session_state["pipeline_run"]
                                and st.session_state["processed_df"]
                            ):
                                preproc_preview.container(height=20, border=False)
                                preproc_preview.dataframe(
                                    st.session_state["processed_df"].limit(10),
                                    hide_index=True,
                                )
                            st.button(
                                "Next",
                                use_container_width=True,
                                type="primary",
                                on_click=set_state,
                                args=[2],
                                key="pproc_nxt",
                            )
        if 2 in st.session_state["recorded_steps"] and st.session_state.get("dataset") is not None:
            # Validate we have the necessary data for modeling
            if not st.session_state.get("selected_features") or not st.session_state.get("selected_target"):
                st.error("❌ No feature or target columns selected. Please go back to preprocessing.")
                if st.button("← Back to Preprocessing", key="back_to_preprocessing"):
                    set_state(1)
                    st.rerun()
                return
                
            modeling_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with modeling_chat:
                st.write("Review your model options")
                with st.expander(
                    "Modeling", expanded=st.session_state["app_state"] == 2
                ):
                    st.header("Modeling Options")
                    model_types = [
                        {
                            "type": "Regression",
                            "models": [
                                "XGBRegressor",
                                "LinearRegression",
                                "ElasticNet",
                            ],
                        },
                        {
                            "type": "Classification",
                            "models": [
                                "XGBClassifier",
                                "LogisticRegression",
                            ],
                        },
                    ]

                    model_type = st.radio(
                        "Model Type",
                        options=[i.get("type") for i in model_types],
                        horizontal=True,
                    )
                    available_models = [
                        i for i in model_types if i.get("type") == model_type
                    ][0].get("models")
                    model_selections = st.selectbox("Model", options=available_models)
                    if bool(model_selections):
                        fit_menu = st.columns(4, gap="large")
                        show_metrics = fit_menu[0].toggle(
                            "Retrieve Model Metrics", value=True
                        )
                        fit_btn = fit_menu[1].button(
                            "Fit Model & Run Prediction(s)", use_container_width=True
                        )

                        with fit_menu[2].popover(
                            "Download Notebook",
                            use_container_width=True,
                            disabled=not st.session_state["model_ran"],
                        ):
                            st.markdown("Please input Project Name, then hit Save")
                            proj_name = st.text_input("Project Name")
                            registry_db = st.selectbox(
                                "Database",
                                options=get_databases(),
                                index=None,
                                placeholder="Choose a Database",
                                label_visibility="collapsed",
                                key="registry_nb_db",
                            )
                            if registry_db:
                                registry_schema = st.selectbox(
                                    "Schema",
                                    options=get_schemas(
                                        _session=self.session,
                                        database_name=(
                                            registry_db if registry_db else []
                                        ),
                                    ),
                                    index=None,
                                    placeholder="Choose a Schema",
                                    label_visibility="collapsed",
                                    key="registry_nb_schema",
                                )
                                # TODO: Make this not if-if-if nested.
                                if all([proj_name, registry_db, registry_schema]):
                                    download_column1, download_column2 = st.columns(2)
                                    st.session_state.notebook_btn = (
                                        download_column1.download_button(
                                            label="Download",
                                            data=create_notebook(
                                                st.session_state[
                                                    "full_qualified_table_nm"
                                                ],
                                                st.session_state.complete_pipeline,
                                                project_name=proj_name,
                                                registry_database=registry_db,
                                                registry_schema=registry_schema,
                                                context="local",
                                            ),
                                            file_name=proj_name + ".ipynb",
                                            mime="application/x-ipynb+json",
                                        )
                                    )
                                    upload_button = download_column2.button(
                                        label="Create Snowflake Notebook"
                                    )
                                    if upload_button:
                                        # TODO: Context is being set in app based on source data. Notebook creation should take user's context.
                                        stage_name = str(
                                            st.session_state["session"]
                                            .get_session_stage()
                                            .replace('"', "")
                                            .replace("@", "")
                                        )
                                        target_path = (
                                            f"@{stage_name}/ntbk/{proj_name}.ipynb"
                                        )
                                        self.session.file.put_stream(
                                            create_notebook(
                                                st.session_state[
                                                    "full_qualified_table_nm"
                                                ],
                                                st.session_state.complete_pipeline,
                                                project_name=proj_name,
                                                registry_database=registry_db,
                                                registry_schema=registry_schema,
                                                context="snowflake",
                                            ),
                                            target_path,
                                            auto_compress=False,
                                        )
                                        self.session.sql(
                                            f"""CREATE NOTEBOOK {registry_db}.{registry_schema}.{proj_name}
 FROM '@{stage_name}/ntbk'
 MAIN_FILE = '{proj_name}.ipynb'
 QUERY_WAREHOUSE = {self.session.get_current_warehouse()}"""
                                        ).collect()

                        if fit_btn:
                            # Get validated column selections from session state
                            session_feature_cols = st.session_state.get("selected_features", [])
                            session_target_col = st.session_state.get("selected_target")
                            
                            if not session_feature_cols or not session_target_col:
                                st.error("❌ Missing column selections. Please go back to preprocessing and select features and target.")
                                return
                            
                            # Validate that session state columns match current dataset
                            try:
                                current_dataset_columns = st.session_state["dataset"].columns
                                missing_features = [col for col in session_feature_cols if col not in current_dataset_columns]
                                if missing_features or session_target_col not in current_dataset_columns:
                                    st.error(f"❌ Column mismatch detected. Please refresh and re-select columns.")
                                    if missing_features:
                                        st.error(f"Missing features: {missing_features}")
                                    if session_target_col not in current_dataset_columns:
                                        st.error(f"Missing target: {session_target_col}")
                                    return
                            except Exception as e:
                                st.error(f"❌ Dataset validation error: {str(e)}")
                                return
                            
                            # Use validated columns for modeling
                            feature_cols = session_feature_cols
                            target_col = session_target_col
                            fit_prg_cont = st.empty()
                            fit_prg = fit_prg_cont.progress(
                                value=0, text="Fitting Model"
                            )

                            model_classes = {
                                "XGBClassifier": (XGBClassifier, {}),
                                "LinearRegression": (LinearRegression, {}),
                                "ElasticNet": (ElasticNet, {}),
                                "XGBRegressor": (XGBRegressor, {}),
                                "LogisticRegression": (LogisticRegression, {}),
                            }

                            # Validate columns before creating model parameters
                            if not feature_cols or not target_col:
                                st.error("❌ Invalid column selections for model training")
                                return
                            
                            # Debug: Show what we're using for modeling
                            if st.session_state.get("debug_cols", False):
                                st.write(f"🔍 **Model Configuration**")
                                st.write(f"🔍 Model type: {model_selections}")
                                st.write(f"🔍 Feature columns: {feature_cols}")
                                st.write(f"🔍 Target column: {target_col}")
                                st.write(f"🔍 Preprocessing steps: {len(pprocessing_steps)}")
                            
                            try:
                                shared_params = {
                                    "random_state": 42,
                                    "input_cols": feature_cols,
                                    "n_jobs": -1,
                                    "label_cols": target_col,
                                }

                                if model_selections in model_classes:
                                    model_class, specific_params = model_classes[
                                        model_selections
                                    ]
                                    # Remove incompatible parameters for specific models
                                    if model_selections == "LinearRegression":
                                        shared_params.pop("random_state", None)
                                        shared_params.pop("n_jobs", None)
                                    elif model_selections == "ElasticNet":
                                        shared_params.pop("n_jobs", None)
                                    elif model_selections == "LogisticRegression":
                                        shared_params.pop("n_jobs", None)
                                    
                                    # Create model with validated parameters
                                    model = model_class(**shared_params, **specific_params)
                                    pprocessing_steps.append((model_selections, model))
                                    
                                    if st.session_state.get("debug_cols", False):
                                        st.write(f"🔍 Model created successfully: {type(model).__name__}")
                                        st.write(f"🔍 Model parameters: {shared_params}")
                                else:
                                    st.error(f"❌ Unknown model selection: {model_selections}")
                                    return
                                    
                            except Exception as e:
                                st.error(f"❌ Model creation failed: {str(e)}")
                                st.error(f"Model: {model_selections}")
                                st.error(f"Parameters: {shared_params}")
                                return

                            # Create pipeline with validation
                            try:
                                if not pprocessing_steps:
                                    st.error("❌ No pipeline steps configured")
                                    return
                                    
                                complete_pipeline = Pipeline(steps=pprocessing_steps)
                                
                                if st.session_state.get("debug_cols", False):
                                    st.write(f"🔍 Pipeline created with {len(pprocessing_steps)} steps")
                                    for i, (name, step) in enumerate(pprocessing_steps):
                                        st.write(f"  Step {i+1}: {name} - {type(step).__name__}")
                                        
                            except Exception as e:
                                st.error(f"❌ Pipeline creation failed: {str(e)}")
                                st.error(f"Steps: {[(name, type(step).__name__) for name, step in pprocessing_steps]}")
                                return
                            
                            # Enhanced pipeline fitting with comprehensive error handling
                            try:
                                # Step 1: Re-validate dataset state before pipeline fitting
                                current_columns = st.session_state["dataset"].columns
                                all_needed_columns = feature_cols + [target_col]
                                
                                if st.session_state.get("debug_cols", False):
                                    st.write(f"🔍 **Pipeline Fitting Phase**")
                                    st.write(f"🔍 Current dataset columns: {current_columns}")
                                    st.write(f"🔍 Required columns: {all_needed_columns}")
                                    st.write(f"🔍 Pipeline steps: {len(complete_pipeline.steps)}")
                                
                                # Step 2: Validate all required columns exist
                                missing_columns = [col for col in all_needed_columns if col not in current_columns]
                                
                                if missing_columns:
                                    st.error("🚨 **Pipeline Fitting Failed - Missing Columns**")
                                    st.error(f"Missing columns: {missing_columns}")
                                    st.error(f"Available columns: {current_columns}")
                                    
                                    with st.expander("🔍 Detailed Column Analysis", expanded=True):
                                        st.write("**Feature Columns Analysis:**")
                                        for i, col in enumerate(feature_cols):
                                            status = "✅" if col in current_columns else "❌"
                                            st.write(f"  {status} Feature {i+1}: '{col}'")
                                            if col not in current_columns:
                                                matches = [c for c in current_columns if col.lower() in c.lower()]
                                                if matches:
                                                    st.write(f"      Possible matches: {matches}")
                                        
                                        st.write("**Target Column Analysis:**")
                                        status = "✅" if target_col in current_columns else "❌"
                                        st.write(f"  {status} Target: '{target_col}'")
                                        if target_col not in current_columns:
                                            matches = [c for c in current_columns if target_col.lower() in c.lower()]
                                            if matches:
                                                st.write(f"      Possible matches: {matches}")
                                    
                                    st.error("**Possible Causes:**")
                                    st.error("- Column state changed between preprocessing and modeling phases")
                                    st.error("- Dataset was refreshed or modified during processing")
                                    st.error("- Session state corruption or encoding issues")
                                    
                                    # Recovery options for pipeline fitting
                                    st.info("**Recovery Options:**")
                                    pipeline_recovery_cols = st.columns(3)
                                    
                                    if pipeline_recovery_cols[0].button("🔄 Refresh & Retry", key="pipeline_refresh"):
                                        try:
                                            db = st.session_state.get('aml_mpa.sel_db')
                                            schema = st.session_state.get('aml_mpa.sel_schema') 
                                            table = st.session_state.get('aml_mpa.sel_table')
                                            if all([db, schema, table]):
                                                st.session_state["dataset"] = self.session.table(f"{db}.{schema}.{table}")
                                                st.success("Dataset refreshed - please retry modeling")
                                                st.rerun()
                                        except Exception as refresh_e:
                                            st.error(f"Refresh failed: {str(refresh_e)}")
                                    
                                    if pipeline_recovery_cols[1].button("← Back to Preprocessing", key="back_preprocess"):
                                        set_state(1)
                                        st.rerun()
                                    
                                    if pipeline_recovery_cols[2].button("🏠 Start Over", key="start_over"):
                                        # Clear all session state and restart
                                        keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('aml_', 'selected_', 'dataset', 'processed_'))]
                                        for key in keys_to_clear:
                                            del st.session_state[key]
                                        set_state(0)
                                        st.rerun()
                                    
                                    st.stop()
                                
                                # Step 3: Create validated column selection for training
                                validated_features = [col for col in feature_cols if col in current_columns]
                                training_columns = validated_features + [target_col]
                                
                                if st.session_state.get("debug_cols", False):
                                    st.write(f"🔍 Training columns: {training_columns}")
                                    st.write(f"🔍 Validated features: {validated_features}")
                                
                                # Step 4: Select training data with validated columns
                                try:
                                    training_data = st.session_state["dataset"].select(training_columns)
                                    
                                    if st.session_state.get("debug_cols", False):
                                        st.write(f"🔍 Training data columns: {training_data.columns}")
                                        try:
                                            row_count = training_data.count()
                                            st.write(f"🔍 Training data shape: {row_count} rows")
                                            if row_count == 0:
                                                st.error("❌ Training data is empty!")
                                                return
                                        except Exception as count_e:
                                            st.warning(f"⚠️ Could not count rows: {str(count_e)}")
                                    
                                except Exception as data_e:
                                    st.error(f"❌ Failed to select training data: {str(data_e)}")
                                    st.error(f"Columns requested: {training_columns}")
                                    st.error(f"Available columns: {current_columns}")
                                    return
                                
                                # Step 5: Fit the pipeline with enhanced error handling
                                fit_prg.progress(25, "Validating data and pipeline...")
                                
                                try:
                                    # Add a small delay to allow UI to update
                                    import time
                                    time.sleep(0.1)
                                    
                                    fit_prg.progress(30, "Starting pipeline fitting...")
                                    complete_pipeline.fit(training_data)
                                    fit_prg.progress(75, "Pipeline fitted successfully...")
                                    
                                except Exception as fit_error:
                                    # Handle specific Snowflake ML errors
                                    error_msg = str(fit_error)
                                    fit_prg_cont.empty()  # Clear progress bar
                                    
                                    st.error("🚨 **Pipeline Fitting Failed**")
                                    st.error(f"**Error:** {error_msg}")
                                    
                                    # Provide specific guidance based on error type
                                    if "telemetry" in error_msg.lower():
                                        st.error("**Issue:** Snowflake ML internal error")
                                        st.error("**Suggestions:**")
                                        st.error("- Try with fewer features")
                                        st.error("- Remove preprocessing steps")
                                        st.error("- Check data types compatibility")
                                    elif "column" in error_msg.lower():
                                        st.error("**Issue:** Column-related error")
                                        st.error("**Suggestions:**")
                                        st.error("- Verify column names")
                                        st.error("- Check for null values")
                                        st.error("- Ensure proper data types")
                                    elif "data" in error_msg.lower():
                                        st.error("**Issue:** Data-related error")
                                        st.error("**Suggestions:**")
                                        st.error("- Check for empty dataset")
                                        st.error("- Verify data quality")
                                        st.error("- Remove rows with null values")
                                    
                                    # Recovery options specific to fitting errors
                                    st.info("**Recovery Options:**")
                                    fit_recovery_cols = st.columns(4)
                                    
                                    if fit_recovery_cols[0].button("🔧 Try Minimal Pipeline", key="minimal_pipeline"):
                                        # Create a minimal pipeline with just the model
                                        try:
                                            minimal_steps = [(model_selections, model)]
                                            minimal_pipeline = Pipeline(steps=minimal_steps)
                                            minimal_pipeline.fit(training_data)
                                            st.success("✅ Minimal pipeline fitted successfully!")
                                            st.session_state["complete_pipeline"] = minimal_pipeline
                                            st.rerun()
                                        except Exception as minimal_e:
                                            st.error(f"Minimal pipeline also failed: {str(minimal_e)}")
                                    
                                    if fit_recovery_cols[1].button("📊 Check Data Sample", key="check_data"):
                                        try:
                                            sample_data = training_data.limit(5).to_pandas()
                                            st.write("**Data Sample:**")
                                            st.dataframe(sample_data)
                                            st.write("**Data Types:**")
                                            st.write(sample_data.dtypes)
                                        except Exception as sample_e:
                                            st.error(f"Cannot sample data: {str(sample_e)}")
                                    
                                    if fit_recovery_cols[2].button("🔄 Retry with Debug", key="retry_debug"):
                                        st.session_state["debug_cols"] = True
                                        st.rerun()
                                    
                                    if fit_recovery_cols[3].button("← Back to Setup", key="back_to_setup_fit"):
                                        set_state(1)
                                        st.rerun()
                                    
                                    return  # Stop execution after fitting error
                                
                            except Exception as e:
                                st.error("🚨 **Critical Pipeline Fitting Error**")
                                st.error(f"**Error:** {str(e)}")
                                st.error(f"**Error Type:** {type(e).__name__}")
                                
                                with st.expander("🔍 Complete Error Details", expanded=True):
                                    st.write(f"**Selected Features:** {feature_cols}")
                                    st.write(f"**Selected Target:** {target_col}")
                                    st.write(f"**Dataset Columns:** {st.session_state['dataset'].columns}")
                                    st.write(f"**Pipeline Steps:** {len(pprocessing_steps)} steps")
                                    
                                    # Show pipeline configuration
                                    st.write("**Pipeline Configuration:**")
                                    for i, (name, step) in enumerate(pprocessing_steps):
                                        st.write(f"  Step {i+1}: {name} - {type(step).__name__}")
                                    
                                    # Show the exact error traceback if available
                                    import traceback
                                    st.code(traceback.format_exc())
                                
                                # Error-specific recovery suggestions
                                error_msg = str(e).lower()
                                st.error("**Suggested Actions:**")
                                
                                if "column" in error_msg and "not present" in error_msg:
                                    st.error("- This appears to be a column mismatch error")
                                    st.error("- Try refreshing the dataset and re-selecting columns")
                                    st.error("- Check if column names have changed or contain special characters")
                                elif "session" in error_msg:
                                    st.error("- This appears to be a Snowflake session error")
                                    st.error("- Check your Snowflake connection")
                                    st.error("- Try restarting the application")
                                elif "pipeline" in error_msg:
                                    st.error("- This appears to be a pipeline configuration error")
                                    st.error("- Check preprocessing steps and model parameters")
                                    st.error("- Try simplifying the pipeline")
                                else:
                                    st.error("- This is an unexpected error")
                                    st.error("- Try clearing all selections and starting over")
                                    st.error("- Check the application logs for more details")
                                
                                # Recovery buttons for pipeline errors
                                error_recovery_cols = st.columns(4)
                                
                                if error_recovery_cols[0].button("🔄 Retry", key="pipeline_retry"):
                                    st.rerun()
                                
                                if error_recovery_cols[1].button("🔧 Simplify Pipeline", key="simplify_pipeline"):
                                    # Clear preprocessing steps and retry with just the model
                                    st.session_state["preprocessing_steps"] = []
                                    st.info("Preprocessing steps cleared - trying with model only")
                                    st.rerun()
                                
                                if error_recovery_cols[2].button("← Back to Setup", key="back_setup"):
                                    set_state(1)
                                    st.rerun()
                                
                                if error_recovery_cols[3].button("📋 Copy Error", key="copy_pipeline_error"):
                                    error_details = f"""
Pipeline Fitting Error:
Type: {type(e).__name__}
Message: {str(e)}
Features: {feature_cols}
Target: {target_col}
Dataset Columns: {st.session_state['dataset'].columns}
Pipeline Steps: {len(pprocessing_steps)}
                                    """
                                    st.code(error_details)
                                
                                st.stop()
                            fit_prg.progress(
                                value=50, text="Model Fitted, running predictions"
                            )
                            st.session_state["complete_pipeline"] = complete_pipeline
                            predictions = complete_pipeline.predict(
                                st.session_state["dataset"]
                            )
                            st.session_state["ml_model_predictions"] = predictions
                            fit_prg.progress(value=100, text="Predictions Complete")
                            st.session_state["model_ran"] = True
                            if show_metrics:
                                metric_results = {}
                                model_metrics = ModelMetrics(targel_col=target_col)
                                metrics = model_metrics.metrics_map.get(
                                    str(model_type).title()
                                )
                                for idx, metric in enumerate(metrics):
                                    metric_fn = metrics.get(metric).get("fn")
                                    metric_kw = metrics.get(metric).get("kw")
                                    metric_results[metric] = metric_fn(
                                        st.session_state["ml_model_predictions"],
                                        **metric_kw,
                                    )

                            if st.session_state["model_ran"]:
                                st.session_state["pipeline_object"] = complete_pipeline
                                if show_metrics:
                                    st.session_state["pipeline_metrics"] = (
                                        metric_results
                                    )
                                st.rerun()

                        if st.session_state["ml_model_predictions"]:
                            st.dataframe(
                                st.session_state["ml_model_predictions"].limit(15),
                                hide_index=True,
                                use_container_width=True,
                            )
                            with fit_menu[3].popover(
                                "Save to Registry",
                                use_container_width=True,
                                disabled=not st.session_state["model_ran"],
                            ):
                                st.markdown("Please input Project Name, then hit Save")
                                if st.session_state["environment"] == "sis":
                                    tgt_database = st.session_state[
                                        "session"
                                    ].get_current_database()
                                    tgt_schema = st.session_state[
                                        "session"
                                    ].get_current_schema()
                                    st.caption(
                                        f"{tgt_database[1:-1]}.{tgt_schema[1:-1]}"
                                    )

                                else:
                                    tgt_database = st.selectbox(
                                        "Database",
                                        options=get_databases(),
                                        index=None,
                                        placeholder="Choose a Database",
                                        label_visibility="collapsed",
                                        key="tgt_db",
                                    )
                                    tgt_schema = st.selectbox(
                                        "Schema",
                                        options=(
                                            get_schemas(
                                                database_name=(
                                                    tgt_database if tgt_database else []
                                                ),
                                                _session=self.session,
                                            )
                                            if tgt_database
                                            else []
                                        ),
                                        index=None,
                                        placeholder="Choose a Schema",
                                        label_visibility="collapsed",
                                        key="tgt_schema",
                                    )
                                tgt_model_name = st.text_input(
                                    "",
                                    label_visibility="collapsed",
                                    placeholder="Model Name",
                                    key="tgt_forecast_name",
                                )
                                target_location = (
                                    ".".join([tgt_database, tgt_schema, tgt_model_name])
                                    if all([tgt_database, tgt_schema, tgt_model_name])
                                    else None
                                )
                                if target_location:
                                    reg = Registry(
                                        session=self.session,
                                        database_name=tgt_database,
                                        schema_name=tgt_schema,
                                    )
                                    try:
                                        reg.get_model(tgt_model_name)
                                        st.write(
                                            f"Model {tgt_model_name} already exists in {tgt_database}.{tgt_schema}. Would you like to save as a new version?"
                                        )
                                        button_text = "Save New Version"
                                    except ValueError:
                                        button_text = "Register"

                                    register_columns = st.columns(2)

                                    if register_columns[0].button(button_text):
                                        try:
                                            with register_columns[1]:
                                                with st.spinner("Saving..."):
                                                    query_tag_comment = '{"origin": "sf_sit", "name": "ml_sidekick", "version": {"major":1, "minor":0}, "attributes":{"component":"model"}}'
                                                    reg.log_model(
                                                        model=st.session_state[
                                                            "pipeline_object"
                                                        ],
                                                        model_name=tgt_model_name,
                                                        metrics=st.session_state[
                                                            "pipeline_metrics"
                                                        ],
                                                        comment = query_tag_comment
                                                    )
                                                    reg.get_model(tgt_model_name).comment = query_tag_comment
                                                    st.toast("Model Registered")

                                        except Exception as e:
                                            st.toast(
                                                f"Failed to register model \n\n {e}"
                                            )

                        if st.session_state["pipeline_metrics"]:
                            st.header("Model Metrics", anchor=False)
                            metric_columns = st.columns(2, gap="small")
                            metric_pills = []
                            for key, metric in st.session_state[
                                "pipeline_metrics"
                            ].items():
                                if key == "Confusion Matrix":
                                    tp = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            0
                                        ][0]
                                    )
                                    fp = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            0
                                        ][1]
                                    )
                                    fn = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            1
                                        ][0]
                                    )
                                    tn = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            1
                                        ][1]
                                    )

                                    data = [
                                        [
                                            tp,
                                            "TP",
                                            "Positive",
                                            "Positive",
                                            "pos",
                                        ],
                                        [fp, "FP", "Positive", "Negative", "neg"],
                                        [fn, "FN", "Negative", "Positive", "neg"],
                                        [tn, "TN", "Negative", "Negative", "pos"],
                                    ]

                                    df = pd.DataFrame(
                                        data,
                                        columns=[
                                            "value",
                                            "label",
                                            "predicted",
                                            "actual",
                                            "color",
                                        ],
                                    )
                                    df["calculated_text"] = df.apply(
                                        lambda x: x["label"] + ":" + str(x["value"]),
                                        axis=1,
                                    )
                                    colors = ["#29b5e8", "grey"]
                                    domains = ["pos", "neg"]
                                    base = (
                                        alt.Chart(df)
                                        .mark_rect(height=90, width=90, cornerRadius=5)
                                        .encode(
                                            x=alt.X(
                                                "actual",
                                                type="nominal",
                                                sort="y",
                                                title="Actual Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="top",
                                                    labelAngle=0,
                                                    labelFontSize=20,
                                                    labelColor="black",
                                                    titleColor="black",
                                                ),
                                            ),
                                            y=alt.Y(
                                                "predicted",
                                                type="nominal",
                                                sort="x",
                                                title="Predicted Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="left",
                                                    labelAngle=-90,
                                                    labelFontSize=20,
                                                    labelColor="black",
                                                    titleColor="black",
                                                ),
                                            ),
                                            color=alt.Color(
                                                field="color",
                                                type="nominal",
                                                scale=alt.Scale(
                                                    domain=domains, range=colors
                                                ),
                                                title="Region",
                                                legend=None,
                                            ),
                                            tooltip=alt.value(None),
                                        )
                                    )
                                    text = (
                                        alt.Chart(df)
                                        .mark_text(
                                            fontSize=16,
                                        )
                                        .encode(
                                            x=alt.X(
                                                "actual",
                                                type="nominal",
                                                sort="y",
                                                title="Actual Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="top",
                                                    labelAngle=0,
                                                ),
                                            ),
                                            y=alt.Y(
                                                "predicted",
                                                type="nominal",
                                                sort="x",
                                                title="Predicted Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="left",
                                                    labelAngle=-90,
                                                ),
                                            ),
                                            color=alt.value("white"),
                                            text=alt.Text("calculated_text"),
                                            tooltip=alt.value(None),
                                        )
                                    )
                                    layered = base + text
                                    metric_columns[1].subheader(
                                        "Confusion Matrix", divider=True
                                    )
                                    metric_columns[1].altair_chart(
                                        layered.properties(
                                            width=360, height=330, padding={"left": 100}
                                        )
                                    )
                                else:
                                    metric_pills.append(create_metric_card(key, metric))
                            with metric_columns[0]:
                                st.subheader("Metrics", divider=True)
                                pills = "\n".join(metric_pills)
                                html(
                                    f"""{st.session_state["css_styles"]}
                                    <div class="contact_card">
                                                {pills}
                                            </div>
                                                """,
                                )
                            feat_imp_df = get_feature_importance_df(
                                st.session_state.get("pipeline_object")
                            )
                            feat_imp_df["ABS_VALUE"] = (
                                feat_imp_df["IMPORTANCE"].abs().astype(float)
                            )
                            feat_imp_df["IMPORTANCE"] = (
                                feat_imp_df["IMPORTANCE"].round(3).astype(float)
                            )
                            feat_imp_df = feat_imp_df[
                                ["FEATURE", "ABS_VALUE", "IMPORTANCE"]
                            ]

                            metric_columns[0].subheader(
                                "Feature Importance", divider=True
                            )

                            metric_columns[0].dataframe(
                                feat_imp_df,
                                column_config={
                                    "ABS_VALUE": st.column_config.ProgressColumn(
                                        "Relative Importance",
                                        format=" ",
                                        min_value=feat_imp_df["ABS_VALUE"].min(),
                                        max_value=feat_imp_df["ABS_VALUE"].max(),
                                        width="large",
                                    ),
                                    "FEATURE": st.column_config.TextColumn(
                                        "Feature", width="small"
                                    ),
                                    "IMPORTANCE": st.column_config.TextColumn(
                                        "Importance", width="small"
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True,
                            )
