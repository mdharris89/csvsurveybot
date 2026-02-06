__version__ = '1.5.2'
__build__ = '20260205'

import streamlit as st
from streamlit import fragment
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import random
import time
import json
import io
import base64
import re
from typing import List, Optional, Union, Type, Dict, Any
import string
import copy
from difflib import SequenceMatcher
import itertools

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import os
import signal
import sys
from pathlib import Path

import logging
import datetime
import traceback

###############################################################################
# DEPENDENCY LOGGING
###############################################################################
import tempfile

def log_dependencies_on_quit():
    """
    Captures and logs all loaded Python modules at application exit time.
    
    Creates a timestamped log file in the temporary directory that lists all modules 
    that were loaded during the application's execution, organized by package.
    This is useful for debugging dependency issues, especially when packaging
    the application with PyInstaller.
    
    The log includes:
    - Python version and platform information
    - Command line arguments
    - All loaded modules with their file paths
    
    No parameters or return values.
    """
    try:
        # 1. Define Log Directory and File Path
        # Use tempfile to get a reliable temporary directory path
        temp_dir = tempfile.gettempdir() 
        log_dir = os.path.join(temp_dir, "surveybot_temp_logs")
        print(f"DEBUG: Attempting to use log directory for quit log: {log_dir}") # Keep for debug if needed
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f"dependencies_at_quit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        dependency_log_path = os.path.join(log_dir, log_filename)

        # 2. Configure Logging (Force re-configuration)
        # Use basicConfig with force=True to ensure it overrides any previous config
        logging.basicConfig(
            filename=dependency_log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True # Necessary if logging might have been configured elsewhere
        )
        logging.info("Dependency logging initiated on quit.")

        # 3. Log System Info
        logging.info(f"Python version: {sys.version}")
        logging.info(f"Platform: {sys.platform}")
        logging.info(f"Process quitting with args: {sys.argv}")

        # 4. Get and Log All Currently Loaded Modules
        current_modules = set(sys.modules.keys())
        logging.info("\n=== All Loaded Modules at Quit ===")
        
        package_modules = {}
        for name in sorted(current_modules):
            if not name.startswith('_'): # Skip internal/pseudo modules
                # Group by top-level package
                top_package = name.split('.')[0]
                if top_package not in package_modules:
                    package_modules[top_package] = []
                package_modules[top_package].append(name)
                
        # Log organized by package
        for package, modules in sorted(package_modules.items()):
            logging.info(f"\nPackage: {package}")
            for module in sorted(modules):
                try:
                    mod_obj = sys.modules.get(module)
                    # Use Path for better cross-platform handling if path exists
                    file_path = 'built-in'
                    if mod_obj and hasattr(mod_obj, '__file__') and mod_obj.__file__:
                         file_path = Path(mod_obj.__file__).resolve()
                    elif mod_obj is None:
                         file_path = 'unknown (module not found in sys.modules)'
                         
                    logging.info(f"  {module}: {file_path}")
                except Exception as e:
                    logging.info(f"  {module}: Error getting info - {str(e)}")

        logging.info("=== End of Module List ===")
        
    except Exception as e:
        # Try to log the error itself, might fail if logging setup failed
        try:
            logging.error(f"FATAL Error during dependency logging on quit: {str(e)}")
            logging.error(traceback.format_exc()) # Log stack trace
        except:
            # Fallback print if logging fails completely
            print(f"FATAL Error during dependency logging on quit: {str(e)}")
            print(traceback.format_exc())
            
    finally:
        # 5. Shutdown Logging
        # Ensure logging is shut down regardless of errors
        logging.shutdown()
        DEBUG_LOGS = False
        if DEBUG_LOGS:
            print(f"DEBUG: Dependency log written to {dependency_log_path}")

###############################################################################
# 0. SESSION STATE INITIALIZATION
###############################################################################
# Session state contract (canonical keys used across the app):
# - data: survey, unique_id_column, active_question, processed_questions,
#   final_results, profiling_columns
# - reasoning: theme_sets_by_question, active_theme_set_ids, current_themes,
#   theme_counts
# - classification: results_by_question, justifications_by_theme_set, df_indices,
#   sparse_results, justifications
# - model/settings: openai_client, reasoning_model_selected, reasoning_temperature,
#   classification_model_selected, classification_temperature, batch_size,
#   additional_theme_generation_instructions, additional_classification_instructions,
#   enable_evaluate_and_improve_examples, enable_consolidate_themes,
#   reasoning_effort, verbosity
# - ui/runtime: themes_to_delete, backup_themes, reclassify_requested,
#   stored_model_for_reclassify, stored_temperature_for_reclassify,
#   stored_batch_size_for_reclassify, include_examples_in_classification
def initialise_session_states():
    """
    Initializes all Streamlit session state variables required by the application.
    
    This function ensures all required state keys exist with proper default values,
    creating a clean initial state for the application. Session state is organized
    into several categories:
    
    - Data state: Stores survey data, active question, and processed questions
    - Reasoning state: Stores theme sets, current themes, and acceptance status
    - Classification state: Stores classification results and justifications
    - Model settings: Stores LLM model configurations and parameters
    - UI state: Essential interface-related state variables
    
    No parameters or return values.
    """
    def init_state(key, default_value):
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Data state
    init_state("data", {
        "survey": None, 
        "unique_id_column": None, 
        "active_question": None, 
        "processed_questions": [], 
        "final_results": None,
        "profiling_columns": []
    })

    # Reasoning state
    init_state("reasoning", {
        "theme_sets_by_question": {},  # {question: {theme_set_id: {name, themes, created_at, instructions}}}
        "active_theme_set_ids": {},    # {question: active_theme_set_id}
        "current_themes": [],          # Themes for currently active set
        "theme_counts": {}             # Counts for the current active theme set
    })

    # Classification state
    init_state("classification", {
        "results_by_question": {},
        "justifications_by_theme_set": {},
        "df_indices": [],
        # Current active sparse data
        "sparse_results": {},
        "justifications": {}
    })

    # Model and parameter settings
    init_state("openai_client", None)
    init_state("reasoning_model_selected", "OpenAI: gpt-5.2")
    init_state("sample_size", 20) # sample size is for reasoning
    init_state("max_reasoning_batches", 20)
    init_state("additional_theme_generation_instructions", "")
    init_state("reasoning_temperature", 0.8)
    init_state("reasoning_effort", "low")
    init_state("verbosity", "low")
    
    init_state("classification_model_selected", "OpenAI: gpt-5.2")
    init_state("batch_size", 10) # batch size is for classification
    init_state("additional_classification_instructions", "")
    init_state("classification_temperature", 0.0)

    init_state("enable_evaluate_and_improve_examples", True)
    init_state("enable_consolidate_themes", False)

    # UI state
    init_state("themes_to_delete", set())
    init_state("backup_themes", None)
    init_state("reclassify_requested", False)
    init_state("stored_model_for_reclassify", None)
    init_state("stored_temperature_for_reclassify", None)
    init_state("stored_batch_size_for_reclassify", None)
    init_state("include_examples_in_classification", True)

MAX_WORKERS = 50

###############################################################################
# 1. SIDEBAR MANAGEMENT
###############################################################################
def load_csv_file(file):
    df = pd.read_csv(file)
    st.session_state["data"]["survey"] = df

    # Shuffle all indices once when data is loaded
    # This provides a consistent random order for sampling for all questions
    all_rows_indices = list(range(len(df)))
    random.shuffle(all_rows_indices)
    st.session_state["shuffled_indices"] = all_rows_indices

def quit_app():
    """Handle application quit with a nice 'goodbye' overlay, then kill this process."""

    log_dependencies_on_quit()

    st.markdown(
        """
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                    display: flex; align-items: center; justify-content: center; 
                    background-color: rgba(255, 255, 255, 0.9); z-index: 1000;">
            <div style="background-color: white; padding: 30px; border-radius: 10px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;">
                <h2>Application Closed</h2>
                <p>Thank you for using Survey Bot  =) </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Delay briefly so the user can see the overlay
    time.sleep(2)
    if getattr(sys, 'frozen', False):
        # If running as compiled app, use sys.exit
        sys.exit(0)
    else:
        # If running as script, use the original signal approach
        os.kill(os.getpid(), signal.SIGTERM)

def setup_sidebar():
    """
    Sidebar: 
      - Optionally add openai api key
      - Upload CSV 
      - Question/ID selection and theme set management when CSV loaded
      - When we generate themes later they will show here
    """
    sidebar_container = st.sidebar.empty()
    with sidebar_container.container():
        st.title("LLM Theme Categorization")
        st.caption(f"v{__version__} (build {__build__})")

        if 'browser_session_active' not in st.session_state:
            st.session_state['browser_session_active'] = True
            
            def on_session_end():
                # If user simply closes the browser tab, we kill ourselves:
                time.sleep(2)  # small delay
                os.kill(os.getpid(), signal.SIGTERM)
            
            try:
                import streamlit.runtime.scriptrunner_utils.script_run_context as src
                ctx = src.get_script_run_ctx()
                if ctx:
                    session_info = ctx.session_info
                    if hasattr(session_info, 'on_session_end'):
                        session_info.on_session_end(on_session_end)
            except Exception as e:
                pass
        
        if st.button("Quit Application", key="quit_button"):
            quit_app()

        # Save and load functionality
        with st.expander("Load/ Save Analysis State"):
            
            # Load button: let the user upload a saved state file.
            uploaded_state = st.file_uploader("Load Analysis State", type=["json"], key="state_file_uploader", help="Upload a previously saved .json state file to restore your entire analysis session (data, themes, classifications).")
            if uploaded_state is not None:
                # Use file content hash to detect new uploads
                file_content = uploaded_state.getvalue()
                file_hash = hash(file_content)

                # Check if this is a new file
                if st.session_state.get("last_loaded_state_hash") != file_hash:
                    try:
                        uploaded_state.seek(0)  # Reset file pointer
                        if load_state_json(uploaded_state):
                            st.success("State loaded successfully!")
                            st.session_state["last_loaded_state_hash"] = file_hash
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load state: {e}")

            # Save button: let the user download the state file as a json.
            if "save_state_filename" not in st.session_state:
                st.session_state["save_state_filename"] = f"analysis_state_{time.strftime('%Y%m%d_%H%M%S')}"
            custom_filename = st.text_input("Filename for save state:", key="save_state_filename")
            if not custom_filename.endswith('.json'):
                custom_filename += '.json'
            json_str = save_state_json()

            # Safety check: warn if the exported JSON looks like it contains secrets.
            # This is defensive for future UI additions; API keys are explicitly excluded in `save_state_json()`.
            _secret_patterns = [
                r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}",  # OpenAI
                r"AKIA[0-9A-Z]{16}",                 # AWS access key id
                r"ghp_[A-Za-z0-9]{36}",              # GitHub classic token
                r"github_pat_[A-Za-z0-9_]{20,}",     # GitHub fine-grained token
            ]
            if any(re.search(p, json_str or "") for p in _secret_patterns):
                st.warning(
                    "Potential secret-like string detected in the analysis-state export. "
                    "Review the JSON before sharing publicly."
                )
                
            st.download_button(
                label="Save Analysis State",
                data=json_str,
                file_name=custom_filename,
                mime="application/json",
                help="Downloads a .json file containing your full analysis state. Does not include your API key."
            )

        # Openai key
        openai_api_key = st.text_input(
            "For OpenAI models, enter your API Key",
            type="password",
            key="openai_api_key_input",
        )
        if openai_api_key:
            st.session_state["openai_client"] = OpenAI(api_key=openai_api_key)
            try:
                st.session_state["openai_client"].models.list() 
                st.write("OpenAI API key set successfully!")
            except Exception as e:
                st.write(f"Invalid API key or error: {e}")

        with st.expander("CSV uploader"):
            file = st.file_uploader("Upload wide-format CSV", type=["csv"], key="csv_file_uploader", help="CSV where each column is a survey question and each row is one respondent. The app will let you select which column(s) to analyse.")
            if file:
                # Check if we need to reset state for a new file
                file_content = file.getvalue()
                file_hash = hash(file_content)

                if "last_loaded_file_hash" not in st.session_state or st.session_state["last_loaded_file_hash"] != file_hash:
                    # Load the file first so we have access to the data
                    load_csv_file(file)
                    
                    # Reset state for new CSV
                    st.session_state["data"]["processed_questions"] = []
                    st.session_state["reasoning"]["theme_sets_by_question"] = {}
                    st.session_state["data"]["final_results"] = None
                    
                    # Use existing function to reset the rest
                    reset_for_new_question()
                    
                    # Store the file hash to detect changes
                    st.session_state["last_loaded_file_hash"] = file_hash
                    
                    st.success("CSV loaded successfully! Previous state has been reset.")
                else:
                    # Same file as before, just load it without resetting
                    load_csv_file(file)
                    st.success("CSV loaded successfully!")

                # Optionally select a unique id
                if st.session_state["data"]["unique_id_column"] is None:
                    df = st.session_state["data"]["survey"]
                    columns = list(df.columns)
                    unique_columns = [col for col in columns if df[col].is_unique]                
                    # Get the current unique ID or default to None
                    current_unique_id = st.session_state["data"]["unique_id_column"]
                    default_index = 0  # Default to "None"
                    
                    if current_unique_id and current_unique_id in unique_columns:
                        default_index = unique_columns.index(current_unique_id) + 1
                    
                    unique_id_col = st.selectbox(
                        "Unique identifier column", 
                        options=["None"] + unique_columns, 
                        key="unique_id_column_selector",
                        index=default_index,
                        help = "If provided, unique id will be included when exporting"
                    )
                    
                    if unique_id_col != "None":
                        st.session_state["data"]["unique_id_column"] = unique_id_col
                    else:
                        st.session_state["data"]["unique_id_column"] = None
                        st.warning("Warning: No unique identifier column selected. Row numbers will be used.")

        # Processed questions expander
        if st.session_state["data"]["survey"] is not None and st.session_state["data"]["processed_questions"]:
            st.divider()
            with st.expander("Processed Questions", expanded=False):
                for q in st.session_state["data"]["processed_questions"]:
                    # Count themes across all theme sets for this question
                    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(q, {})
                    total_themes = 0
                    theme_set_names = []
                    
                    for theme_set_id, theme_set in theme_sets.items():
                        total_themes += len(theme_set["themes"])
                        theme_set_names.append(theme_set["name"])
                    
                    theme_sets_info = ", ".join(theme_set_names) if theme_set_names else "No theme sets"
                    st.write(f"✓ {q} ({total_themes} themes across {len(theme_sets)} sets: {theme_sets_info})")

            if st.button("Analyse new question (resets UI)", key="reset_ui_button", disabled=is_theme_editor_dirty()):
                st.session_state["reasoning"]["current_themes"] = []
                reset_for_new_question()
                st.rerun()

        # If CSV is loaded, show question selection UI
        if st.session_state["data"]["survey"] is not None:
            st.divider()
            selected_question = select_question_to_analyse()
            
            # Show theme management if theme sets exist
            active_question = st.session_state["data"]["active_question"]
            if active_question:
                with st.expander("Import Theme Set", expanded=False):
                    uploaded_theme_set = st.file_uploader(
                        "Upload theme set JSON", 
                        type=["json"],
                        key=f"import_theme_set_{active_question}",
                        help="Upload a theme set previously exported from this app. Useful for applying the same themes to a different dataset."
                    )
                    
                    if uploaded_theme_set is not None:
                        # Show file info to confirm upload worked
                        st.info(f"File uploaded: {uploaded_theme_set.name}")
                        
                        if st.button("Import", key=f"import_button_{active_question}"):
                            imported_id = import_theme_set(uploaded_theme_set, active_question)
                            if imported_id:
                                st.success(f"Theme set imported successfully!")
                                st.rerun()

                has_theme_sets = (active_question in st.session_state["reasoning"]["theme_sets_by_question"] and 
                                st.session_state["reasoning"]["theme_sets_by_question"].get(active_question, {}))
                is_processed = active_question in st.session_state["data"]["processed_questions"]
                
                if is_processed or has_theme_sets:
                    manage_theme_sets(active_question)

###############################################################################
# 2. STATE LOADING AND SAVING
###############################################################################
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {
                "_type": "pandas.DataFrame",
                "data": base64.b64encode(obj.to_csv(index=True).encode()).decode()
            }
       # Handle pandas Series
        if isinstance(obj, pd.Series):
            return {
                "_type": "pandas.Series", 
                "data": base64.b64encode(obj.to_csv(index=True).encode()).decode()
            }
        
        if isinstance(obj, set):
            return {
                "_type": "set",
                "data": list(obj)  # Convert set to list for JSON serialization
            }

        # Handle NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Handle Pydantic models - call model_dump() (v2) or dict() (v1) depending on version
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return {
                "_type": "pydantic.model",
                "class": obj.__class__.__name__,
                "data": obj.model_dump()
            }
        elif hasattr(obj, "dict") and callable(obj.dict):  # Pydantic v1
            return {
                "_type": "pydantic.model",
                "class": obj.__class__.__name__,
                "data": obj.dict()
            }
            
        # Default handling
        return super().default(obj)
            
def custom_json_decoder(obj):
    if isinstance(obj, dict) and "_type" in obj:
        if obj["_type"] == "pandas.DataFrame":
            return pd.read_csv(io.StringIO(base64.b64decode(obj["data"]).decode()), index_col=0)
        if obj["_type"] == "pandas.Series":
            # Reconstruct a true Series from the serialized CSV
            _df = pd.read_csv(io.StringIO(base64.b64decode(obj["data"]).decode()), index_col=0)
            if isinstance(_df, pd.DataFrame) and _df.shape[1] == 1:
                return _df.iloc[:, 0]
            # Fallback: squeeze columns without collapsing to scalar if possible
            _squeezed = _df.squeeze("columns")
            return _squeezed
        if obj["_type"] == "set":
            return set(obj["data"])
        if obj["_type"] == "pydantic.model":
            return obj["data"]

    return obj

# Widget state keys/prefixes should not be rehydrated from saved analysis state.
# These are UI-only and can break if Streamlit attempts to restore stale widget IDs.
WIDGET_STATE_KEYS = (
    "add_theme_button",
    "delete_theme_set_button",
    "rename_theme_set_button",
    "save_state_filename",
    "openai_api_key_input",
    # Back-compat: this was the implicit widget key before we added an explicit one.
    "For OpenAI models, enter your API Key",
    "quit_button",
    "reasoning_model_selected",
    "classification_model_selected",
    "unique_id_column_selector",
    "reset_ui_button",
)
WIDGET_STATE_PREFIXES = (
    "delete_theme_",
    "theme_name_",
    "theme_description_",
    "theme_examples_",
    "theme_set_selector_",
    "filter_",
    "show_filters_",
)

# Minimal, targeted normalization for older state files
def _normalize_sparse_results_structure(sparse):
    """
    Normalize sparse_results to dict[index -> list[str]].
    Supports:
      - dict of index -> list/str
      - pandas Series (converted to dict)
      - pandas DataFrame: single-column or dense 0/1 matrix (old format)
    """
    # Dict: ensure list-of-strings
    if isinstance(sparse, dict):
        out = {}
        for k, v in sparse.items():
            if isinstance(v, list):
                out[k] = [str(x) for x in v]
            elif pd.isna(v):
                continue
            else:
                out[k] = [str(v)]
        return out

    # Series -> dict
    if isinstance(sparse, pd.Series):
        return _normalize_sparse_results_structure(sparse.to_dict())

    # DataFrame
    if isinstance(sparse, pd.DataFrame):
        # Single column -> Series dict
        if sparse.shape[1] == 1:
            return _normalize_sparse_results_structure(sparse.iloc[:, 0].to_dict())
        # Known schema: index/themes columns
        if "index" in sparse.columns and "themes" in sparse.columns:
            return _normalize_sparse_results_structure(dict(zip(sparse["index"], sparse["themes"])))
        # Dense 0/1 matrix: collect truthy columns per row
        candidate_cols = [
            c for c in sparse.columns
            if set(pd.unique(sparse[c].dropna())).issubset({0, 1, 0.0, 1.0, True, False})
        ]
        if not candidate_cols:
            return {}
        result = {}
        for idx, row in sparse[candidate_cols].iterrows():
            active = [str(c) for c in candidate_cols if pd.notna(row[c]) and bool(row[c])]
            if active:
                result[idx] = active
        return result

    # Unknown shapes
    return {}

def _normalize_loaded_classification_results():
    cls = st.session_state.get("classification", {})
    # Normalize results_by_question
    rbq = cls.get("results_by_question", {})
    for q, theme_sets in list(rbq.items()):
        for ts_id, res in list(theme_sets.items()):
            rbq[q][ts_id] = _normalize_sparse_results_structure(res)
    # Normalize current sparse_results if present
    if "sparse_results" in cls:
        cls["sparse_results"] = _normalize_sparse_results_structure(cls["sparse_results"]) 

def save_state_json():
    state = dict(st.session_state)
    state.pop("openai_client", None)
    # Never persist secrets in exported analysis state.
    state.pop("openai_api_key_input", None)
    # Back-compat: exclude legacy implicit widget key (pre explicit `key=`).
    state.pop("For OpenAI models, enter your API Key", None)

    # Remove file uploader objects which can't be serialized
    keys_to_remove = []
    for key in state.keys():
        if key.startswith("import_theme_set_") or key == "state_file_uploader" or key == "csv_file_uploader":
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        state.pop(key, None)

    # Serialize to JSON with custom encoder
    json_str = json.dumps(state, cls=CustomJSONEncoder)
    return json_str

def load_state_json(file_object=None):
    try:
        if file_object is None:
            return False

        # Read JSON from uploaded file
        json_str = file_object.getvalue().decode('utf-8')
        # Deserialize with custom decoder
        loaded_state = json.loads(json_str, object_hook=custom_json_decoder)
        
        # Remove any keys from the loaded state that match widget keys or prefixes
        for key in list(loaded_state.keys()):
            if key in WIDGET_STATE_KEYS or any(key.startswith(prefix) for prefix in WIDGET_STATE_PREFIXES):
                del loaded_state[key]
        
        # Update st.session_state with the filtered state
        for key, value in loaded_state.items():
            st.session_state[key] = value

        # Backward-compat for older state files
        if "reasoning_temperature_slider" in st.session_state and "reasoning_temperature" not in st.session_state:
            st.session_state["reasoning_temperature"] = st.session_state["reasoning_temperature_slider"]
        st.session_state.pop("reasoning_temperature_slider", None)

        # Ensure all required session keys exist.
        initialise_session_states()
        # Reset active question–related state and widget keys
        reset_for_new_question()

        # Normalize potentially old/variant shapes in classification results
        _normalize_loaded_classification_results()

        return True
    except Exception as e:
        st.error(f"Failed to load state: {e}")
        return False

def export_theme_set(question, theme_set_id):
    """Export a single theme set to JSON format."""
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
    theme_set = theme_sets.get(theme_set_id)
    
    if not theme_set:
        return None
    
    # Create export data with selected fields
    export_data = {
        "name": theme_set["name"],
        "themes": theme_set["themes"],
        "additional_instructions": theme_set.get("additional_instructions", ""),
        "created_at": theme_set.get("created_at"),
        "exported_from_question": question,
        "export_timestamp": int(time.time())
    }
    
    return json.dumps(export_data, indent=2)

def rebuild_final_results_from_state():
    """
    Rebuild the final results DataFrame from the current state so exports
    reflect what's shown in the charts, without requiring an explicit save.

    Uses:
      - st.session_state["classification"]["results_by_question"] as the source of
        coded results for each question and theme set
      - st.session_state["reasoning"]["theme_sets_by_question"] to obtain current
        theme set names and theme names

    Returns:
        pd.DataFrame | None: Rebuilt final results, or None if nothing to export
    """
    # Ensure there is classification data to export
    results_by_question = st.session_state["classification"].get("results_by_question", {})
    if not results_by_question:
        return None

    survey_df = st.session_state["data"].get("survey")
    if survey_df is None or survey_df.empty:
        return None

    # Create base DataFrame with unique ID or row_id
    unique_id_col = st.session_state["data"].get("unique_id_column")
    if unique_id_col:
        final_df = survey_df[[unique_id_col]].copy()
    else:
        final_df = pd.DataFrame(index=survey_df.index)
        final_df["row_id"] = range(1, len(final_df) + 1)

    final_index = final_df.index
    new_columns = {}

    # For each question with classification results, add original text and theme columns
    theme_sets_by_question = st.session_state["reasoning"].get("theme_sets_by_question", {})

    for question, theme_sets_results in results_by_question.items():
        # Add original question column
        if question in survey_df.columns:
            new_columns[question] = survey_df[question].reindex(final_index)

        # For each theme set with results
        question_theme_sets = theme_sets_by_question.get(question, {})

        for theme_set_id, sparse_results in theme_sets_results.items():
            theme_set = question_theme_sets.get(theme_set_id)
            if not theme_set:
                # If the theme set metadata isn't present, skip gracefully
                continue

            theme_set_name = theme_set.get("name", str(theme_set_id))
            themes = theme_set.get("themes", [])
            if not themes:
                continue

            if sparse_results is None:
                # Nothing to place for this theme set
                continue

            theme_names = [theme.get("name") for theme in themes if theme.get("name")]
            theme_to_indices = {name: [] for name in theme_names}
            theme_name_set = set(theme_to_indices)

            # Build a lookup to reconcile index key types (e.g., str vs int)
            index_lookup = {str(ix): ix for ix in final_index}
            for idx_key, response_themes in sparse_results.items():
                resolved_idx = index_lookup.get(str(idx_key))
                if resolved_idx is None:
                    continue
                for theme_name in response_themes:
                    if theme_name in theme_name_set:
                        theme_to_indices[theme_name].append(resolved_idx)

            for theme in themes:
                theme_name = theme.get("name")
                if not theme_name:
                    continue

                qualified_name = f"{question}_{theme_set_name}_{theme_name}"
                column_values = pd.Series(0, index=final_index)
                if theme_to_indices[theme_name]:
                    column_values.loc[theme_to_indices[theme_name]] = 1
                new_columns[qualified_name] = column_values

    # Avoid fragmented DataFrame by concatenating new columns once
    if new_columns:
        new_columns_df = pd.DataFrame(new_columns, index=final_index)
        existing_cols = list(final_df.columns)
        ordered_columns = []
        for col in existing_cols:
            if col in new_columns_df.columns:
                ordered_columns.append(new_columns_df[col])
            else:
                ordered_columns.append(final_df[col])
        for col in new_columns_df.columns:
            if col not in existing_cols:
                ordered_columns.append(new_columns_df[col])

        final_df = pd.concat(ordered_columns, axis=1)

    # Persist rebuilt results into session for consistency with rest of app
    st.session_state["data"]["final_results"] = final_df
    return final_df

def export_to_csv():
    """
    Export all processed classification results to CSV.
    Rebuilds the final results from current state so it reflects charts.
    
    Returns:
        str: CSV data as string, or None if no data available
    """
    # Rebuild from current classification results and theme sets
    final_results = rebuild_final_results_from_state()

    if final_results is None or final_results.empty:
        return None

    return final_results.to_csv(index=False)

###############################################################################
# 3. QUESTION SELECTION
###############################################################################
def select_question_to_analyse():
    """
    Displays question selection dropdown.
    
    Returns:
      The selected column name, or None if no column is selected.
    """
    
    df = st.session_state["data"]["survey"]
    columns = list(df.columns)
    
    processed_questions = st.session_state["data"]["processed_questions"]
    column_options = ["No column selected"]
    for col in columns:
        if col in processed_questions:
            column_options.append(f"{col} (processed)")
        else:
            column_options.append(col)
    
    st.subheader("Select Question to analyse")

    # Question selection dropdown
    chosen_option = st.selectbox("Column to analyse", options=column_options, key="question_select", help="Select the open-ended question column whose responses you want to theme-code.")
    chosen_col = None
    
    if chosen_option != "No column selected":
        chosen_col = chosen_option.split(" (processed)")[0]

        available_columns = [col for col in columns 
                             if col != chosen_col 
                             and col != st.session_state["data"]["unique_id_column"]]

        dirty = is_theme_editor_dirty()
        if dirty:
            st.warning("Unsaved theme edits detected. Save Changes or Revert Changes (in the theme editor) before switching questions.")

        if st.button("Confirm column selection", disabled=dirty):
            st.session_state["data"]["active_question"] = chosen_col
            get_initial_theme_generation_sample(chosen_col)

            if "classification_batches" in st.session_state:
                del st.session_state["classification_batches"]

            st.rerun()
    
    return chosen_col

def manage_theme_sets(question):
    """
    Displays theme set management UI for the given question.
    
    Parameters:
      question: The question column name.
    """
    if not question:
        return
    
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)

    is_processed = question in st.session_state["data"]["processed_questions"]
    message = "This question has been processed before." if is_processed else "You have started analysis on this question."
    st.info(f"{message} What do you want to do?")

    # Get available theme sets for this question
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})

    # initialize theme set variable
    selected_theme_set = "Please select"
    
    if theme_sets:
        # Create a list of theme set names and IDs for the dropdown
        theme_set_options = ["Please select"]
        theme_set_ids = {}
        for ts_id, ts in theme_sets.items():
            theme_set_options.append(ts["name"])
            theme_set_ids[ts["name"]] = ts_id
        
        # Add a dropdown to select a theme set
        active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
        active_theme_set = theme_sets.get(active_theme_set_id, {})
        active_theme_set_name = active_theme_set.get("name", "") if active_theme_set else ""
        
        default_index = 0  # Default to "Please select"
        if active_theme_set_name:
            default_index = theme_set_options.index(active_theme_set_name)

        if "previous_theme_set_selection" not in st.session_state:
            st.session_state["previous_theme_set_selection"] = {}
        previous_selection = st.session_state["previous_theme_set_selection"].get(question)

        selected_theme_set = st.selectbox("Select Theme Set", options=theme_set_options, index=default_index, key=f"theme_set_selector_{question}")

        if (previous_selection is not None and 
            previous_selection != selected_theme_set and 
            selected_theme_set != "Please select"):
            
            selected_theme_set_id = theme_set_ids[selected_theme_set]
            if is_theme_editor_dirty():
                st.warning("Unsaved theme edits detected. Save Changes or Revert Changes before switching theme sets.")
                # Revert the selector back to the previous selection
                selector_key = f"theme_set_selector_{question}"
                revert_to = previous_selection if previous_selection else active_theme_set_name if active_theme_set_name else "Please select"
                st.session_state[selector_key] = revert_to
            else:
                if switch_active_theme_set(question, selected_theme_set_id):
                    # Update previous selection to avoid re-triggering
                    st.session_state["previous_theme_set_selection"][question] = selected_theme_set
                    st.success(f"Switched to theme set: {selected_theme_set}")
                    st.rerun()  # Force a rerun to update the UI after the switch
        else:
            # Record current selection if no change
            st.session_state["previous_theme_set_selection"][question] = selected_theme_set
    
    createnewsetbutton = st.columns(1)[0]  # Single column for "Please select" case
    if selected_theme_set == "Please select":
        with createnewsetbutton:
            if st.button("Create New Theme Set"):
                theme_set_id = create_theme_set(question)
                if theme_set_id:
                    st.session_state["just_created_theme_set"] = True
                    # Update the previous selection with the new theme set's name immediately
                    new_theme_name = st.session_state["reasoning"]["theme_sets_by_question"][question][theme_set_id]["name"]
                    st.session_state.setdefault("previous_theme_set_selection", {})[question] = new_theme_name
                    st.success("Created new theme set")
                    st.rerun()

    if selected_theme_set != "Please select":# Rest of theme set management (create, delete, rename)
        
        createnewsetbutton, duplicatesetbutton, deletesetbutton = st.columns(3)
        with createnewsetbutton:
            if st.button("Create New Theme Set"):
                theme_set_id = create_theme_set(question)
                if theme_set_id:
                    st.session_state["just_created_theme_set"] = True
                    # Update the previous selection with the new theme set's name immediately
                    new_theme_name = st.session_state["reasoning"]["theme_sets_by_question"][question][theme_set_id]["name"]
                    st.session_state.setdefault("previous_theme_set_selection", {})[question] = new_theme_name
                    st.success("Created new theme set")
                    st.rerun()
        with duplicatesetbutton:
            if theme_sets:
                if st.button("Duplicate Theme Set", key="duplicate_theme_set_button"):
                    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
                    if active_theme_set_id:
                        new_theme_set_id = duplicate_theme_set(question, active_theme_set_id)
                        if new_theme_set_id:
                            # Switch to the new theme set
                            if switch_active_theme_set(question, new_theme_set_id):
                                # Update the previous selection with the new theme set's name
                                new_theme_name = st.session_state["reasoning"]["theme_sets_by_question"][question][new_theme_set_id]["name"]
                                st.session_state.setdefault("previous_theme_set_selection", {})[question] = new_theme_name
                                st.success("Theme set duplicated successfully!")
                                st.rerun()
        with deletesetbutton:
            if theme_sets:
                if st.button("Delete selected theme set", key="delete_theme_set_button"):
                    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
                    if active_theme_set_id:
                        if delete_theme_set(question, active_theme_set_id):
                            st.rerun()
        
        if theme_sets:
            with st.expander("Rename Current Theme Set"):
                active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
                active_theme_set = theme_sets.get(active_theme_set_id, {})
                active_theme_set_name = active_theme_set.get("name", "") if active_theme_set else ""
                
                new_name = st.text_input("New Name", value=active_theme_set_name, key="theme_set_rename_input")
                if st.button("Rename Theme Set", key="rename_theme_set_button"):
                    if rename_theme_set(question, active_theme_set_id, new_name):
                        st.success(f"Theme set renamed to: {new_name}")
                        st.rerun()
    
        # View additional instructions for the active theme set (edit is opt-in)
        active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
        if active_theme_set_id:
            ts = theme_sets.get(active_theme_set_id, {})
            with st.expander("Theme set instructions"):
                current_instructions = ts.get("additional_instructions", "")
                st.text_area(
                    "Instructions (read-only)",
                    value=current_instructions,
                    key=f"theme_set_additional_instructions_view_{active_theme_set_id}",
                    disabled=True
                )

        # Get the active theme set ID
        active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
        
        # Export button
        if active_theme_set_id and st.button("Export Theme Set", key="export_theme_set_button", help="Downloads the selected theme set as JSON, including theme names, descriptions, examples, and instructions."):
            export_json = export_theme_set(question, active_theme_set_id)
            if export_json:
                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_theme_set = "".join(c for c in selected_theme_set if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_question}_{safe_theme_set}_{timestamp}.json"
                
                st.download_button(
                    label="Download Theme Set JSON",
                    data=export_json,
                    file_name=filename,
                    mime="application/json",
                    key=f"download_theme_set_{active_theme_set_id}"  # Fixed: use active_theme_set_id
                )

    # Load the active theme set's themes
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)

def show_sample_data():
    """
    Displays a preview of the loaded survey data in the Streamlit interface.
    
    Shows the first 5 rows of the dataset and the overall dimensions
    (number of rows and columns) to give users a quick overview of the
    data structure without overwhelming the interface.
    
    No parameters or return values.
    """
    df = st.session_state["data"]["survey"]
    st.subheader("Data overview")
    st.write("Sample of data (first 5 rows):")
    st.dataframe(df.head(5), use_container_width=True)
    st.write(f"Data dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

def _render_info_card(inner_html: str):
    st.markdown(
        f"""
        <div style="background-color:white; padding:15px; border-radius:5px; border:1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom:15px;">
            {inner_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def import_theme_set(file_object, question):
    """Import a theme set from JSON file and create new theme set."""
    try:
        # Read and parse JSON
        json_str = file_object.getvalue().decode('utf-8')
        data = json.loads(json_str)
        
        # Validate structure
        required_fields = ["name", "themes"]
        for field in required_fields:
            if field not in data:
                st.error(f"Invalid theme set file: missing '{field}' field")
                return None
        
        # Validate themes structure
        if not isinstance(data["themes"], list):
            st.error("Invalid theme set file: 'themes' must be a list")
            return None
            
        for i, theme in enumerate(data["themes"]):
            if not isinstance(theme, dict):
                st.error(f"Invalid theme {i+1}: must be a dictionary")
                return None
            if "name" not in theme or "description" not in theme:
                st.error(f"Invalid theme {i+1}: missing name or description")
                return None
            if "examples" not in theme:
                theme["examples"] = []  # Default to empty list
                
        # Create new theme set
        theme_set_id = create_theme_set(question)
        
        # Update with imported data
        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"][question]
        theme_sets[theme_set_id]["name"] = f"{data['name']} (imported)"
        theme_sets[theme_set_id]["themes"] = data["themes"]
        theme_sets[theme_set_id]["additional_instructions"] = data.get("additional_instructions", "")
        
        st.session_state["reasoning"]["current_themes"] = data["themes"]

        switch_active_theme_set(question, theme_set_id)

        st.write(f"Imported {len(data['themes'])} themes")
        st.write(f"Theme set now has {len(theme_sets[theme_set_id]['themes'])} themes")
        
        # Update the previous selection to avoid re-triggering
        new_theme_name = st.session_state["reasoning"]["theme_sets_by_question"][question][theme_set_id]["name"]
        st.session_state.setdefault("previous_theme_set_selection", {})[question] = new_theme_name

        return theme_set_id
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e}")
        return None
    except Exception as e:
        st.error(f"Error importing theme set: {e}")
        return None

###############################################################################
# 4. THEME SET MANAGEMENT
###############################################################################
def is_current_theme_set_accepted():
    """Check if the currently active theme set has been accepted."""
    question = st.session_state["data"]["active_question"]
    if not question:
        return False
    
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if not active_theme_set_id:
        return False
    
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
    theme_set = theme_sets.get(active_theme_set_id, {})
    return theme_set.get("accepted", False)

def create_theme_set(question):
    """
    Creates a new theme set for the given question with validation for unique names.
    
    Parameters:
      question: The question column name.
      
    Returns:
      The ID of the newly created theme set.
    """
    # Generate a unique ID for the theme set
    timestamp = int(time.time())
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    theme_set_id = f"ts_{timestamp}_{random_str}"
    
    # Create the theme set structure
    theme_set = {
        "id": theme_set_id,
        "name": "Temp name",
        "themes": [],
        "created_at": timestamp,
        "accepted": False,
        "additional_instructions": st.session_state.get("additional_theme_generation_instructions", ""),
        "grouping_variable": None,   # Column name used for grouping (e.g. "ad_column")
        "valid_groups": [],          # List of valid group values (e.g. ["Ad A", "Ad B"])
    }
    
    # Initialize the question in theme_sets_by_question if it doesn't exist
    if question not in st.session_state["reasoning"]["theme_sets_by_question"]:
        st.session_state["reasoning"]["theme_sets_by_question"][question] = {}
    
    # Add the theme set to the question
    st.session_state["reasoning"]["theme_sets_by_question"][question][theme_set_id] = theme_set
    
    # Set as active theme set for the question
    st.session_state["reasoning"]["active_theme_set_ids"][question] = theme_set_id
    
    # Initialize current_themes as empty list
    st.session_state["reasoning"]["current_themes"] = []
    st.session_state["classification"]["sparse_results"] = {}
    st.session_state["classification"]["justifications"] = {}
    
    st.session_state['newly_created_theme_set'] = True

    return theme_set_id

def switch_active_theme_set(question, theme_set_id):
    """
    Switches the active theme set for a question.
    
    Parameters:
      question: The question column name.
      theme_set_id: The ID of the theme set to activate.
      
    Returns:
      True if successful, False otherwise.
    """
    # Check if the theme set exists
    if (question not in st.session_state["reasoning"]["theme_sets_by_question"] or 
        theme_set_id not in st.session_state["reasoning"]["theme_sets_by_question"][question]):
        st.error(f"Theme set not found for question '{question}'.")
        return False
    
    # Save current themes to current theme set if there is one
    current_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if current_theme_set_id:
        current_theme_set = st.session_state["reasoning"]["theme_sets_by_question"][question][current_theme_set_id]
        current_theme_set["themes"] = st.session_state["reasoning"]["current_themes"]

        # Also save current justifications to the theme set we're switching from
        if st.session_state["classification"]["justifications"]:
            if "justifications_by_theme_set" not in st.session_state["classification"]:
                st.session_state["classification"]["justifications_by_theme_set"] = {}
            
            if question not in st.session_state["classification"]["justifications_by_theme_set"]:
                st.session_state["classification"]["justifications_by_theme_set"][question] = {}
                
            st.session_state["classification"]["justifications_by_theme_set"][question][current_theme_set_id] = \
                st.session_state["classification"]["justifications"].copy()
    
    # Set the new active theme set
    st.session_state["reasoning"]["active_theme_set_ids"][question] = theme_set_id
    
    # Load themes from the new active theme set
    new_theme_set = st.session_state["reasoning"]["theme_sets_by_question"][question][theme_set_id]
    st.session_state["reasoning"]["current_themes"] = new_theme_set["themes"]
    
    # Load classification results if available
    if (question in st.session_state["classification"]["results_by_question"] and 
        theme_set_id in st.session_state["classification"]["results_by_question"][question]):
        st.session_state["classification"]["sparse_results"] = st.session_state["classification"]["results_by_question"][question][theme_set_id]
    else:
        st.session_state["classification"]["sparse_results"] = {}
    
    # Load justifications for this theme set if available, otherwise reset
    if "justifications_by_theme_set" in st.session_state["classification"] and \
       question in st.session_state["classification"]["justifications_by_theme_set"] and \
       theme_set_id in st.session_state["classification"]["justifications_by_theme_set"][question]:
        st.session_state["classification"]["justifications"] = \
            st.session_state["classification"]["justifications_by_theme_set"][question][theme_set_id].copy()
    else:
        st.session_state["classification"]["justifications"] = {}

    # Update theme counts for visualization
    update_theme_counts()

    if "previous_theme_set_selection" not in st.session_state:
        st.session_state["previous_theme_set_selection"] = {}
    st.session_state["previous_theme_set_selection"][question] = new_theme_set.get("name", "")

    if "classification_batches" in st.session_state:
        del st.session_state["classification_batches"]

    return True

def delete_theme_set(question, theme_set_id):
    # Check if the theme set exists
    if (question not in st.session_state["reasoning"]["theme_sets_by_question"] or 
        theme_set_id not in st.session_state["reasoning"]["theme_sets_by_question"][question]):
        st.error(f"Theme set not found for question '{question}'.")
        return False
    
    # Get all theme sets for this question
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"][question]
    
    # Check if this is the last theme set
    if len(theme_sets) <= 1:
        st.warning("Cannot delete the last theme set for a question.")
        return False
    
    # Store theme set name before deletion
    theme_set_name = theme_sets[theme_set_id]["name"]
    
    # IMPORTANT: If the deleted theme set was active, switch to another one FIRST (before deleting)
    if st.session_state["reasoning"]["active_theme_set_ids"].get(question) == theme_set_id:
        # Find alternative theme set ID
        next_theme_set_id = next((ts_id for ts_id in theme_sets.keys() if ts_id != theme_set_id), None)
        if next_theme_set_id:
            # Update active theme set ID directly (don't call switch function yet)
            st.session_state["reasoning"]["active_theme_set_ids"][question] = next_theme_set_id
    
    # Now delete the theme set
    del theme_sets[theme_set_id]
    
    # Now load the new active theme set data
    current_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if current_theme_set_id and current_theme_set_id in theme_sets:
        # Load themes from the new active theme set
        st.session_state["reasoning"]["current_themes"] = theme_sets[current_theme_set_id]["themes"]
        
        # Load classification results if available
        if (question in st.session_state["classification"]["results_by_question"] and 
            current_theme_set_id in st.session_state["classification"]["results_by_question"][question]):
            st.session_state["classification"]["sparse_results"] = st.session_state["classification"]["results_by_question"][question][current_theme_set_id]
        else:
            st.session_state["classification"]["sparse_results"] = {}
        
        # Reset justifications
        st.session_state["classification"]["justifications"] = {}
    
    # Remove classification results for the deleted theme set
    if (question in st.session_state["classification"]["results_by_question"] and 
        theme_set_id in st.session_state["classification"]["results_by_question"][question]):
        del st.session_state["classification"]["results_by_question"][question][theme_set_id]
    
    st.success(f"Theme set '{theme_set_name}' deleted successfully.")
    return True

def get_current_theme_set(question):
    """
    Get the currently active theme set for a question.
    
    Parameters:
      question: The question column name.
      
    Returns:
      A dictionary containing the active theme set details, or None if not found.
    """
    if not question:
        return None
    
    theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if not theme_set_id:
        return None
    
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
    return theme_sets.get(theme_set_id)

def rename_theme_set(question, theme_set_id, new_name):
    """
    Renames a theme set with validation for unique names.
    
    Parameters:
      question: The question column name.
      theme_set_id: The ID of the theme set to rename.
      new_name: The new name for the theme set.
      
    Returns:
      True if successful, False otherwise.
    """
    # Validate input
    if not question or not theme_set_id or not new_name or new_name.strip() == "":
        st.error("Theme set name cannot be empty.")
        return False
    
    # Check if the theme set exists
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
    if theme_set_id not in theme_sets:
        st.error(f"Theme set not found for question '{question}'.")
        return False
    
    # Check if the name is already in use by another theme set for this question
    for ts_id, ts in theme_sets.items():
        if ts_id != theme_set_id and ts.get("name") == new_name:
            st.error(f"Theme set name '{new_name}' is already in use for this question.")
            return False
    
    # Store old name for reference
    old_name = theme_sets[theme_set_id].get("name", "")
    
    # Rename the theme set
    theme_sets[theme_set_id]["name"] = new_name
    
    # Update previous_theme_set_selection
    if "previous_theme_set_selection" in st.session_state and question in st.session_state["previous_theme_set_selection"]:
        if st.session_state["previous_theme_set_selection"][question] == old_name:
            st.session_state["previous_theme_set_selection"][question] = new_name
    
    # Update theme set name references in final results if needed
    if (st.session_state["data"]["final_results"] is not None and 
        old_name and old_name != new_name):
        final_df = st.session_state["data"]["final_results"]
        for col in final_df.columns:
            # Look for columns with the pattern: question_oldname_theme
            if col.startswith(f"{question}_{old_name}_"):
                # Extract the theme part
                theme_part = col.split(f"{question}_{old_name}_")[1]
                # Create new column name
                new_col = f"{question}_{new_name}_{theme_part}"
                # Rename the column
                final_df.rename(columns={col: new_col}, inplace=True)
    
    return True

def duplicate_theme_set(question, theme_set_id):
    """
    Duplicates an existing theme set with all its themes.
    
    Parameters:
      question: The question column name.
      theme_set_id: The ID of the theme set to duplicate.
      
    Returns:
      The ID of the newly created duplicate theme set, or None if failed.
    """
    
    # Check if the theme set exists
    if (question not in st.session_state["reasoning"]["theme_sets_by_question"] or 
        theme_set_id not in st.session_state["reasoning"]["theme_sets_by_question"][question]):
        st.error(f"Theme set not found for question '{question}'.")
        return None
    
    # Get the original theme set
    original_theme_set = st.session_state["reasoning"]["theme_sets_by_question"][question][theme_set_id]
    
    # Generate a unique ID for the new theme set
    timestamp = int(time.time())
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    new_theme_set_id = f"ts_{timestamp}_{random_str}"
    
    # Create a deep copy of the original theme set
    new_theme_set = copy.deepcopy(original_theme_set)
    
    # Update properties
    new_theme_set["id"] = new_theme_set_id
    new_theme_set["name"] = f"{original_theme_set['name']} (copy)"
    new_theme_set["created_at"] = timestamp
    
    # Add the new theme set to the session state
    st.session_state["reasoning"]["theme_sets_by_question"][question][new_theme_set_id] = new_theme_set
    
    return new_theme_set_id

###############################################################################
# 5. MODEL CALLING
###############################################################################
class LLMMessage(BaseModel):
    role: str
    content: str

class Theme(BaseModel):
    name: str
    description: str
    examples: List[str] = Field(default_factory=list)
    group: Optional[str] = Field(default=None, description="Group this theme belongs to (None = universal / 'No group')")

class ThemeList(BaseModel):
    root: List[Theme]

class ThemeEditResult(BaseModel):
    name: str
    description: str
    examples: List[str] = Field(default_factory=list)
    mark_for_deletion: bool = Field(default=False)
    deletion_reason: Optional[str] = Field(default=None)

class ThemeEditResults(BaseModel):
    root: List[ThemeEditResult]

class ClassificationResult(BaseModel):
    index: int
    themes: List[str]
    justifications: Dict[str, List[str]] = Field(default_factory=dict)

class ClassificationResults(BaseModel):
    root: List[ClassificationResult]

class LLMCall(BaseModel):
    messages: list[LLMMessage]
    model_name: str = Field("OpenAI: gpt-4.1-2025-04-14", description="Selected model string (e.g. 'OpenAI: gpt-4.1-2025-04-14')")
    temperature: Optional[float] = Field(0.0, description="Model temperature for creative responses.")
    format: Optional[Type[BaseModel]] = None # Accept either a Pydantic model class or None

class LLMResponse(BaseModel):
    content: str
    usage: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ThemeEvaluation(BaseModel):
    score: int
    reason: Optional[str] = Field(default=None)

class ThemeEvaluations(BaseModel):
    root: Dict[str, ThemeEvaluation]

class ExampleRating(BaseModel):
    score: int
    reason: Optional[str] = Field(default=None)

class ExampleRatings(BaseModel):
    root: Dict[str, ExampleRating]

class DuplicateVerification(BaseModel):
    is_duplicate: int

@st.cache_data(ttl=60)
def cached_openai_models(_client) -> list[str]:
    if _client is None:
        return []
    try:
        models = _client.models.list()
        return [m.id for m in models.data]
    except Exception:
        return []

def get_openai_models():
    model_ids = cached_openai_models(st.session_state.get("openai_client"))
    accepted_models = ["gpt-5.2", "gpt-5-mini",
                       "o3", "gpt-4.1-2025-04-14"]

    return [m for m in accepted_models if m in model_ids]

def get_ollama_models():
    return ["qwen3:4b", "gemma3:4b"]

def get_all_models():
    openai_models = get_openai_models()
    ollama_models = get_ollama_models()
    return [f"OpenAI: {m}" for m in openai_models] + [f"Ollama: {m}" for m in ollama_models]

class BaseProcessingConfig:
    """Base configuration class for LLM processing operations."""
    
    def __init__(self, session_state=None):
        """Initialize with common settings from session state."""
        ss = session_state or {}
        
        # OpenAI client (common to all operations)
        self.openai_client = ss.get("openai_client")
        self.reasoning_effort = ss.get("reasoning_effort", None)  # "none" | "low" | "medium" | "high" | "xhigh"
        self.verbosity = ss.get("verbosity", None)                # "low" | "medium" | "high"
    
    def get_effective_temperature(self):
        """Returns the temperature to use or None if model doesn't support it"""
        return self.temperature if self.model_name not in MODELS_WITHOUT_TEMPERATURE else None

class ReasoningConfig(BaseProcessingConfig):
    """Configuration for theme generation operations."""
    
    def __init__(self, session_state=None):
        """Initialize with reasoning-specific settings."""
        super().__init__(session_state)
        ss = session_state or {}
        
        # Model parameters for reasoning
        self.model_name = ss.get("reasoning_model_selected", "OpenAI: gpt-5.2")
        self.temperature = ss.get("reasoning_temperature", 0.8)
        
        # Additional instructions for theme generation
        self.additional_instructions = ss.get("additional_theme_generation_instructions", "")

class ClassificationConfig(BaseProcessingConfig):
    """Configuration for classification operations."""
    
    def __init__(self, session_state=None):
        """Initialize with classification-specific settings."""
        super().__init__(session_state)
        ss = session_state or {}
        
        # Model parameters for classification
        self.model_name = ss.get("classification_model_selected", "OpenAI: o3")
        self.temperature = ss.get("classification_temperature", 0.0)
        
        # Additional instructions for classification
        self.additional_instructions = ss.get("additional_classification_instructions", "")
        
        # Batch size for classification
        self.batch_size = ss.get("batch_size", 3)

        # Whether to include examples in classification prompt
        self.include_examples = ss.get("include_examples_in_classification", True)

def render_gpt5_ui_if_applicable(selected_model: str, key_prefix: str = ""):
    base = selected_model.split("OpenAI: ")[1] if selected_model.startswith("OpenAI: ") else selected_model
    if not base.startswith("gpt-5"):
        st.session_state["reasoning_effort"] = None
        st.session_state["verbosity"] = None
        return

    st.caption("GPT-5 options")
    effort_key = f"{key_prefix}reasoning_effort_select"
    verbosity_key = f"{key_prefix}verbosity_select"
    effort_options = ["(model default)", "low", "medium", "high"]
    if base.startswith("gpt-5.2"):
        effort_options.append("xhigh")
    effort_choice = st.selectbox(
        "Reasoning effort (optional)",
        effort_options,
        key=effort_key,
        index=1 if effort_key not in st.session_state else
              effort_options.index(st.session_state[effort_key]),
    )
    verbosity_choice = st.selectbox(
        "Verbosity (optional)",
        ["(model default)", "low", "medium", "high"],
        key=verbosity_key,
        index=1 if verbosity_key not in st.session_state else
              ["(model default)", "low", "medium", "high"].index(st.session_state[verbosity_key]),
    )
    st.session_state["reasoning_effort"] = None if effort_choice == "(model default)" else effort_choice
    st.session_state["verbosity"] = None if verbosity_choice == "(model default)" else verbosity_choice

MODELS_WITHOUT_TEMPERATURE = (
    "OpenAI: o1",
    "OpenAI: o3-mini",
    "OpenAI: o3",
    "OpenAI: gpt-5.2",
    "OpenAI: gpt-5-mini-2025-08-07",
)
def model_supports_temperature(model_name: str) -> bool:
    return model_name not in MODELS_WITHOUT_TEMPERATURE

def call_llm(request: LLMCall, config: BaseProcessingConfig) -> LLMResponse:
    """
    A unified LLM call that routes the request to either the OpenAI API or the local Ollama model.

    Parameters:
      request: An LLMCall object containing messages, model selection, temperature, and output schema.
      config: Configuration object containing the OpenAI client and other settings.
    
    Returns:
      An LLMResponse object with the content returned from the LLM.
    """
    if request.model_name.startswith("OpenAI: "):
        backend = "openai"
        model = request.model_name.split("OpenAI: ")[1]
    elif request.model_name.startswith("Ollama: "):
        backend = "ollama"
        model = request.model_name.split("Ollama: ")[1]
    else:
        return LLMResponse(content=f"Error: Invalid model format '{request.model_name}'. Expected 'OpenAI: model_name' or 'Ollama: model_name'.")

    use_temperature = request.temperature is not None and model_supports_temperature(request.model_name)

    if backend == "openai":
        client = config.openai_client

        if client is None:
            # Only show UI errors in the main thread
            if threading.current_thread() is threading.main_thread():
                st.error("No OpenAI client found. Check you've provided a valid OpenAI API key.")
            return LLMResponse(content="Error: No OpenAI client available")
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]     
        # Special handling flags for GPT-5 models (reasoning/verbosity)
        # Note: The Chat Completions API does not currently accept 'reasoning'/'verbosity'.
        # We therefore omit them here to avoid errors; schema-constrained JSON still works.
        is_gpt5 = model.startswith("gpt-5")
        extra_kwargs = {}

        try:
            if request.format:           
                # Manually define the schema for each structured output type
                if request.format.__name__ == "ThemeList":
                    schema = {
                        "type": "object",
                        "properties": {
                            "root": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "examples": {
                                            "type": "array", 
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["name", "description", "examples"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["root"],
                        "additionalProperties": False
                    }
                elif request.format.__name__ == "Theme":
                    schema = {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["name", "description", "examples"],
                        "additionalProperties": False
                    }
                elif request.format.__name__ == "ThemeEditResults":
                    schema = {
                        "type": "object",
                        "properties": {
                            "root": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "examples": {
                                            "type": "array", 
                                            "items": {"type": "string"}
                                        },
                                        "mark_for_deletion": {"type": "boolean"},
                                        "deletion_reason": {"type": ["string", "null"]}  # Changed to allow null
                                    },
                                    "required": ["name", "description", "examples", "mark_for_deletion", "deletion_reason"],  # Added deletion_reason
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["root"],
                        "additionalProperties": False
                    }
                elif request.format.__name__ == "ClassificationResults":
                    schema = {
                        "type": "object",
                        "properties": {
                            "root": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "themes": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "justifications": {
                                            "type": "object",
                                            "additionalProperties": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        }
                                    },
                                    "required": ["index", "themes"], # Justifications are optional; only index and themes are required.
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["root"],
                        "additionalProperties": False
                    }
                elif request.format.__name__ == "ThemeEvaluations":
                     schema = {
                        "type": "object",
                        "properties": {
                            "root": {
                                "type": "object",
                                "properties": {},  # No fixed properties
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer"},
                                        "reason": {"type": "string"}
                                    },
                                    "required": ["score", "reason"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["root"],
                        "additionalProperties": False
                    }
                elif request.format.__name__ == "ExampleRatings":
                    schema = {
                        "type": "object",
                        "properties": {
                            "root": {
                                "type": "object",
                                "properties": {},  # No fixed properties
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer"},
                                        "reason": {"type": "string"}
                                    },
                                    "required": ["score", "reason"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["root"],
                        "additionalProperties": False
                    }
                elif request.format.__name__ == "DuplicateVerification":
                    schema = {
                        "type": "object",
                        "properties": {
                            "is_duplicate": {"type": "integer"}
                        },
                        "required": ["is_duplicate"],
                        "additionalProperties": False
                    }

                # Get completion with the json schema
                if use_temperature:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=request.temperature,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": getattr(request.format, '__name__', 'output_schema').lower(),
                                "schema": schema,
                                "strict": True
                            }
                        },
                        **extra_kwargs                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": getattr(request.format, '__name__', 'output_schema').lower(),
                                "schema": schema,
                                "strict": True
                            }
                        },
                        **extra_kwargs
                    )
                
                content = response.choices[0].message.content

                # Capture usage data
                usage_data = {}
                if hasattr(response, "usage"):
                    usage = response.usage
                    usage_data["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                    usage_data["completion_tokens"] = getattr(usage, "completion_tokens", 0)
                    usage_data["total_tokens"] = getattr(usage, "total_tokens", 0)
                    
                    # Extract reasoning tokens if available
                    if hasattr(usage, "completion_tokens_details"):
                        details = usage.completion_tokens_details
                        if hasattr(details, "reasoning_tokens"):
                            usage_data["reasoning_tokens"] = details.reasoning_tokens

                if request.format.__name__ in ["ThemeList", "ClassificationResults", "Theme"]:
                    try:
                        parsed_content = json.loads(content)
                        # If we got an array directly but need a {"root": [...]} object
                        if isinstance(parsed_content, list) and request.format.__name__ != "Theme":
                            content = json.dumps({"root": parsed_content})
                    except:
                        # If parsing fails, just use the original content
                        pass

                return LLMResponse(content=content, usage=usage_data)
            else:
                # Regular completion without structured output
                if use_temperature:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=request.temperature,
                        **extra_kwargs
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **extra_kwargs
                    )
                content = response.choices[0].message.content

                # Capture usage data
                usage_data = {}
                if hasattr(response, "usage"):
                    usage = response.usage
                    usage_data["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                    usage_data["completion_tokens"] = getattr(usage, "completion_tokens", 0)
                    usage_data["total_tokens"] = getattr(usage, "total_tokens", 0)
                    
                    # Extract reasoning tokens if available
                    if hasattr(usage, "completion_tokens_details"):
                        details = usage.completion_tokens_details
                        if hasattr(details, "reasoning_tokens"):
                            usage_data["reasoning_tokens"] = details.reasoning_tokens

                return LLMResponse(content=content, usage=usage_data)

        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
            return LLMResponse(content=f"Error: {str(e)}")
    elif backend == "ollama":
        chat_model = ChatOllama(model=model, temperature=request.temperature)
        system_parts = []
        user_parts = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "user":
                user_parts.append(msg.content)
        system_prompt = "\n".join(system_parts).strip()
        user_prompt = "\n".join(user_parts).strip()
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            result = chat_model.invoke(full_prompt)
            
            if request.format:
                if request.format.__name__ == "ThemeList":
                    # Use the theme generation validator
                    validated_data = validate_llm_theme_generation_output(result.content, 1, 1)
                    if validated_data:
                        result.content = json.dumps({"root": validated_data})
                    else:
                        # If validation fails, try to extract JSON as fallback
                        json_str = extract_last_json_from_text(result.content)
                        if json_str:
                            result.content = json.dumps({"root": json.loads(json_str)})
                
                elif request.format.__name__ == "ClassificationResults":
                    # Use the classification validator
                    validated_data = validate_llm_classification_output(result.content, 1, 1)
                    if validated_data:
                        result.content = json.dumps({"root": validated_data})
                    else:
                        # If validation fails, try to extract JSON as fallback
                        json_str = extract_last_json_from_text(result.content)
                        if json_str:
                            result.content = json.dumps({"root": json.loads(json_str)})
                
                elif request.format.__name__ == "Theme":
                    # For individual themes, extract and validate single JSON object
                    json_str = re.search(r'({.*?})', result.content, re.DOTALL)
                    if json_str:
                        try:
                            theme_data = json.loads(json_str.group(1))
                            if "name" in theme_data and "description" in theme_data:
                                result.content = json_str.group(1)
                        except:
                            pass
            
            return LLMResponse(content=result.content, usage={})
        except Exception as e:
            st.error(f"Error calling Ollama model: {str(e)}")
            return LLMResponse(content=f"Error: {str(e)}")

def extract_last_json_from_text(text):
    """
    Extracts the LAST occurrence of a JSON array from a given text.
    """
    matches = re.findall(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)  # Find all JSON arrays

    return matches[-1].strip() if matches else ""  # Return the last match, or empty if none

def validate_llm_theme_generation_output(raw_output, batch_num, total_batches):
    """
    Validates and extracts JSON content from LLM theme generation output.
    """
    json_text = extract_last_json_from_text(raw_output)

    if not json_text:
        st.error(f"Could not extract valid JSON from the LLM output for batch {batch_num} of {total_batches}.")
        with st.expander(f"See raw output for batch {batch_num} of {total_batches}:"):
            st.write(raw_output)
        return False
    
    try:
        data = json.loads(json_text)

        if not isinstance(data, list):
            st.error(f"Extracted JSON must be a list of themes for batch {batch_num} of {total_batches}.")
            return False

        for item in data:
            if not isinstance(item, dict):
                st.error(f"Each theme must be a dictionary for batch {batch_num} of {total_batches}.")
                return False
            
            required_keys = {"name", "description", "examples"} 
            if not required_keys.issubset(item.keys()):
                st.error(f"Missing required fields in theme for batch {batch_num} of {total_batches}: {item}")
                return False
            
            # Validate name and description separately
            if not isinstance(item["name"], str) or not item["name"].strip():
                st.error(f"Name must be a non-empty string for batch {batch_num} of {total_batches}: {item}")
                return False
                
            if not isinstance(item["description"], str) or not item["description"].strip():
                st.error(f"Description must be a non-empty string for batch {batch_num} of {total_batches}: {item}")
                return False
            
            # Validate examples separately
            if not isinstance(item["examples"], list):
                st.error(f"Examples must be a list for batch {batch_num} of {total_batches}: {item}")
                return False
            
        return data  # Return the extracted and validated JSON
    except json.JSONDecodeError:
        st.error(f"Extracted content is not valid JSON for batch {batch_num} of {total_batches}.")
        return False

def validate_llm_classification_output(raw_output, batch_num, total_batches):
    """
    Validates and extracts JSON content from LLM classification output.
    """
    json_text = extract_last_json_from_text(raw_output)

    if not json_text:
        st.error(f"Could not extract valid JSON from the LLM output for batch {batch_num} of {total_batches}.")
        with st.expander(f"See raw output for batch {batch_num} of {total_batches}:"):
            st.write(raw_output)
        return False
    
    try:
        data = json.loads(json_text)

        if not isinstance(data, list):
            st.error(f"Extracted JSON must be a list of responses for batch {batch_num} of {total_batches}.")
            return False

        for item in data:
            if not isinstance(item, dict):
                st.error(f"Each response must be a dictionary for batch {batch_num} of {total_batches}.")
                return False
            
            required_keys = {"index", "themes"} 
            if not required_keys.issubset(item.keys()):
                st.error(f"Missing required fields for response in batch {batch_num} of {total_batches}: {item}")
                return False
            
            if not isinstance(item["index"], int) or not isinstance(item["themes"], list) or not all(isinstance(theme, str) for theme in item["themes"]):
                st.error(f"Fields in incorrect format for batch {batch_num} of {total_batches}: {item}")
                return False
            
            themes_set = {t.lower() for t in item.get("themes", [])}
            justifications = item.get("justifications", {})
            
            # Check for theme names in justifications that aren't in themes array
            for theme_key in justifications.keys():
                if theme_key.lower() not in themes_set:
                    st.warning(f"Justification provided for '{theme_key}' but not listed in themes for response {item.get('index')} (batch {batch_num})")
        return data  # Return the extracted and validated JSON

    except json.JSONDecodeError:
        st.error(f"Extracted content is not valid JSON for batch {batch_num} of {total_batches}.")
        return False

def validate_llm_theme_edit_output(raw_output: str, expected_count: int) -> Optional[List[Dict]]:
    """
    Validates and extracts JSON content from LLM theme edit output.
    """
    # Try to find JSON in the response
    json_text = extract_last_json_from_text(raw_output)
    
    if not json_text:
        # Try to parse the entire response as JSON
        try:
            data = json.loads(raw_output)
            if isinstance(data, dict) and "root" in data:
                data = data["root"]
            if isinstance(data, list):
                json_text = json.dumps(data)
            else:
                return None
        except:
            # Last resort: try to find JSON-like structure
            match = re.search(r'\[\s*\{.*?\}\s*\]', raw_output, re.DOTALL)
            if match:
                json_text = match.group(0)
            else:
                return None
    
    try:
        data = json.loads(json_text)
        
        if not isinstance(data, list):
            return None
            
        if len(data) != expected_count:
            return None
            
        # Validate and fix each theme
        for item in data:
            if not isinstance(item, dict):
                return None
            
            # Ensure required fields exist
            if "name" not in item or "description" not in item:
                return None
                
            # Add default values for optional fields
            if "examples" not in item:
                item["examples"] = []
            if "mark_for_deletion" not in item:
                item["mark_for_deletion"] = False
            if "deletion_reason" not in item:
                item["deletion_reason"] = None
                    
        return data
        
    except json.JSONDecodeError:
        return None

## token counting
import tiktoken

TOKEN_USAGE_WARNING_THRESHOLD = 50  # Percentage that triggers yellow warning
TOKEN_USAGE_DANGER_THRESHOLD = 80   # Percentage that triggers red warning

def strip_provider_prefix(model_name):
    """
    Extracts the base model name without provider prefix.
    """
    if "OpenAI: " in model_name:
        return model_name.split("OpenAI: ")[1]
    elif "Ollama: " in model_name:
        return model_name.split("Ollama: ")[1]
    return model_name

def get_model_context_limit(model_name):
    """
    Returns the context window limit (maximum tokens) for the specified model.
    """
    base_model = strip_provider_prefix(model_name)
    
    # Define context limits for known models
    # Note: Extend this dict when you add/remove models to the application
    context_limits = {
        # OpenAI models
        "o3": 200000,
        "gpt-4.1-2025-04-14": 1047576,
        "gpt-5.2": 400000,
        "gpt-5-mini": 400000,
        
        # Ollama models - note these can vary by specific checkpoint/build
        "qwen3:4b": 128000,
        "gemma3:4b": 128000,
    }
    
    # Return the limit if found, otherwise return a default
    return context_limits.get(base_model, 8192)  # Default to 8K for unknown models

def count_tokens(text, model_name):
    """
    Counts the number of tokens in the provided text for the specified model.
    """
    try:
        base_model = strip_provider_prefix(model_name)
        
        # Try to get the specific encoding for this model first
        try:
            encoding = tiktoken.encoding_for_model(base_model)
        except KeyError:
            # Fall back to cl100k_base (used by recent OpenAI models)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Count tokens
        return len(encoding.encode(text))
    except Exception as e:
        # If there's any error, return an estimate
        approx_tokens = len(text) // 4  # Rough estimate: ~4 chars per token
        return approx_tokens

def calculate_token_usage(system_msg, user_msg, model_name, usage_type="reasoning"):
    """
    Calculates token usage for messages and updates session state.
    
    Parameters:
        system_msg: System message content
        user_msg: User message content
        model_name: Model name for token counting
        usage_type: Either "reasoning" or "classification" (default "reasoning")
        
    Returns:
        Dict containing token count, limit, and percentage
    """
    complete_prompt = system_msg + "\n" + user_msg
    token_count = count_tokens(complete_prompt, model_name)
    context_limit = get_model_context_limit(model_name)
    percentage = (token_count / context_limit) * 100 if context_limit > 0 else 0
    
    # Update session state
    token_usage = {
        "count": token_count,
        "limit": context_limit,
        "percentage": percentage
    }
    
    if threading.current_thread() is threading.main_thread():
            key = f"current_{usage_type}_token_usage"
            st.session_state[key] = token_usage
    return token_usage

def show_token_usage_status(usage, title=None, description="Tokens", show_progress_bar=True, reasoning_tokens=None, container=None):
    """
    Displays token usage information in the UI with a customizable format.
    
    Parameters:
        usage: Dictionary containing count, limit, and percentage data
        title: Main title for the usage display
        description: Description of what tokens are being displayed
        show_progress_bar: Whether to display a progress bar
        reasoning_tokens: Optional count of reasoning tokens to display
        container: Optional Streamlit container to render within (defaults to st)
    """
    # Skip UI updates when called from worker threads
    if not is_main_thread():
        return
    
    # Use provided container or default to st
    display = container if container is not None else st
    
    # Extract usage information
    count = usage.get("count", 0)
    limit = usage.get("limit", 0)
    percentage = usage.get("percentage", 0)
    
    # Determine color based on token usage percentage
    if percentage > TOKEN_USAGE_DANGER_THRESHOLD:
        color = "red"
    elif percentage > TOKEN_USAGE_WARNING_THRESHOLD:
        color = "orange" 
    else:
        color = "green"
    
    # Prepare token count display text
    token_text = f"{count:,} / {limit:,} tokens ({percentage:.0f}%)"
    if reasoning_tokens is not None and reasoning_tokens > 0:
        token_text += f", including {reasoning_tokens:,} reasoning tokens"
    
    # Build HTML in parts to avoid nested template issues
    html = '<div style="padding: 5px; border-radius: 4px; margin-bottom: 5px; font-size: 0.85rem;">'
    if title is not None:
        html += f'<div style="font-weight: 600; margin-bottom: 2px;">{title}</div>'
    html += f'<div style="margin-bottom: 4px;">{description}: {token_text}</div>'
    
    # Add progress bar if requested
    if show_progress_bar:
        progress_width = min(percentage, 100)
        html += '<div style="background: #e0e0e0; width: 100%; height: 4px; border-radius: 2px;">'
        html += f'<div style="width: {progress_width}%; height: 4px; border-radius: 2px; background-color: {color};"></div>'
        html += '</div>'
    
    html += '</div>'
    
    # Display the HTML
    display.markdown(html, unsafe_allow_html=True)

###############################################################################
# 6. REASONING STEP (THEME GENERATION)
###############################################################################
# ui
def show_reasoning_model_selection():
    # count valid rows
    total_available_responses = 0
    df = None
    question_col = st.session_state["data"]["active_question"]
    if question_col and st.session_state["data"]["survey"] is not None:
        df = st.session_state["data"]["survey"]
        shuffled_indices = st.session_state.get("shuffled_indices", [])
        
        # Filter to non-missing and non-whitespace-only responses for this question
        valid_mask = df[question_col].fillna("").astype(str).str.strip() != ""
        valid_indices = [idx for idx in shuffled_indices 
                         if idx < len(df) and bool(valid_mask.iloc[idx])]
        
        total_available_responses = len(valid_indices)
        st.markdown(f"Total available responses: {total_available_responses}")
    
    # Get available models
    all_models = get_all_models()
    
    if st.session_state.get("reasoning_model_selected") not in all_models:
        st.session_state["reasoning_model_selected"] = all_models[0]

    selected_reasoning_model = st.selectbox(
        "LLM Model for reasoning:",
        all_models,
        key="reasoning_model_selected",
        help="Model used for theme generation (Step 1). Reasoning-capable models tend to produce better themes.",
    )
    render_gpt5_ui_if_applicable(selected_reasoning_model, key_prefix="reasoning_")

    if selected_reasoning_model.startswith("Ollama: "):
        ollama_model = selected_reasoning_model.split("Ollama: ")[1]
        st.warning(f"Warning: To use the selected Ollama model, run locally with: `ollama run {ollama_model}`")

    if selected_reasoning_model not in MODELS_WITHOUT_TEMPERATURE:
        st.slider(
            "Temperature (creativity):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("reasoning_temperature"),
            step=0.1,
            help="Higher values make output more random, lower values more deterministic.",
            key="reasoning_temperature"
        )
    
    batching_approach = st.radio(
        "Choose how to sample responses:",
        options=["Fixed sample size (randomised sampling)", "Group by variable"],
        index=0,
        key="batching_approach",
        help="Fixed sampling draws random batches from all responses. Group-by processes responses within each category of a chosen variable separately, which can surface group-specific themes."
    )

    if batching_approach == "Group by variable" and df is not None:
        # Get categorical columns for grouping
        categorical_columns = []
        for col in df.columns:
            if col != question_col and col != st.session_state["data"]["unique_id_column"]:
                # Only include columns with reasonable number of categories
                if len(df[col].dropna().unique()) < 30:
                    categorical_columns.append(col)
        
        if categorical_columns:
            st.session_state["grouping_variable"] = st.selectbox(
                "Group responses by:", 
                options=categorical_columns,
                index=0,
                key="grouping_variable_selectbox",
                help="Responses will be batched within each value of this variable. Only columns with fewer than 30 unique values are shown."
            )
            
            # Show preview of grouping
            if st.session_state["grouping_variable"]:
                group_var = st.session_state["grouping_variable"]
                # Calculate counts only among rows with non-missing, non-whitespace responses to the question
                valid_mask = df[question_col].fillna("").astype(str).str.strip() != ""
                group_counts = df.loc[valid_mask, group_var].value_counts().sort_values(ascending=False)
                
                st.write("Preview of groups:")
                for value, count in group_counts.items():
                    st.write(f"• {value}: {count} responses")
        else:
            st.warning("No suitable categorical variables found for grouping.")
            st.session_state["grouping_variable"] = None

    current_sample_size = st.session_state.get("sample_size", total_available_responses)
    if total_available_responses > 0 and current_sample_size > total_available_responses:
        current_sample_size = total_available_responses

    if batching_approach == "Fixed sample size (randomised sampling)":
        sample_help_text = "Number of responses to sample for each batch of analysis"
    else:
        sample_help_text = "Number of responses to sample for each batch within each group"

    st.number_input(
        "Sample size per batch",
        min_value=3,
        max_value=total_available_responses,
        value=current_sample_size,
        help=sample_help_text,
        key="sample_size")

    if batching_approach == "Fixed sample size (randomised sampling)":
        max_batches_help_text = "Maximum number of batches to process when sampling all responses"
    else:
        max_batches_help_text = "Maximum number of batches to process within each group"
    
    # Show the max number of batches to process
    st.number_input(
        "Maximum number of batches to process",
        min_value=1,
        max_value=1000,
        value=st.session_state.get("max_reasoning_batches", 10),
        help="Limits how many batches are processed. Set lower to do a quick exploratory pass, or higher to cover more of the dataset.",
        key="max_reasoning_batches"
    )

    # Check if sample size has changed
    previous_sample_size = st.session_state.get("previous_sample_size", 10)
    if st.session_state["sample_size"] != previous_sample_size:
        st.session_state["previous_sample_size"] = st.session_state["sample_size"]
        # Update sample based on new size
        get_initial_theme_generation_sample()

    st.radio(
        "Choose how to process batches:",
        options=["Serial", "Parallel"],
        index=0,
        key="reasoning_processing_mode",
        help="Serial: each batch builds on themes from the previous batch, producing more coherent results. Parallel: batches run independently then themes are merged. Faster, but consolidation is recommended afterwards."
    )

    st.text_area(
        "Additional instructions for the reasoning model (e.g. specify what kinds of themes you are looking for or the number of themes - optional):",
        help="Provide any extra instructions for the model to consider during theme generation.",
        key="additional_theme_generation_instructions",
    )
    
    # Calculate token usage
    if st.session_state.get("initial_theme_generation_sample"):
        # Get prompt components
        config = ReasoningConfig(st.session_state)
        system_msg, user_msg = build_generate_themes_prompt(st.session_state.get("initial_theme_generation_sample", {}).get("texts", []), config=ReasoningConfig(st.session_state))
        
        # Calculate token usage
        estimated_token_usage = calculate_token_usage(system_msg, user_msg, config.model_name, usage_type="reasoning")
        
        # Display token usage status
        show_token_usage_status(
            estimated_token_usage,
            description="Input token usage"
        )

    st.toggle(
        "Evaluate and improve theme examples",
        value=st.session_state.get("enable_evaluate_and_improve_examples", True),
        help="When enabled, the model will assess the quality of examples for each theme and improve them if needed. (Note this can be LLM intensive.)",
        key="enable_evaluate_and_improve_examples"
    )
        
    st.toggle(
        "Auto-consolidate themes after sampling",
        value=st.session_state.get("enable_consolidate_themes", False),
        help="When enabled, the model will try to merge similar themes after sampling all responses. (Only when more than 1 batch of responses.)",
        key="enable_consolidate_themes"
    )

def handle_theme_naming_and_acceptance():
    # Get the active theme set
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(st.session_state["data"]["active_question"])
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(st.session_state["data"]["active_question"], {})
    active_theme_set = theme_sets.get(active_theme_set_id, {})
    
    # Default to the current name (or Temporary if just created)
    current_name = active_theme_set.get("name")
    
    # Ask for the theme set name # TO DO: prevent duplicate names here
    theme_set_name = st.text_input(
        "Name for this theme set:",
        value="" if current_name is None else current_name,
        key="theme_set_name_input",
        help="Give a descriptive name to this set of themes (e.g., 'Initial Analysis', 'Product Features', etc.)",
    )

    if st.button("Accept These Themes", help="Locks in this theme set so you can proceed to classification. You can still edit themes in the sidebar after accepting."):
        if active_theme_set_id:
            theme_set_name = st.session_state.theme_set_name_input
            final_name = theme_set_name.strip() if theme_set_name else "Default Theme Set"
            theme_sets[active_theme_set_id]["name"] = final_name
            theme_sets[active_theme_set_id]["accepted"] = True  # Set accepted on the theme set
        st.success("Themes accepted. Proceed to classification below.")
        st.rerun()

# prompt building functions
def get_theme_description():
    """
    Returns a standardised description of what makes a good theme and how to format each theme.
    
    Returns:
      A string with the theme guidelines and format.
    """
    return """
    For each theme:
    - Give it a short **name** (2-4 words max).
    - Write 1 sentence characterising the theme.
    - Provide up to 3 example quotes from the sample that illustrate the theme.

    What makes a good theme:
    A good theme simplifies the data.
    For example, its description should convey the common essence of a set of responses, not just list multiple similar examples.
    A good theme is single-minded.
    Though it may have multiple aspects or nuances, it should express a single unifying idea.
    All themes should have internal consistency.
    I.e. all examples of the theme should share something essential to the theme.
    And all themes should be clearly distinct from each other.
    For example, 'price sensitivity' and 'value for money' are too similar to be separate themes.

    What makes a good example quote:
    A good set of example quotes together reflect the difference aspects of the theme found in the data, helping communicate it's nuances.
    Each example should be an exact quote from the sample. The quote can be a subset of a response or a full response.
    A good example quote only contains the parts of the response relevant to the theme.
    Return only the selected quote, with no quotation marks.
    """

def get_theme_json_output_example():
    """
    Returns the expected JSON output format for theme generation.
    
    Returns:
      A string with the expected JSON output format.
    """
    return """
    Your final output should be a JSON array of objects, following the below example format. Examples must be exact verbatim quotes, not paraphrased, and without quotation marks.

    [
        {
            "name": "<name>",
            "description": "<Short description of the theme>",
            "examples": [
                "<Example quote 1 representing the theme>",
                "<Example quote 2 representing the theme>",
                "<Example quote 3 representing the theme>"
                ]
        },
        ...
    ]
    """

def build_generate_themes_prompt(sample_texts: List[str],
                                 config: ReasoningConfig,
                                 include_sample_texts: bool = True
                                 ) -> tuple[str, str]:
    """
    Builds a complete prompt for theme generation with system/user messages.
    
    Parameters:
      sample_texts: List of sample texts for this specific batch.
      config: ReasoningConfig object containing settings like additional_instructions.
      include_sample_texts: Whether to include sample texts in the output user message.
    
    Returns:
      A tuple containing:
        - system_msg_content: Complete system message for LLM
        - user_msg_content: Complete user message for LLM
    """

    sample_size = len(sample_texts)
    additional_theme_generation_instructions = config.additional_instructions
    
    # Build the prompt text
    prompt_text = f"""
    Below is a sample of {len(sample_texts)} verbatim responses from consumer research. 
    Please propose a set of distinct themes that reflect the breadth of responses. 
    You should think about each response individually, making sure that all relevant themes for that response are captured.
    {get_theme_description()}
    """
    if additional_theme_generation_instructions:
        prompt_text += f"\n\nAdditional Instructions:\n{additional_theme_generation_instructions}\nTake care to ensure that theme descriptions reflect these additional instructions.\n"

    prompt_text += f"""
    {get_theme_json_output_example()}
    """
    
    # Format sample texts if requested
    formatted_samples = ""
    if include_sample_texts and sample_texts:
        formatted_samples = "\n".join(f"{i+1}) {txt}" for i, txt in enumerate(sample_texts))
        user_msg_content = prompt_text + "\n\nResponses:\n" + formatted_samples
    else:
        user_msg_content = prompt_text
    
    system_msg_content = "You are a highly skilled qualitative market researcher, trained in the nuanced identification of themes from verbatim responses and the sensitive adherence to additional instructions provided by your client. You must return valid JSON only."
    
    return system_msg_content, user_msg_content

def build_successive_themes_prompt(existing_themes: List[Dict],
                                   sample_texts: List[str],
                                   config: ReasoningConfig
                                   ) -> tuple[str, str]:
    """
    Builds a prompt to suggest additional themes based on already identified themes.
    
    Parameters:
      existing_themes: The list of themes identified so far.
      sample_texts: A list of sample survey responses.
      config: ReasoningConfig object containing settings like additional_instructions.
      
    Returns:
      A tuple containing:
        - system_msg_content: Complete system message for LLM
        - user_msg_content: Complete user message for LLM
    """
    additional_theme_generation_instructions = config.additional_instructions

    # Build the prompt text
    prompt_text = f"""
    Below is a sample of {len(sample_texts)} verbatim responses from consumer research and {len(existing_themes)} themes identified from earlier responses.
    Please review the verbatim responses and propose any new themes that are needed to reflect the breadth and nuance of these responses.
    You should think about each response individually, making sure that all relevant themes for that response are captured.
    {get_theme_description()}
    """
    if additional_theme_generation_instructions:
        prompt_text += f"\n\nAdditional Instructions:\n{additional_theme_generation_instructions}\nTake care to ensure that theme descriptions reflect these additional instructions.\n"

    prompt_text += f"""
    \nExisting Themes:
    {json.dumps(existing_themes, indent=2)}

    {get_theme_json_output_example()}
    """

    formatted_samples = "\n".join(f"{i+1}) {txt}" for i, txt in enumerate(sample_texts))
    user_msg_content = prompt_text + "\n\nResponses:\n" + formatted_samples

    system_msg_content = """You are a highly skilled qualitative market researcher, 
    trained in the nuanced identification of themes from verbatim responses 
    and the sensitive adherence to additional instructions provided by your client. 
    You have been tasked with reviewing new responses against existing themes, then identifying any additional themes
    or subtle tweaks to existing themes needed to capture the breadth and nuance of the responses.
    You must return valid JSON only."""

    return system_msg_content, user_msg_content

def get_initial_theme_generation_sample(question_col=None):
    """
    Prepares and stores sample texts for theme generation based on current UI settings.
    Uses pre-shuffled indices to select a consistent random sample.
    """
    # If no question specified, use active question
    if question_col is None:
        question_col = st.session_state["data"]["active_question"]
        if not question_col:
            return
    
    df = st.session_state["data"]["survey"]
    sample_size = st.session_state.get("sample_size", 10)
    
    # Get pre-shuffled indices
    shuffled_indices = st.session_state.get("shuffled_indices", [])
    
    # Filter to non-missing and non-whitespace-only responses for this question
    non_whitespace_mask = df[question_col].fillna("").astype(str).str.strip() != ""
    valid_indices = [idx for idx in shuffled_indices 
                     if idx < len(df) and bool(non_whitespace_mask.iloc[idx])]
    
    if not valid_indices:
        st.session_state["initial_theme_generation_sample"] = {
            "indices": [],
            "texts": []
        }
        return
    
    # Take the first batch according to current sample size setting
    sample_indices = valid_indices[:min(sample_size, len(valid_indices))]
    sample_texts = df.loc[sample_indices, question_col].tolist()
    
    # Store in session state for later use
    st.session_state["initial_theme_generation_sample"] = {
        "indices": sample_indices,
        "texts": sample_texts,
        "available_indices": valid_indices
    }

def build_theme_consolidation_prompt(current_themes, additional_instructions):
    """
    Builds a prompt for automatically consolidating themes after sampling all responses.
    
    Parameters:
      current_themes: The list of themes to consolidate.
      additional_instructions: Any additional instructions from the theme generation step.
      
    Returns:
      A string prompt for the LLM.
    """
    # Build the prompt text with different guidance based on whether additional instructions exist
    prompt_text = f"""
    Below is a list of {len(current_themes)} themes that were generated from analyzing all survey responses.
    """
    
    if additional_instructions:
        prompt_text += f"""
        Please check that these meet the brief given by the below additional instructions and make any necessary adjustments.
        If the additional instructions specify a target number of themes, this may require merging themes to reach that number.
        
        ADDITIONAL INSTRUCTIONS:
        {additional_instructions}
        """
    else:
        prompt_text += """
        Please do a final check of the themes and make any final edits ensure they are distinct and coherent.
        You may need to merge or split themes to achieve this.
        """
    
    prompt_text += f"""
    {get_theme_description()}
    
    \nExisting Themes:
    {json.dumps(current_themes, indent=2)}

    {get_theme_json_output_example()}
    """
    
    return prompt_text

def build_get_more_themes_prompt(existing_themes, sample_texts, additional_instructions):
    """
    Builds a prompt to get additional themes based on already identified themes.
    
    Parameters:
      existing_themes: The list of themes identified so far.
      sample_texts: A list of sample survey responses.
      
    Returns:
      A string prompt for the LLM.
    """
    # additional_instructions = st.session_state.get("additional_theme_generation_instructions", "")

    # Build the prompt text
    prompt_text = f"""
    Below is a sample of {len(sample_texts)} open-ended survey responses and {len(existing_themes)} existing themes.
    Please review the survey responses and look for additional themes. These may be variations on existing themes.
    {get_theme_description()}
    """
    if additional_instructions:
        prompt_text += f"\n\nAdditional Instructions:\n{additional_instructions}\nTake care to ensure that theme descriptions reflect the additional instructions.\n"

    prompt_text += f"""
    \nExisting Themes:
    {json.dumps(existing_themes, indent=2)}

    {get_theme_json_output_example()}

    \nResponses:
    """
    
    return prompt_text

def build_theme_splits_merges_prompt(current_themes, instructions):
    """
    Builds a prompt for suggesting splits or merges of existing themes based purely on 
    theme descriptions and examples, without referencing new survey responses.
    
    Parameters:
      current_themes: The list of themes to analyze for potential splits/merges.
      instructions: User-provided instructions for splitting or merging.
      
    Returns:
      A string prompt for the LLM.
    """
    # Build the prompt text
    prompt_text = f"""
    Below is a list of {len(current_themes)} existing themes with their descriptions and examples.
    Please analyze these themes and suggest a new complete set of themes that addresses the following instructions:
    
    {instructions}
    
    When suggesting splits or merges:
    1. For splits: Identify themes that may contain multiple distinct concepts or nuances and split them into more specific themes.
    2. For merges: Identify themes that overlap conceptually and combine them into more coherent themes.
    3. Maintain the overall essence of the themes but refine the structure based on the instructions.
    
    {get_theme_description()}
    
    \nExisting Themes:
    {json.dumps(current_themes, indent=2)}

    {get_theme_json_output_example()}
    """
    
    return prompt_text

def build_theme_edit_prompt(themes_batch: List[Dict],
                           instructions: str,
                           include_validated_examples: bool) -> tuple[str, str]:
    """
    Builds a prompt for editing theme descriptions and examples.
    
    Parameters:
        themes_batch: List of themes to edit in this batch
        instructions: User-provided editing instructions
        include_validated_examples: If True, examples must remain unchanged
        
    Returns:
        Tuple of (system_message, user_message)
    """
    
    prompt_text = f"""
    You have been given {len(themes_batch)} themes to edit according to specific instructions.
    
    Instructions: {instructions}
    
    For each theme, you should:
    1. Update theme names, descriptions {"according to the instructions" if include_validated_examples else "and examples according to the instructions"}
    2. If a theme should be removed based on the instructions, set "mark_for_deletion" to true and provide a brief reason
    
    Important:
    - You must return EXACTLY {len(themes_batch)} theme objects (including any marked for deletion)
    - Each theme in your response must correspond to a theme in the input, in the same order
    - Even if marking a theme for deletion, you must still include it in the response
    - ONLY make changes specified in the instructions; do not change theme names if not asked to
    
    Current themes to edit (note: count and percentage fields are for your reference only and should not be included in your response):
    {json.dumps(themes_batch, indent=2)}
    
    Return a JSON array with exactly {len(themes_batch)} objects following this format:
    [
        {{
            "name": "<edited name>",
            "description": "<edited description>",
            "examples": ["<example 1>", "<example 2>", "<example 3>"],
            "mark_for_deletion": false,
            "deletion_reason": null
        }},
        ...
    ]
    """
    
    system_msg = """You are a highly skilled qualitative market researcher, trained in refining and improving theme definitions.
    You must return exactly the same number of themes as provided, in the same order.
    You must return valid JSON only."""
    
    return system_msg, prompt_text

# llm calls
def run_reasoning_llm(
        system_message_content: str,
        user_message_content: str,
        config: ReasoningConfig
        ) -> tuple[Optional[List[Dict]], Dict]:
    """
    Calls the reasoning LLM with a combined prompt and sample responses, then validates the JSON output.
    
    Parameters:
      system_message_content: The complete system message string.
      user_message_content: The complete user message string (inc. instructions & samples).
      config: Configuration object containing LLM settings
      
    Returns:
      A tuple containing:
        - A list of Theme dictionaries if the output is valid; otherwise, None.
        - A dictionary containing token usage data or error info.
    """
    
    system_msg = LLMMessage(role="system", content=system_message_content)
    user_msg = LLMMessage(role="user", content=user_message_content)

    model_name = config.model_name

    # Calculate token usage before making the API call
    calculate_token_usage(system_msg.content, user_msg.content, model_name, usage_type="reasoning")

    # Create the LLM call using Pydantic for input validation and include the structured output schema.
    llm_request = LLMCall(
        messages=[system_msg, user_msg],
        model_name=model_name,
        temperature=config.get_effective_temperature(),
        format=ThemeList
    )
    llm_response = call_llm(llm_request, config)

    llm_call_usage = llm_response.usage if hasattr(llm_response, "usage") else {}

    # Validate and parse the structured output using Pydantic.
    try:
        # Parse the JSON string in the content field
        response_data = json.loads(llm_response.content)

        # Extract themes from the response
        if isinstance(response_data, dict) and "root" in response_data:
            # If we get the expected structure
            return response_data["root"], llm_call_usage
        else:
            if threading.current_thread() is threading.main_thread():
                st.error(f"Unexpected response format: {response_data}")
                with st.expander("Raw LLM output:"):
                    st.write(llm_response.content)
            return None, {"error": "Unexpected response format", **llm_call_usage}
          
    except Exception as e:
        if threading.current_thread() is threading.main_thread():
            st.error(f"Structured output validation error: {e}")
            with st.expander("Raw LLM output:"):
                st.write(llm_response.content)
        return None, {"error": str(e), **llm_call_usage}
    
def prepare_theme_generation_batches(df, question_col, available_rows=None):
    """
    Prepare batches for theme generation based on current settings.
    
    Parameters:
        df: DataFrame containing survey responses
        question_col: Column name containing responses
        available_rows: Optional list of row indices to use (if None, uses all valid rows)
    
    Returns:
        tuple: (all_batches, grouped_indices, processing_mode, batching_approach)
            - all_batches: List of batch indices for fixed sample size approach
            - grouped_indices: Dict of group_value -> indices for group by approach
            - processing_mode: "Serial" or "Parallel"
            - batching_approach: "Fixed sample size (randomised sampling)" or "Group by variable"
    """
    processing_mode = st.session_state.get("reasoning_processing_mode", "Serial")
    batching_approach = st.session_state.get("batching_approach", "Fixed sample size (randomised sampling)")
    sample_size = st.session_state.get("sample_size", 10)
    
    all_batches = []
    grouped_indices = {}
    
    if batching_approach == "Group by variable":
        grouping_variable = st.session_state.get("grouping_variable")
        if grouping_variable not in df.columns:
            st.error(f"Grouping variable {grouping_variable} not found in data")
            return [], {}, processing_mode, batching_approach
            
        # Group responses by the selected variable
        if available_rows is None:
            non_whitespace_mask = df[question_col].fillna("").astype(str).str.strip() != ""
            valid_indices = df.index[non_whitespace_mask].tolist()
        else:
            valid_indices = available_rows
            
        for idx in valid_indices:
            group_value = df.loc[idx, grouping_variable]
            if pd.notna(group_value):
                grouped_indices.setdefault(group_value, []).append(idx)
    else:
        # Fixed sample size approach
        if available_rows is None:
            # Use initial sample if available
            sample_data = st.session_state.get("initial_theme_generation_sample", {})
            all_indices = sample_data.get("available_indices", [])
        else:
            all_indices = available_rows
            
        max_batches = st.session_state.get("max_reasoning_batches", 10)
        
        # Create batches
        total_batches = min((len(all_indices) + sample_size - 1) // sample_size, max_batches)
        
        for batch_num in range(total_batches):
            start_idx = batch_num * sample_size
            end_idx = min(start_idx + sample_size, len(all_indices))
            batch_indices = all_indices[start_idx:end_idx]
            all_batches.append(batch_indices)
    
    return all_batches, grouped_indices, processing_mode, batching_approach


def generate_themes():
    """
    Generates themes from sample responses.
    Uses structured output validation with Pydantic and updates the session state.
    """
    # Get from state
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]

    config = ReasoningConfig(st.session_state) # model_name, temperature, additional_instructions
    
    sample_size = st.session_state.get("sample_size", 10)

    all_themes = []
    existing_theme_names = set() # use a set for efficient look ups
    st.session_state["backup_themes"] = []

    all_batches, grouped_indices, processing_mode, batching_approach = prepare_theme_generation_batches(
            df, question_col
        )

    if batching_approach == "Group by variable":
        max_batches_per_group = st.session_state.get("max_reasoning_batches", 10)
        
        # Process each group
        if processing_mode == "Serial":
            # Serial processing - Process each batch within group in series
            if len(grouped_indices) > 1:
                # Multiple groups - can process groups in parallel but serially within each group
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(grouped_indices))) as executor:
                    futures = []
                    
                    for group_value, indices in grouped_indices.items():
                        # Create batches for this group
                        group_batches = []
                        total_batches = min((len(indices) + sample_size - 1) // sample_size, max_batches_per_group)
                        
                        for batch_num in range(total_batches):
                            start_idx = batch_num * sample_size
                            end_idx = min(start_idx + sample_size, len(indices))
                            batch_indices = indices[start_idx:end_idx]
                            group_batches.append(batch_indices)
                        
                        # Submit this group for processing
                        future = executor.submit(
                            process_serial_batches,
                            df, question_col, group_batches, config
                        )
                        futures.append((group_value, future))
                    
                    # Collect themes from all groups
                    for group_value, future in futures:
                        group_themes = future.result()
                        for theme in group_themes:
                            if theme["name"].lower() not in existing_theme_names:
                                theme["group"] = group_value  # Tag with source group
                                all_themes.append(theme)
                                existing_theme_names.add(theme["name"].lower())
                                st.write(f"New theme from group '{group_value}': {theme['name']}")
            else:
                # Only one group - process it directly
                for group_value, indices in grouped_indices.items():
                    # Create batches for this group
                    group_batches = []
                    total_batches = min((len(indices) + sample_size - 1) // sample_size, max_batches_per_group)
                    
                    for batch_num in range(total_batches):
                        start_idx = batch_num * sample_size
                        end_idx = min(start_idx + sample_size, len(indices))
                        batch_indices = indices[start_idx:end_idx]
                        group_batches.append(batch_indices)
                    
                    # Process this group serially
                    group_themes = process_serial_batches(
                        df, question_col, group_batches, config
                    )
                    
                    # Add themes from this group
                    for theme in group_themes:
                        if theme["name"].lower() not in existing_theme_names:
                            theme["group"] = group_value  # Tag with source group
                            all_themes.append(theme)
                            existing_theme_names.add(theme["name"].lower())
        else:
            # Parallel processing - Process each group's batches in parallel, tag with group
            for group_value, indices in grouped_indices.items():
                group_batches = []
                total_batches = min((len(indices) + sample_size - 1) // sample_size, max_batches_per_group)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * sample_size
                    end_idx = min(start_idx + sample_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    group_batches.append(batch_indices)
                
                group_themes = process_parallel_batches(
                    df, question_col, group_batches, config
                )
                for theme in group_themes:
                    if theme["name"].lower() not in existing_theme_names:
                        theme["group"] = group_value  # Tag with source group
                        all_themes.append(theme)
                        existing_theme_names.add(theme["name"].lower())
    else:
        # Fixed sample size approach
        sample_data = st.session_state.get("initial_theme_generation_sample", {}) # so we can all available indices
        all_indices = sample_data.get("available_indices", [])
        max_batches = st.session_state.get("max_reasoning_batches", 10)
        
        # Create batches
        all_batches = []
        total_batches = min((len(all_indices) + sample_size - 1) // sample_size, max_batches)
        
        for batch_num in range(total_batches):
            start_idx = batch_num * sample_size
            end_idx = min(start_idx + sample_size, len(all_indices))
            batch_indices = all_indices[start_idx:end_idx]
            all_batches.append(batch_indices)

        # Process batches according to selected mode
        if processing_mode == "Serial":
            all_themes = process_serial_batches(
                df, question_col, all_batches, config
            )
        else:
            all_themes = process_parallel_batches(
                df, question_col, all_batches, config
            )

    # Update session state with the themes
    st.session_state["reasoning"]["current_themes"] = all_themes
    st.session_state["backup_themes"] = all_themes.copy()
    
    # Update the active theme set
    question = st.session_state["data"]["active_question"]
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if active_theme_set_id:
        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
        if active_theme_set_id in theme_sets:
            theme_sets[active_theme_set_id]["themes"] = all_themes.copy()
            # Store grouping metadata on the theme set
            if batching_approach == "Group by variable":
                grouping_variable = st.session_state.get("grouping_variable")
                theme_sets[active_theme_set_id]["grouping_variable"] = grouping_variable
                theme_sets[active_theme_set_id]["valid_groups"] = list(grouped_indices.keys())
            else:
                theme_sets[active_theme_set_id]["grouping_variable"] = None
                theme_sets[active_theme_set_id]["valid_groups"] = []
    
    # Post-processing steps
    total_batches = len(all_batches) if all_batches else sum(
        min((len(indices) + sample_size - 1) // sample_size, st.session_state.get("max_reasoning_batches", 10))
        for indices in grouped_indices.values()
    )

    if st.session_state.get("enable_evaluate_and_improve_examples", True) and total_batches > 1 and len(all_themes) > 0:
        with st.spinner('Evaluating and improving theme examples...'):
            themes_improved = evaluate_and_improve_theme_examples(all_themes)
            if themes_improved > 0:
                st.success(f"Improved examples for {themes_improved} themes.")
    
    if st.session_state.get("enable_consolidate_themes", False) and total_batches > 1 and len(all_themes) > 5:
        consolidate_themes_after_sampling()

def consolidate_themes_after_sampling():
    """
    Automatically consolidates themes after sampling all responses.
    Makes a call to the reasoning LLM to merge similar themes and reduce redundancy.
    Consolidation is done within each group separately to preserve group boundaries.
    """
    # Get the current themes
    current_themes = st.session_state["reasoning"]["current_themes"]
    df = st.session_state["data"]["survey"]
    config = ReasoningConfig(st.session_state)
    model_name=config.model_name
    
    if not current_themes or len(current_themes) <= 5:  # Skip if we have very few themes
        return
    
    # Get the active question and theme set to fetch additional instructions
    question = st.session_state["data"]["active_question"]
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    additional_instructions = ""
    
    if active_theme_set_id:
        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
        if active_theme_set_id in theme_sets:
            additional_instructions = theme_sets[active_theme_set_id].get("additional_instructions", "")

    # Partition themes by group for within-group consolidation
    has_any_groups = any(t.get("group") is not None for t in current_themes)
    if has_any_groups:
        groups_map = {}  # group_value -> [themes]
        for theme in current_themes:
            g = theme.get("group")  # None means "No group"
            groups_map.setdefault(g, []).append(theme)
        
        all_consolidated = []
        original_theme_count = len(current_themes)
        
        for group_value, group_themes in groups_map.items():
            group_label = group_value if group_value is not None else "No group"
            if len(group_themes) <= 3:
                # Too few to consolidate, keep as-is
                all_consolidated.extend(group_themes)
                continue
            
            with st.spinner(f'Consolidating themes for group "{group_label}"...'):
                consolidated = _consolidate_theme_list_via_llm(
                    group_themes, additional_instructions, config, model_name, df, question
                )
                # Preserve group tag on consolidated themes
                for t in consolidated:
                    t["group"] = group_value
                all_consolidated.extend(consolidated)
        
        # Replace current themes with consolidated themes
        st.session_state["reasoning"]["current_themes"] = all_consolidated
        st.success(f"Consolidated {original_theme_count} themes into {len(all_consolidated)} themes.")
    else:
        # No groups - consolidate all together (original behaviour)
        with st.spinner('Consolidating themes after sampling all responses...'):
            consolidated = _consolidate_theme_list_via_llm(
                current_themes, additional_instructions, config, model_name, df, question
            )
        original_theme_count = len(current_themes)
        st.session_state["reasoning"]["current_themes"] = consolidated
        st.success(f"Consolidated {original_theme_count} themes into {len(consolidated)} themes.")
    
    # Update the active theme set
    consolidated_themes = st.session_state["reasoning"]["current_themes"]
    if active_theme_set_id:
        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
        if active_theme_set_id in theme_sets:
            theme_sets[active_theme_set_id]["themes"] = consolidated_themes.copy()

def _consolidate_theme_list_via_llm(themes_to_consolidate, additional_instructions, config, model_name, df, question):
    """
    Core LLM consolidation logic. Takes a list of themes and returns a consolidated list.
    Does not modify session state directly.
    """
    # Build the prompt
    prompt_text = build_theme_consolidation_prompt(themes_to_consolidate, additional_instructions)
    
    # Build LLM messages
    system_msg = LLMMessage(
        role="system",
        content="You are a highly skilled qualitative market researcher, trained in the nuanced identification of themes from verbatim responses and the sensitive adherence to additional instructions provided by your client. You have been tasked with consolidating an existing set of themes to remove any duplication. You must return valid JSON only."
    )
    user_msg = LLMMessage(
        role="user",
        content=prompt_text
    )
    
    llm_request = LLMCall(
        messages=[system_msg, user_msg],
        model_name=model_name,
        temperature=config.get_effective_temperature(), 
        format=ThemeList
    )
    llm_response = call_llm(llm_request, config)

    reasoning_tokens = llm_response.usage.get("reasoning_tokens", 0)
    total_tokens = llm_response.usage.get("total_tokens", 0)
    
    context_limit = get_model_context_limit(model_name)
    batch_usage = {
        "count": total_tokens,
        "limit": context_limit,
        "percentage": (total_tokens / context_limit * 100) 
                    if context_limit > 0 else 0
    }

    if hasattr(llm_response, "usage") and llm_response.usage:
        show_token_usage_status(
            batch_usage,
            description="Theme consolidation token usage",
            reasoning_tokens=llm_response.usage.get("reasoning_tokens", None)
        )    
    
    # Parse the LLM response
    try:
        response_data = json.loads(llm_response.content)
        if "root" in response_data:
            new_themes = response_data["root"]
        else:
            st.error("Unexpected response format from LLM.")
            return themes_to_consolidate  # Return originals on failure
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        return themes_to_consolidate  # Return originals on failure
    
    # Validate and update each theme with fresh examples from sample texts
    validated_themes = []
    for theme in new_themes:
        validated_theme = validate_theme_example(
                theme,
                df,
                question,
                config
            )
        if "examples" in validated_theme:
            theme["examples"] = validated_theme["examples"]
        validated_themes.append(theme)
    
    return validated_themes

def validate_theme_example(theme: dict, df, question_col, config, max_retries=5, sample_size=30) -> dict:
    """
    Checks whether the current theme's examples exist in the provided sample responses.
    If not, it calls the reasoning LLM (retrying up to max_retries times) to provide new examples.
    
    Parameters:
      theme: A dictionary with keys "name", "description", and "examples".
      
    Returns:
      A dictionary with validated examples.
    """

    # Extract examples from the theme
    if "examples" in theme and isinstance(theme["examples"], list):
        examples = theme["examples"]
    else:
        examples = []

    if isinstance(examples, str):
        examples = [examples]

    examples = [ex for ex in examples if ex]

    if len(examples) >= 3:
        return {"examples": examples[:3]}

    # Access responses data
    valid_examples = examples.copy()

    # Try to find additional examples to reach 3 if needed
    remaining_needed = 3 - len(valid_examples)
    for attempt in range(max_retries):
        # Sample responses
        # Use only non-missing, non-whitespace-only responses
        available_rows = df.index[df[question_col].fillna("").astype(str).str.strip() != ""].tolist()
        if not available_rows:
            break
            
        # Avoid previously used examples by checking the text
        current_sample_indices = random.sample(available_rows, min(sample_size, len(available_rows)))
        current_sample_texts = df.loc[current_sample_indices, question_col].tolist()
        
        # Find new examples using the current sample
        for _ in range(remaining_needed):
            new_example = find_example_for_theme(theme, current_sample_texts, valid_examples, config)
            if new_example and new_example not in valid_examples:
                valid_examples.append(new_example)
                remaining_needed -= 1
        
        # If we found enough examples, stop trying
        if remaining_needed <= 0:
            break
    
    return {"examples": valid_examples}

def find_example_for_theme(theme, sample_responses, existing_examples, config: ReasoningConfig, max_retries=2):
    """
    Finds a single new example quote that illustrates a given theme.
    
    Calls the LLM to identify one response from the sample that best represents
    the theme, avoiding examples that are already used.
    
    Parameters:
        theme (dict): Theme dictionary containing name and description
        sample_responses (list): List of survey responses to search for examples
        existing_examples (list): Examples already associated with this theme
        config: Configuration object containing LLM settings
        max_retries (int): Maximum number of attempts to get a valid example
    
    Returns:
        str: A quote that illustrates the theme, or empty string if none found
    """
    # Filter out responses that are already used as examples
    available_responses = [r for r in sample_responses if r not in existing_examples]
    if not available_responses:
        return ""  # No more available responses
    prompt = f"""
    The theme details are as follows:
    Name: {theme.get('name', '').strip()}
    Description: {theme.get('description', '').strip()}
    
    Below are a set of sample survey responses:
    {chr(10).join(available_responses)}
    
    Please think hard to select one quote that illustrates this theme well. 
    The quote can be a subset of a response or a full response.
    Return only the selected quote, with no quotation marks.
    """
    
    system_msg = LLMMessage(
        role="system",
        content="You are a highly skilled qualitative market researcher. Extract one illustrative quote from the provided verbatim responses that best reflects the given theme name and description."
    )
    user_msg = LLMMessage(
        role="user",
        content=prompt.strip()
    )
    
    # Try up to max_retries times to get a valid example
    for attempt in range(max_retries):
        llm_request = LLMCall(
            messages=[system_msg, user_msg],
            model_name=config.model_name,
            temperature=config.get_effective_temperature(), 
            format=None
        )
        llm_response = call_llm(llm_request, config)
        new_example = llm_response.content.strip()
        clean_example = new_example.strip('"\'') # remove any quotation marks
        
        # Validate that the new example appears in one of the sample responses.
        if clean_example and any(clean_example in response for response in available_responses):
            return clean_example
    
    # If we can't find a new example, just return an empty string
    return ""

GOOD_EXAMPLE_THRESHOLD = 7
def improve_theme_examples(theme, max_attempts=5):
    """
    Improves the quality of examples for a theme by finding better illustrative quotes.
    
    Samples random responses from the dataset, evaluates potential examples for quality,
    and replaces existing examples with higher-quality alternatives when found.
    
    Parameters:
        theme (dict): Theme dictionary containing name, description, and examples
        max_attempts (int): Maximum number of sampling attempts to make
        
    Returns:
        bool: True if the theme's examples were improved, False otherwise
    """
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    
    # Start with existing examples (if any)
    candidate_examples = theme.get("examples", []).copy()
    good_examples = [] 
    
    for attempt in range(max_attempts):
        # Stop if we already have 3 good examples
        if len(good_examples) >= 3:
            break
            
        # Sample fresh responses for this attempt
        sample_size = st.session_state.get("sample_size", 30)
        available_rows = df.index[df[question_col].fillna("").astype(str).str.strip() != ""].tolist()
        if not available_rows:
            break
            
        sample_indices = random.sample(available_rows, min(sample_size, len(available_rows)))
        sample_texts = df.loc[sample_indices, question_col].tolist()
        
        # Find new candidate examples (aim for 6 to score)
        new_candidates = []
        for _ in range(6): 
            existing = candidate_examples + new_candidates + good_examples
            config = ReasoningConfig(st.session_state)
            new_example = find_example_for_theme(theme, sample_texts, existing, config)
            if new_example and new_example not in existing:
                new_candidates.append(new_example)
        
        if not new_candidates:
            continue  # No new candidates, try another sample
        
        # Add new candidates to our full candidate list
        candidate_examples.extend(new_candidates)
        
        # Evaluate candidate examples for quality
        if candidate_examples:
            eval_examples_prompt = f"""
            Theme Name: {theme.get('name', '').strip()}
            Theme Description: {theme.get('description', '').strip()}
            
            Below are candidate examples. Rate each example from 1-10 based on how well it illustrates the theme.
            A score of {GOOD_EXAMPLE_THRESHOLD} or higher means the example sufficiently demonstrates the theme.
            
            {chr(10).join([f"{i+1}. {ex}" for i, ex in enumerate(candidate_examples)])}
            
            Return ratings as a JSON object where keys are example indices (0-based) and values are scores:
            Example: {{"0": 8, "1": 5, "2": 9}}
            """
            
            system_msg = LLMMessage(
                role="system",
                content="You are a highly skilled qualitative market researcher. You have been tasked with rating how well different verbatim quotes illustrate a set of themes."
            )
            user_msg = LLMMessage(
                role="user",
                content=eval_examples_prompt
            )
            
            config = ReasoningConfig(st.session_state)
            llm_request = LLMCall(
                messages=[system_msg, user_msg],
                model_name=st.session_state.get("reasoning_model_selected"),
                temperature=config.get_effective_temperature(), 
                format=ExampleRatings
            )
            llm_response = call_llm(llm_request, config)
            
            # Parse the rating response
            try:
                ratings = ExampleRatings.model_validate_json(llm_response.content).root
                
                # Create a dict to store scores for sorting later
                example_scores = {}
                
                # Add high scoring examples to our good list
                for idx_str, rating in ratings.items():
                    try:
                        idx = int(idx_str)
                        if idx < len(candidate_examples) and rating.score >= 7:
                            example = candidate_examples[idx]
                            # Store the score for this example
                            example_scores[example] = rating.score
                            # Make sure we don't add duplicates
                            if example not in good_examples:
                                good_examples.append(example)
                    except (ValueError, TypeError):
                        continue
                
                # Sort good examples by score (highest first)
                good_examples.sort(key=lambda ex: example_scores.get(ex, 0), reverse=True)
                
            except Exception as e:
                st.warning(f"Error parsing example ratings: {e}")
        
        # Clear candidates that didn't score well enough 
        candidate_examples = []
    
    # Update theme with good examples (up to 3)
    if good_examples:
        theme["examples"] = good_examples[:3]
        return True
    elif len(candidate_examples) > 0:
        # If we couldn't find enough good examples, use the candidates we have
        theme["examples"] = candidate_examples[:3]
        return True
    
    return False

def evaluate_and_improve_theme_examples(current_themes):
    """
    Evaluates the quality of theme examples and improves them in a single efficient pass.
    
    Parameters:
        current_themes: List of theme dictionaries
    
    Returns:
        Number of themes that were improved
    """
    if not current_themes:
        return 0

    # Create a consolidated evaluation prompt for all themes
    eval_prompt = "Evaluate how well the examples represent each theme description. Score each from 1-10:\n\n"
    
    # Add each theme and its examples to the prompt
    for i, theme in enumerate(current_themes):
        theme_name = theme.get("name", f"Theme {i+1}")
        description = theme.get("description", "")
        
        # Extract examples from wherever they might be stored
        examples = []
        if "examples" in theme and isinstance(theme["examples"], list):
            examples = theme["examples"]
        elif "example" in theme and isinstance(theme["example"], dict) and "examples" in theme["example"]:
            examples = theme["example"]["examples"]
            
        eval_prompt += f"Theme {i+1}: {theme_name}\n"
        eval_prompt += f"Description: {description}\n"
        eval_prompt += "Examples:\n"
        
        if examples:
            for j, example in enumerate(examples):
                eval_prompt += f"  {j+1}) {example}\n"
        else:
            eval_prompt += "  (No examples provided)\n"
            
        eval_prompt += "\n"
    
    eval_prompt += "\nProvide scores in JSON format with theme indices and brief reasons:"
    eval_prompt += '\n{"0": {"score": 7, "reason": "..."}, "1": {"score": 4, "reason": "..."}}'
    
    # Call LLM to evaluate example quality
    system_msg = LLMMessage(
        role="system",
        content="You are a highly skilled qualitative market researcher. You have been tasked with rating how well different verbatim quotes illustrate a set of themes."
    )
    user_msg = LLMMessage(
        role="user",
        content=eval_prompt
    )
    
    model_name=st.session_state.get("reasoning_model_selected")

    llm_request = LLMCall(
        messages=[system_msg, user_msg],
        model_name=model_name,
        temperature=st.session_state.get("reasoning_temperature", 0.8) if model_name not in MODELS_WITHOUT_TEMPERATURE else None, 
        format=ThemeEvaluations
    )
    config = ReasoningConfig(st.session_state)
    llm_response = call_llm(llm_request, config)
    
    reasoning_tokens = llm_response.usage.get("reasoning_tokens", 0)
    total_tokens = llm_response.usage.get("total_tokens", 0)
    
    context_limit = get_model_context_limit(model_name)
    batch_usage = {
        "count": total_tokens,
        "limit": context_limit,
        "percentage": (total_tokens / context_limit * 100) 
                    if context_limit > 0 else 0
    }

    if hasattr(llm_response, "usage") and llm_response.usage:
        show_token_usage_status(
            batch_usage,
            description="Example evaluation token usage",
            reasoning_tokens=llm_response.usage.get("reasoning_tokens", None)
        )

    # Parse evaluation results
    theme_scores = {}
    try:
        evaluations = ThemeEvaluations.model_validate_json(llm_response.content)
        theme_scores = evaluations.root
    except Exception as e:
        st.warning(f"Error parsing theme evaluation: {e}")
        return 0
    
    # Find themes with low scores that need improvement
    themes_to_improve = []
    for idx_str, eval_data in theme_scores.items():
        try:
            idx = int(idx_str)
            score = eval_data.score
            if score < GOOD_EXAMPLE_THRESHOLD and idx < len(current_themes):
                themes_to_improve.append(idx)
        except (ValueError, TypeError):
            continue
    
    # Update status with number of themes needing improvement
    if themes_to_improve:
        st.info(f"Found {len(themes_to_improve)} themes that need better examples.")
    else:
        st.success("All themes have good examples.")
        return 0

    # For themes that need improvement, sample directly from the dataset
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    
    themes_improved = 0
    progress_bar = st.progress(0)
    for i, theme_idx in enumerate(themes_to_improve):
        theme = current_themes[theme_idx]
        theme_name = theme.get("name", f"Theme {theme_idx+1}")
        
        # Update progress
        progress = (i) / max(len(themes_to_improve), 1)
        progress_bar.progress(progress)
        st.info(f"Working on theme {i+1}/{len(themes_to_improve)}: '{theme_name}'")
        
        if improve_theme_examples(theme):
            themes_improved += 1
            st.success(f"Successfully improved examples for '{theme_name}'")
        else:
            st.warning(f"Could not find better examples for '{theme_name}'")
            
        # Update progress again
        progress = (i+1) / max(len(themes_to_improve), 1)
        progress_bar.progress(progress)
    
    # Clear progress bar after completion
    progress_bar.empty()
    return themes_improved

def get_more_themes(additional_instructions):
    """
    Generates additional themes for the active question based on current themes.
    
    Samples a set of responses from the survey data and asks the LLM to identify
    new themes that weren't covered by the existing themes. Only themes with unique
    names (case-insensitive) are added to avoid duplication.
    
    This function is intended to be used after initial theme generation when users
    want to explore additional thematic concepts in the data without regenerating
    all themes from scratch.
    
    Parameters:
        additional_instructions (str): Extra guidance for the LLM on what kind of 
                                      additional themes to look for
    
    Returns:
        None: Results are stored in session state and success/info messages are 
              displayed in the Streamlit interface
    """

    # Get the survey data and active question
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    
    # Use only non-missing, non-whitespace-only responses
    available_rows = df.index[df[question_col].fillna("").astype(str).str.strip() != ""].tolist()
    if not available_rows:
        st.error("No survey responses available for sampling.")
        return

    # Get the current themes (existing themes)
    current_themes = st.session_state["reasoning"]["current_themes"]

    # Include the additional instructions for get_more_themes
    config = ReasoningConfig(st.session_state)
    # Override the additional instructions with the get_more_themes specific ones
    config.additional_instructions = additional_instructions
    all_batches, grouped_indices, processing_mode, batching_approach = prepare_theme_generation_batches(
        df, question_col, available_rows
    )

    new_themes = []

    if batching_approach == "Group by variable":
        max_batches_per_group = st.session_state.get("max_reasoning_batches", 10)
        
        if processing_mode == "Serial":
            # Process each group's batches serially
            existing_theme_names = {theme["name"].lower() for theme in current_themes if "name" in theme}
            
            for group_value, indices in grouped_indices.items():
                # Create batches for this group
                sample_size = st.session_state.get("sample_size", 10)
                group_batches = []
                total_batches = min((len(indices) + sample_size - 1) // sample_size, max_batches_per_group)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * sample_size
                    end_idx = min(start_idx + sample_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    group_batches.append(batch_indices)
                
                # Process this group's batches with existing themes as starting point
                group_themes = process_serial_batches_with_existing_themes(
                    df, question_col, group_batches, config, current_themes
                )
                
                # Add only new themes
                for theme in group_themes:
                    if theme["name"].lower() not in existing_theme_names:
                        theme["group"] = group_value  # Tag with source group
                        new_themes.append(theme)
                        existing_theme_names.add(theme["name"].lower())
                        st.write(f"New theme from group '{group_value}': {theme['name']}")
        else:
            # Parallel processing - process each group's batches in parallel, tag with group
            sample_size = st.session_state.get("sample_size", 10)
            existing_theme_names = {theme["name"].lower() for theme in current_themes if "name" in theme}
            
            for group_value, indices in grouped_indices.items():
                group_batches = []
                total_batches = min((len(indices) + sample_size - 1) // sample_size, max_batches_per_group)
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * sample_size
                    end_idx = min(start_idx + sample_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    group_batches.append(batch_indices)
                
                group_found_themes = process_parallel_batches_with_existing_themes(
                    df, question_col, group_batches, config, current_themes
                )
                
                # Filter to only new themes and tag with group
                for theme in group_found_themes:
                    if theme["name"].lower() not in existing_theme_names:
                        theme["group"] = group_value  # Tag with source group
                        new_themes.append(theme)
                        existing_theme_names.add(theme["name"].lower())
    else:
        # Fixed sample size approach
        if processing_mode == "Serial":
            new_themes = process_serial_batches_with_existing_themes(
                df, question_col, all_batches, config, current_themes
            )
        else:
            new_themes = process_parallel_batches_with_existing_themes(
                df, question_col, all_batches, config, current_themes
            )

    # Calculate total batches processed
    if all_batches:
        total_batches = len(all_batches)
    else:
        # For group by variable approach
        sample_size = st.session_state.get("sample_size", 10)
        max_batches_per_group = st.session_state.get("max_reasoning_batches", 10)
        total_batches = sum(
            min((len(indices) + sample_size - 1) // sample_size, max_batches_per_group)
            for indices in grouped_indices.values()
        )
    
    # Auto-consolidate new themes if enabled and we have enough
    if (st.session_state.get("enable_consolidate_themes", False) and 
        total_batches > 1 and 
        len(new_themes) > 5):
        
        with st.spinner('Consolidating new themes...'):
            consolidated_new_themes = consolidate_themes_list(new_themes, additional_instructions)
            
            if consolidated_new_themes and len(consolidated_new_themes) < len(new_themes):
                st.info(f"Consolidated {len(new_themes)} new themes into {len(consolidated_new_themes)} themes.")
                new_themes = consolidated_new_themes

    # Add only genuinely new themes to the current themes
    current_theme_names = {theme["name"].lower() for theme in current_themes if "name" in theme}
    added = 0
    for theme in new_themes:
        if "name" in theme and theme["name"].lower() not in current_theme_names:
            current_themes.append(theme)
            current_theme_names.add(theme["name"].lower())
            added += 1

    st.session_state["reasoning"]["current_themes"] = current_themes
    
    if added > 0:
        st.success(f"Added {added} new theme(s).")
    else:
        st.info("No additional themes were generated.")

def process_serial_batches_with_existing_themes(df, question_col, batches, config, existing_themes):
    """
    Process batches serially, starting with existing themes.
    This ensures each batch builds upon existing themes + any new themes found so far.
    """
    all_themes = existing_themes.copy()  # Start with existing themes
    existing_theme_names = {theme["name"].lower() for theme in existing_themes if "name" in theme}
    
    total_batches = len(batches)
    
    for batch_num, batch_indices in enumerate(batches):
        sample_texts = df.loc[batch_indices, question_col].tolist()
        
        with st.spinner(f'Working on batch {batch_num+1} of {total_batches} for additional themes...'):
            # Process batch with all accumulated themes
            themes, usage_data = process_theme_batch(
                sample_texts, 
                all_themes,  # Pass all accumulated themes
                config,
                df,
                question_col
            )
            
            # Display token usage information
            if usage_data:
                token_count = usage_data.get("total_tokens", 0)
                reasoning_tokens = usage_data.get("reasoning_tokens", 0)
                context_limit = get_model_context_limit(config.model_name)
                
                batch_usage = {
                    "count": token_count,
                    "limit": context_limit,
                    "percentage": (token_count / context_limit * 100) if context_limit > 0 else 0
                }
                
                show_token_usage_status(
                    batch_usage, 
                    description=f"Batch {batch_num+1} token usage",
                    reasoning_tokens=reasoning_tokens
                )
            
            # Add only new themes
            if themes:
                for theme in themes:
                    if theme["name"].lower() not in existing_theme_names:
                        all_themes.append(theme)
                        existing_theme_names.add(theme["name"].lower())
                        st.write(f"New theme: {theme['name']}")
    
    # Return all themes (existing + newly found)
    return all_themes

def consolidate_themes_list(themes_to_consolidate, additional_instructions=""):
    """
    Consolidates a provided list of themes (e.g., just newly found themes).
    Similar to consolidate_themes_after_sampling but works on a specific list.
    Consolidation is done within each group separately to preserve group boundaries.
    
    Parameters:
        themes_to_consolidate: List of themes to consolidate
        additional_instructions: Any additional instructions to consider
        
    Returns:
        List of consolidated themes
    """
    if not themes_to_consolidate or len(themes_to_consolidate) <= 5:
        return themes_to_consolidate
    
    config = ReasoningConfig(st.session_state)
    model_name = config.model_name
    
    # Check if any themes have group assignments
    has_any_groups = any(t.get("group") is not None for t in themes_to_consolidate)
    
    if has_any_groups:
        # Partition by group, consolidate each partition separately
        groups_map = {}
        for theme in themes_to_consolidate:
            g = theme.get("group")
            groups_map.setdefault(g, []).append(theme)
        
        all_consolidated = []
        for group_value, group_themes in groups_map.items():
            if len(group_themes) <= 3:
                all_consolidated.extend(group_themes)
                continue
            consolidated = _consolidate_themes_list_raw(group_themes, additional_instructions, config, model_name)
            # Preserve group tag on consolidated themes
            for t in consolidated:
                t["group"] = group_value
            all_consolidated.extend(consolidated)
        return all_consolidated
    else:
        # No groups - consolidate all together (original behaviour)
        return _consolidate_themes_list_raw(themes_to_consolidate, additional_instructions, config, model_name)

def _consolidate_themes_list_raw(themes_to_consolidate, additional_instructions, config, model_name):
    """
    Raw LLM consolidation for a flat list of themes (no group awareness).
    Returns the consolidated theme list.
    """
    # Build the consolidation prompt for just these themes
    prompt_text = build_theme_consolidation_prompt(themes_to_consolidate, additional_instructions)
    
    # Build LLM messages
    system_msg = LLMMessage(
        role="system",
        content="You are a highly skilled qualitative market researcher, trained in the nuanced identification of themes from verbatim responses and the sensitive adherence to additional instructions provided by your client. You have been tasked with consolidating an existing set of themes to remove any duplication. You must return valid JSON only."
    )
    user_msg = LLMMessage(
        role="user",
        content=prompt_text
    )
    
    # Make the LLM call
    llm_request = LLMCall(
        messages=[system_msg, user_msg],
        model_name=model_name,
        temperature=st.session_state.get("reasoning_temperature", 0.8) if model_name not in MODELS_WITHOUT_TEMPERATURE else None,
        format=ThemeList
    )
    
    llm_response = call_llm(llm_request, config)
    
    # Parse response
    try:
        response_data = json.loads(llm_response.content)
        if "root" in response_data:
            consolidated_themes = response_data["root"]
            
            # Validate themes have proper structure
            validated_themes = []
            for theme in consolidated_themes:
                if "examples" not in theme:
                    if "example" in theme and isinstance(theme["example"], str):
                        theme["examples"] = [theme["example"]]
                    else:
                        theme["examples"] = []
                validated_themes.append(theme)
            
            return validated_themes
        else:
            st.error("Unexpected response format from LLM during consolidation.")
            return themes_to_consolidate
    except Exception as e:
        st.error(f"Error parsing consolidation response: {e}")
        return themes_to_consolidate

def process_parallel_batches_with_existing_themes(df, question_col, batches, config, existing_themes):
    """
    Process batches in parallel, with each batch seeing only the original existing themes.
    """
    all_themes = []
    existing_theme_names = {theme["name"].lower() for theme in existing_themes if "name" in theme}

    # Create thread-safe data structure for token usage
    usage_lock = threading.Lock()
    usage_stats = {
        "completed_batches": 0,
        "total_tokens": 0,
        "total_reasoning_tokens": 0,
        "max_batch_tokens": 0,
        "max_batch_reasoning": 0,
        "new_themes": []
    }
    
    with st.spinner(f'Processing {len(batches)} batches for additional themes with parallel processing...'):
        progress_bar = st.progress(0)

        progress_status = st.empty()
        token_usage_container = st.empty()
        max_usage_container = st.empty()
        themes_container = st.empty()
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
            futures = []
            
            # Submit all batch processing jobs
            for i, batch_indices in enumerate(batches):
                sample_texts = df.loc[batch_indices, question_col].tolist()
                future = executor.submit(
                    process_theme_batch,
                    sample_texts,
                    existing_themes,  # All batches see the same existing themes
                    config,
                    df,
                    question_col
                )
                futures.append((i, future))
            
            # Process results as they complete
            completed = 0
            
            for future in as_completed([f[1] for f in futures]):
                batch_themes, usage_data = future.result()
                
                # Update usage stats (same as original parallel processing)
                batch_total_tokens = usage_data.get("total_tokens", 0)
                batch_reasoning_tokens = usage_data.get("reasoning_tokens", 0)
                
                with usage_lock:
                    completed += 1
                    usage_stats["completed_batches"] = completed
                    usage_stats["total_tokens"] += batch_total_tokens
                    usage_stats["total_reasoning_tokens"] += batch_reasoning_tokens
                    usage_stats["max_batch_tokens"] = max(usage_stats["max_batch_tokens"], batch_total_tokens)
                    usage_stats["max_batch_reasoning"] = max(usage_stats["max_batch_reasoning"], batch_reasoning_tokens)
                    
                    # Add only new themes
                    if batch_themes:
                        for theme in batch_themes:
                            if theme["name"].lower() not in existing_theme_names:
                                all_themes.append(theme)
                                existing_theme_names.add(theme["name"].lower())
                                usage_stats["new_themes"].append(theme["name"])
                    
                    # Update UI
                    progress_bar.progress(completed / len(batches))
                    progress_status.text(f"Completed {completed}/{len(batches)} batches")
                    
                    if usage_stats["completed_batches"] > 0:
                        avg_tokens = usage_stats["total_tokens"] / usage_stats["completed_batches"]
                        avg_reasoning = usage_stats["total_reasoning_tokens"] / usage_stats["completed_batches"]
                        token_usage_container.text(
                            f"Average tokens per batch: {avg_tokens:.0f} (reasoning: {avg_reasoning:.0f})"
                        )
                        max_usage_container.text(
                            f"Max tokens in a batch: {usage_stats['max_batch_tokens']} "
                            f"(reasoning: {usage_stats['max_batch_reasoning']})"
                        )
                    
                    if usage_stats["new_themes"]:
                        themes_container.text(f"New themes found: {', '.join(usage_stats['new_themes'][:5])}")
        
        # Clear progress indicators
        progress_bar.empty()
        progress_status.empty()
        token_usage_container.empty()
        max_usage_container.empty()
        themes_container.empty()
    
    return all_themes

def suggest_theme_splits_merges(instructions):
    """
    Suggests splits or merges for the current themes based on user instructions.
    Updates the theme set with the new suggestions based solely on existing theme data.
    
    Parameters:
      instructions: User-provided instructions for splitting or merging.
    """
    # Get the current themes
    current_themes = st.session_state["reasoning"]["current_themes"]
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    config = ReasoningConfig(st.session_state)

    model_name = config.model_name
    
    if not current_themes:
        st.error("No themes available to analyze.")
        return
    
    # Collect all examples from current themes to use for validation later
    all_examples = []
    for theme in current_themes:
        if "examples" in theme and isinstance(theme["examples"], list):
            all_examples.extend([ex for ex in theme["examples"] if ex])
    
    # Build the prompt
    prompt_text = build_theme_splits_merges_prompt(current_themes, instructions)
    
    # Build LLM messages
    system_msg = LLMMessage(
        role="system",
        content="You are a highly skilled qualitative market researcher, trained in the nuanced identification of themes from verbatim responses and the sensitive adherence to additional instructions provided by your client. You have been tasked with refining existing theme definitions by splitting and merging existing themes. You must return valid JSON only."
    )
    
    user_msg = LLMMessage(
        role="user",
        content=prompt_text
    )
    
    with st.spinner(f'Analyzing themes for splits and merges using {model_name}...'):
        llm_request = LLMCall(
            messages=[system_msg, user_msg],
            model_name=model_name,
            temperature=config.temperature if model_name not in MODELS_WITHOUT_TEMPERATURE else None, 
            format=ThemeList
        )
        llm_response = call_llm(llm_request, config)

        reasoning_tokens = llm_response.usage.get("reasoning_tokens", 0)
        total_tokens = llm_response.usage.get("total_tokens", 0)
        
        context_limit = get_model_context_limit(model_name)
        batch_usage = {
            "count": total_tokens,
            "limit": context_limit,
            "percentage": (total_tokens / context_limit * 100) 
                        if context_limit > 0 else 0
        }

        if hasattr(llm_response, "usage") and llm_response.usage:
            show_token_usage_status(
                batch_usage,
                description="Theme splits/merges token usage", 
                reasoning_tokens=llm_response.usage.get("reasoning_tokens", None)
            )
    
    # Parse the LLM response
    try:
        response_data = json.loads(llm_response.content)
        if "root" in response_data:
            new_themes = response_data["root"]
        else:
            st.error("Unexpected response format from LLM.")
            return
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        return
    
    # Build old name -> group map for group preservation
    old_group_map = {}  # lowercase name -> group
    for t in current_themes:
        old_group_map[t["name"].lower()] = t.get("group")

    # Validate and update each theme with proper examples
    validated_themes = []
    for theme in new_themes:
        # Ensure examples is properly formatted
        if "examples" not in theme:
            theme["examples"] = []
        elif len(theme["examples"]) < 3 and all_examples:
            # Try to validate examples using existing examples pool
            theme_with_examples = validate_theme_example(theme,
                                                         df,
                                                         question_col,
                                                         config
                                                         )
            if "examples" in theme_with_examples:
                theme["examples"] = theme_with_examples["examples"]
        
        # Preserve group from old themes via name matching
        theme_name_lower = theme.get("name", "").lower()
        if theme_name_lower in old_group_map:
            # Exact name match - inherit group
            theme["group"] = old_group_map[theme_name_lower]
        else:
            # No exact match (split or merged theme) - try partial matching
            matched_groups = set()
            for old_name, old_group in old_group_map.items():
                # Check if old name is contained in new name or vice versa
                if old_name in theme_name_lower or theme_name_lower in old_name:
                    matched_groups.add(old_group)
            if len(matched_groups) == 1:
                # All partial matches agree on group
                theme["group"] = matched_groups.pop()
            else:
                # Ambiguous or no match - default to None (universal)
                theme["group"] = None
        
        validated_themes.append(theme)
    
    # Replace current themes with new themes
    st.session_state["reasoning"]["current_themes"] = validated_themes
    
    # Update the active theme set
    question = st.session_state["data"]["active_question"]
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    
    if active_theme_set_id:
        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
        if active_theme_set_id in theme_sets:
            theme_sets[active_theme_set_id]["themes"] = validated_themes.copy()
    
    st.success(f"Suggested {len(validated_themes)} themes based on split/merge instructions.")

def process_theme_batch(batch_texts: List[str],
                        existing_themes: Optional[List[Dict]], # Keep existing_themes for serial mode
                        config: ReasoningConfig, # Pass config for instructions and settings
                        df: pd.DataFrame,
                        question_col: str
                        ) -> tuple[Optional[List[Dict]], Dict]:
    """
    Process a batch of texts to generate themes.
    
    Parameters:
        batch_texts: List of text responses to analyze
        existing_themes: Optional list of existing themes to build upon
        config: Configuration object containing LLM settings
        df: Survey dataframe
        question_col: Column name with responses
        
    Returns:
        Tuple of (themes, usage_data)
    """
    
    # Build appropriate prompt based on whether we have existing themes
    if existing_themes:
        system_content, user_content = build_successive_themes_prompt(
            existing_themes, batch_texts, config
        )
    else:
        system_content, user_content = build_generate_themes_prompt(
            batch_texts, config
        )
    # Call the LLM
    themes, usage_data = run_reasoning_llm(system_content, user_content, config)

    validated_themes = []
    # Process themes to ensure they have valid examples
    if themes:
        for theme in themes:
            # Ensure theme has proper example format
            if "examples" not in theme:
                if "example" in theme and isinstance(theme["example"], str):
                    theme["examples"] = [theme["example"]]
                else:
                    theme["examples"] = []
            
            # Validate example format
            theme_with_examples = validate_theme_example(
                theme,
                df,
                question_col,
                config
            )
            if "examples" in theme_with_examples:
                theme["examples"] = theme_with_examples["examples"]

            validated_themes.append(theme)
    
    return validated_themes if themes is not None else None, usage_data

def process_serial_batches(df, question_col, batches, config):
    """
    Process a list of batches in serial mode with theme dependencies.
    
    Parameters:
        df: DataFrame containing responses
        question_col: Column name with responses
        batches: List of batch indices to process
        config: Configuration object containing LLM settings
        
    Returns:
        List of themes generated from all batches
    """
    
    all_themes = []
    existing_theme_names = set()
    total_batches = len(batches)
    
    for batch_num, batch_indices in enumerate(batches):
        sample_texts = df.loc[batch_indices, question_col].tolist()
        
        # Show UI feedback only on main thread
        if is_main_thread():
            with st.spinner(f'Working on batch {batch_num+1} of {total_batches}...'):
                # Display sample texts in an expander
                with st.expander(f"Sample texts for batch {batch_num+1} of {total_batches}:"):
                    st.markdown("\n".join(f"{i+1}) {txt}" for i, txt in enumerate(sample_texts)))
        
        # For first batch, no existing themes; for later batches, use accumulated themes
        themes, usage_data = process_theme_batch(
            sample_texts, 
            all_themes if batch_num > 0 else None,
            config,
            df,
            question_col
        )
        
        # Display token usage information (show_token_usage_status has its own main thread guard)
        if usage_data:
            token_count = usage_data.get("total_tokens", 0)
            reasoning_tokens = usage_data.get("reasoning_tokens", 0)
            context_limit = get_model_context_limit(config.model_name)
            
            batch_usage = {
                "count": token_count,
                "limit": context_limit,
                "percentage": (token_count / context_limit * 100) if context_limit > 0 else 0
            }
            
            show_token_usage_status(
                batch_usage, 
                description=f"Batch {batch_num+1} token usage",
                reasoning_tokens=reasoning_tokens
            )
        
        # Add new themes, avoiding duplicates
        if themes:
            for theme in themes:
                if theme["name"].lower() not in existing_theme_names:
                    all_themes.append(theme)
                    existing_theme_names.add(theme["name"].lower())
                    if is_main_thread():
                        st.write(f"New theme: {theme['name']}")
    
    return all_themes

def process_parallel_batches(df: pd.DataFrame,
                             question_col: str,
                             batches: List[List[int]],
                             config: ReasoningConfig # CHANGED: Accept config object
                             ) -> List[Dict]:
    """
    Process a list of batches in parallel mode with no dependencies.
    
    Parameters:
        df: DataFrame containing responses
        question_col: Column name with responses
        batches: List of batch indices to process
        config: Configuration object containing LLM settings
        
    Returns:
        List of themes generated from all batches
    """
    all_themes = []
    existing_theme_names = set()

    # Create thread-safe data structure for token usage
    usage_lock = threading.Lock()
    usage_stats = {
        "completed_batches": 0,
        "total_tokens": 0,
        "total_reasoning_tokens": 0,
        "max_batch_tokens": 0,
        "max_batch_reasoning": 0,
        "new_themes": []
    }
    
    with st.spinner(f'Processing {len(batches)} batches with parallel processing...'):
        progress_bar = st.progress(0)

        progress_status = st.empty()
        token_usage_container = st.empty()
        max_usage_container = st.empty()
        themes_container = st.empty()
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
            futures = []
            
            # Submit all batch processing jobs
            for i, batch_indices in enumerate(batches):
                sample_texts = df.loc[batch_indices, question_col].tolist()
                future = executor.submit(
                    process_theme_batch,
                    sample_texts,
                    None,  # No existing themes in parallel mode
                    config,
                    df,
                    question_col
                )
                futures.append((i, future))
            
            # Process results as they complete
            completed = 0
            results = []
            
            for future in as_completed([f[1] for f in futures]):
                batch_themes, usage_data = future.result()
                
                batch_total_tokens = usage_data.get("total_tokens", 0)
                batch_reasoning_tokens = usage_data.get("reasoning_tokens", 0)
            
                with usage_lock:
                    usage_stats["completed_batches"] += 1
                    usage_stats["total_tokens"] += usage_data.get("total_tokens", 0)
                    usage_stats["total_reasoning_tokens"] += usage_data.get("reasoning_tokens", 0)

                    usage_stats["max_batch_tokens"] = max(usage_stats["max_batch_tokens"], batch_total_tokens)
                    usage_stats["max_batch_reasoning"] = max(usage_stats["max_batch_reasoning"], batch_reasoning_tokens)
                    
                    new_themes_in_batch = []
                    if batch_themes:
                        for theme in batch_themes:
                            if theme["name"].lower() not in existing_theme_names:
                                all_themes.append(theme)
                                existing_theme_names.add(theme["name"].lower())
                                usage_stats["new_themes"].append(theme["name"])
                                new_themes_in_batch.append(theme["name"])
                    
                    # Calculate progress
                    progress = usage_stats["completed_batches"] / len(futures)
                    progress_status.text(f"Progress: {usage_stats['completed_batches']}/{len(batches)} batches completed ({int(progress * 100)}%)")
                    
                    # Update token usage display
                    context_limit = get_model_context_limit(config.model_name)

                    current_batch_usage = {
                        "count": batch_total_tokens,
                        "limit": context_limit,
                        "percentage": (batch_total_tokens / context_limit * 100) if context_limit > 0 else 0
                    }
                    
                    show_token_usage_status(
                        current_batch_usage,
                        title="Latest Batch",
                        description="Token usage in most recent batch",
                        reasoning_tokens=batch_reasoning_tokens,
                        container=token_usage_container
                    )
                    
                    # Show maximum batch token usage
                    max_batch_usage = {
                        "count": usage_stats["max_batch_tokens"],
                        "limit": context_limit,
                        "percentage": (usage_stats["max_batch_tokens"] / context_limit * 100) if context_limit > 0 else 0
                    }
                    
                    show_token_usage_status(
                        max_batch_usage,
                        title="Maximum Usage",
                        description="Highest token usage in any batch",
                        reasoning_tokens=usage_stats["max_batch_reasoning"],
                        container=max_usage_container
                    )
                    
                    # Update themes display
                    if new_themes_in_batch:
                        themes_text = "**Latest Themes Found:**\n"
                        for theme in new_themes_in_batch:
                            themes_text += f"• {theme}\n"
                        themes_text += f"\n**Total Unique Themes:** {len(usage_stats['new_themes'])}"
                        themes_container.markdown(themes_text)
    
    # Final summary after all batches complete
    st.success(f"Completed parallel processing: {len(usage_stats['new_themes'])} unique themes found")
    
    return all_themes

def edit_themes_batch(themes_batch: List[Dict],
                     batch_idx: int,
                     instructions: str,
                     include_validated_examples: bool,
                     config: ReasoningConfig,
                     max_retries: int = 3) -> tuple[int, Optional[List[Dict]], List[str]]:
    """
    Process a single batch of themes for editing.
    
    Returns:
        Tuple of (batch_idx, edited_themes, errors)
    """
    errors = []
    
    last_response = None
    
    for attempt in range(max_retries):
        try:
            # Build prompt
            system_msg, user_msg = build_theme_edit_prompt(
                themes_batch,
                instructions,
                include_validated_examples
            )
            
            # Call LLM
            llm_request = LLMCall(
                messages=[
                    LLMMessage(role="system", content=system_msg),
                    LLMMessage(role="user", content=user_msg)
                ],
                model_name=config.model_name,
                temperature=config.get_effective_temperature(),
                format=ThemeEditResults
            )
            
            llm_response = call_llm(llm_request, config)
            last_response = llm_response.content
            
            # Parse response with validation
            edited_themes = validate_llm_theme_edit_output(
                llm_response.content, 
                len(themes_batch)
            )
            
            if edited_themes is None:
                raise ValueError(f"Invalid response format from LLM")
            
            # Success! Convert to regular dicts
            result_themes = []
            for theme in edited_themes:
                result_themes.append({
                    "name": theme["name"],
                    "description": theme["description"],
                    "examples": theme.get("examples", []),
                    "mark_for_deletion": theme.get("mark_for_deletion", False),
                    "deletion_reason": theme.get("deletion_reason", None)
                })
            
            return batch_idx, result_themes, []
            
        except Exception as e:
            error_msg = f"Batch {batch_idx} attempt {attempt + 1} failed: {str(e)}"
            errors.append(error_msg)
            
            # Continue to next attempt
            continue
    
    # All attempts failed
    return batch_idx, None, errors


def edit_all_themes(instructions: str, batch_size: int, include_validated_examples: bool):
    """
    Edit all current themes according to instructions, processing in batches.
    """
    current_themes = st.session_state["reasoning"]["current_themes"]
    
    if not current_themes:
        st.error("No themes to edit")
        return
    
    config = ReasoningConfig(st.session_state)
    
    # Create batches
    batches = []
    theme_counts = st.session_state["reasoning"].get("theme_counts", {})
    for i in range(0, len(current_themes), batch_size):
        batch = current_themes[i:i + batch_size]
        # Add count data to each theme in the batch
        batch_with_counts = []
        for theme in batch:
            theme_with_count = theme.copy()
            count_data = theme_counts.get(theme["name"], {"count": 0, "percentage": 0})
            theme_with_count["count"] = int(count_data["count"])
            theme_with_count["percentage"] = float(count_data["percentage"])
            batch_with_counts.append(theme_with_count)
        batches.append((i, batch_with_counts))
    
    # Process batches in parallel
    all_edited_themes = [None] * len(current_themes)
    themes_for_deletion = []
    
    with st.spinner(f'Processing {len(batches)} batches...'):
        progress_bar = st.progress(0)
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
            futures = []
            
            # Submit all batches
            for batch_idx, (start_idx, batch_themes) in enumerate(batches):
                future = executor.submit(
                    edit_themes_batch,
                    batch_themes,
                    batch_idx,
                    instructions,
                    include_validated_examples,
                    config
                )
                futures.append((start_idx, future))
            
            # Process results
            completed = 0
            
            for start_idx, future in futures:
                result = future.result()
                
                # Handle both old format (3 items) and new format (4 items)
                batch_idx, edited_batch, errors = result
                
                completed += 1
                progress_bar.progress(completed / len(futures))

                # Place edited themes in correct positions
                for i, edited_theme in enumerate(edited_batch):
                    original_idx = start_idx + i
                    # Preserve group from original theme (LLM doesn't know about groups)
                    if original_idx < len(current_themes):
                        edited_theme["group"] = current_themes[original_idx].get("group")
                    all_edited_themes[original_idx] = edited_theme
                    
                    # Track deletions
                    if edited_theme.get("mark_for_deletion", False):
                        themes_for_deletion.append({
                            "index": original_idx,
                            "theme": edited_theme,
                            "reason": edited_theme.get("deletion_reason", "No reason provided")
                        })
    
    # Check if there are themes marked for deletion
    if themes_for_deletion:
        # Store results for confirmation UI
        st.session_state["edited_themes_result"] = all_edited_themes
        st.session_state["themes_pending_deletion"] = themes_for_deletion
        st.success(f"Theme editing complete. {len(themes_for_deletion)} themes marked for deletion.")
    else:
        # No deletions - apply edits immediately
        # Clean up the edited themes (remove deletion markers)
        final_themes = []
        for theme in all_edited_themes:
            if theme:  # Check theme is not None
                # Remove deletion markers
                clean_theme = theme.copy()
                if "mark_for_deletion" in clean_theme:
                    del clean_theme["mark_for_deletion"]
                if "deletion_reason" in clean_theme:
                    del clean_theme["deletion_reason"]
                final_themes.append(clean_theme)
        
        # Update themes
        st.session_state["reasoning"]["current_themes"] = final_themes
        
        # Update active theme set
        question = st.session_state["data"]["active_question"]
        active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
        
        if active_theme_set_id:
            theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
            if active_theme_set_id in theme_sets:
                theme_sets[active_theme_set_id]["themes"] = final_themes.copy()
        
        st.success(f"Theme editing complete. All {len(final_themes)} themes updated.")

def deduplicate_themes(themes, similarity_threshold=0.85, config=None, debug=False):
    """
    Deduplicate themes that refer to the same concept despite minor
    name variations or spelling differences.
    
    Uses Union-Find data structure to efficiently track duplicate relationships.
    
    Parameters:
        themes: List of theme dictionaries
        similarity_threshold: Threshold for name similarity (0.0-1.0)
        config: Configuration for LLM (defaults to using a cheaper model)
        debug: Whether to show debug info in the UI
        
    Returns:
        tuple: (deduplicated_themes, deduplication_stats)
    """
    DUPLICATE_SCORE_THRESHOLD = 9 # Threshold for LLM to consider two themes as duplicates
    
    if not themes or len(themes) <= 1:
        return themes, {"exact_matches": 0, "approximate_matches": 0, "total_reduced": 0}
    
    stats = {
        "exact_matches": 0,
        "approximate_matches": 0,
        "total_reduced": 0
    }
    
    if debug:
        # Initialize session state for debug info if it doesn't exist
        if "dedup_debug_info" not in st.session_state:
            st.session_state["dedup_debug_info"] = {
                "exact_matches": [],
                "evaluation_results": [],
                "merged_groups": []
            }
        else:
            # Clear previous debug info for a new run
            st.session_state["dedup_debug_info"] = {
                "exact_matches": [],
                "evaluation_results": [],
                "merged_groups": []
            }
    
    # Step 1: Initialize Union-Find data structure
    # Each theme starts in its own set (parent points to itself)
    parents = list(range(len(themes)))
    
    def find(x):
        """Find the root of the set containing x (with path compression)"""
        if parents[x] != x:
            parents[x] = find(parents[x])  # Path compression
        return parents[x]
    
    def union(x, y):
        """Merge the sets containing x and y"""
        parents[find(x)] = find(y)
    
    # Step 2: Find potential duplicates based on name similarity
    potential_duplicates = []
    exact_match_count = 0
    
    for i, j in itertools.combinations(range(len(themes)), 2):
        # Skip empty names
        name_i = themes[i].get('name', '').lower().strip()
        name_j = themes[j].get('name', '').lower().strip()
        
        if not name_i or not name_j:
            continue
        
        # Only compare themes within the same group (or both ungrouped)
        group_i = themes[i].get("group")
        group_j = themes[j].get("group")
        if group_i != group_j:
            continue
        
        # Exact match - immediately union
        if name_i == name_j:
            union(i, j)
            stats["exact_matches"] += 1
            exact_match_count += 1
            if debug:
                st.session_state["dedup_debug_info"]["exact_matches"].append(
                    f"Exact match: '{themes[i]['name']}' and '{themes[j]['name']}'"
                )
            
        # Check similarity for non-exact matches
        similarity = SequenceMatcher(None, name_i, name_j).ratio()
        if similarity >= similarity_threshold:
            potential_duplicates.append((i, j))
    
    # Step 3: Process potential duplicates with LLM
    if potential_duplicates:
        # Convert indices to theme objects for verification
        theme_pairs = [(themes[i], themes[j]) for i, j in potential_duplicates]
        
        # Process in parallel
        verification_results = process_duplicate_pairs_batch(theme_pairs, config)

        all_evaluations = []
        matches_found = []
        
        # Apply results to our Union-Find structure
        for result, (i, j) in zip(verification_results, potential_duplicates):
            score = result['is_duplicate']

            if debug:
                pair_info = {
                    'theme1': themes[i]['name'],
                    'theme2': themes[j]['name'],
                    'score': score,
                    'is_match': score >= DUPLICATE_SCORE_THRESHOLD
                }
                
                # Store evaluation result
                status = "MATCH" if pair_info['is_match'] else "NOT A MATCH"
                st.session_state["dedup_debug_info"]["evaluation_results"].append(
                    f"Pair: '{pair_info['theme1']}' and '{pair_info['theme2']}' - Score: {pair_info['score']}/10 - {status}"
                )
            
            if score >= DUPLICATE_SCORE_THRESHOLD:
                union(i, j)
                stats["approximate_matches"] += 1
    
    # Step 4: Group themes by their root in the Union-Find structure
    groups = {}
    for i in range(len(themes)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Step 5: Merge themes in each group
    merged_themes = []
    merged_groups = []
    
    for root, indices in groups.items():
        if len(indices) == 1:
            # Single theme, no duplicates
            merged_themes.append(themes[indices[0]])
        else:
            # Multiple themes to merge
            merged = themes[indices[0]].copy()

            if debug:
                # Store merged group info
                merged_group_names = [themes[idx]['name'] for idx in indices]
                st.session_state["dedup_debug_info"]["merged_groups"].append(merged_group_names)
                
            for i in range(1, len(indices)):
                merged = merge_duplicate_themes(merged, themes[indices[i]])
            merged_themes.append(merged)
    
    stats["total_reduced"] = len(themes) - len(merged_themes)
    
    return merged_themes, stats

def display_deduplication_debug_info():
    """Display persistent debug information from the last deduplication run."""
    if "dedup_debug_info" in st.session_state and st.session_state["dedup_debug_info"]:
        
        st.markdown("**Exact Matches:**")
        exact_matches = st.session_state["dedup_debug_info"].get("exact_matches", [])
        if exact_matches:
            for match in exact_matches:
                st.write(match)
        else:
            st.write("No exact matches found")
        
        st.markdown("**All LLM Evaluation Results:**")
        eval_results = st.session_state["dedup_debug_info"].get("evaluation_results", [])
        if eval_results:
            for result in eval_results:
                st.write(result)
        else:
            st.write("No evaluation results")
        
        merged_groups = st.session_state["dedup_debug_info"].get("merged_groups", [])
        if merged_groups:
            st.markdown("**Merged Theme Groups:**")
            for idx, group in enumerate(merged_groups):
                st.markdown(f"**Merged group {idx+1}:**")
                for name in group:
                    st.write(f"- {name}")
                st.write("---")

def verify_duplicate_pair(theme_pair, config):
    """Verify if a pair of themes refers to the same concept using LLM."""
    theme1, theme2 = theme_pair
    
    # Default to the classification settings if not specified
    if config is None:
        config = ClassificationConfig(st.session_state)
    
    # Build the prompt
    system_msg = LLMMessage(
        role="system",
        content="Your task is to evaluate whether two themes are likely representing the same underlying concept."
    )
    
    # Get examples for the themes
    examples1 = theme1.get('examples', []) if isinstance(theme1.get('examples', []), list) else []
    examples2 = theme2.get('examples', []) if isinstance(theme2.get('examples', []), list) else []
    
    # Format examples for display
    example_text1 = "\n".join([f"- {ex}" for ex in examples1[:3]]) if examples1 else "No examples"
    example_text2 = "\n".join([f"- {ex}" for ex in examples2[:3]]) if examples2 else "No examples"
    
    # Create prompt focusing on names and examples (not descriptions)
    user_msg_content = f"""
    I have two themes that might be duplicates, just referred to with different names or spellings:
    
    Theme 1: "{theme1['name']}"
    Examples:
    {example_text1}
    
    Theme 2: "{theme2['name']}"
    Examples:
    {example_text2}
    
    How likely do you think it is that these were intended to refer to the same concept?
    Note that if the themes include recognised brand names, two different brand names should always be considered different concepts.
    
    Reply with a number from 0-10, where 0 means "definitely referring to different concepts" and 10 means "definitely referring to the same concept".

    If the themes refer to brand or product names, focus on the theme name over the description as the same concept may be described differently.
    """

    user_msg = LLMMessage(
        role="user",
        content=user_msg_content
    )
    
    llm_request = LLMCall(
        messages=[system_msg, user_msg],
        model_name=config.model_name,
        temperature=config.get_effective_temperature(),
        format=DuplicateVerification
    )
    llm_response = call_llm(llm_request, config)
    
    try:
        # First try to parse the structured data
        verification_data = DuplicateVerification.model_validate_json(llm_response.content)
        score = verification_data.is_duplicate
    except Exception:
        # Fallback to simple text parsing to extract a number
        response_text = llm_response.content.strip()
        score = 0
        number_match = re.search(r'\b([0-9]|10)\b', response_text)
        if number_match:
            score = int(number_match.group(1))
    
    return {
        "theme1": theme1,
        "theme2": theme2,
        "is_duplicate": score,
        "raw_response": llm_response.content
    }

def process_duplicate_pairs_batch(theme_pairs, config, max_workers=MAX_WORKERS):
    """Process a batch of theme pairs in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(theme_pairs))) as executor:
        future_to_pair = {
            executor.submit(verify_duplicate_pair, pair, config): pair 
            for pair in theme_pairs
        }
        
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log error but continue with other pairs
                if is_main_thread():
                    st.error(f"Error processing pair {pair[0]['name']} and {pair[1]['name']}: {str(e)}")
    
    return results

def merge_duplicate_themes(theme1, theme2):
    """Merge two duplicate themes."""
    # Choose the name with the more common format or longer name
    # This is a simple heuristic - could be improved
    if len(theme1['name']) >= len(theme2['name']):
        primary_name = theme1['name']
    else:
        primary_name = theme2['name']
    
    # Combine descriptions
    descriptions = [theme1.get('description', ''), theme2.get('description', '')]
    combined_description = max(descriptions, key=len)  # Use the longer description
    
    # Combine examples, removing duplicates
    examples1 = theme1.get('examples', []) if isinstance(theme1.get('examples', []), list) else []
    examples2 = theme2.get('examples', []) if isinstance(theme2.get('examples', []), list) else []
    
    # Use a set to remove duplicates, preserving order as much as possible
    combined_examples = []
    seen = set()
    
    # Add examples from both themes, avoiding duplicates
    for example in examples1 + examples2:
        if example and example.lower() not in seen:
            combined_examples.append(example)
            seen.add(example.lower())
    
    # Limit to top 3 examples
    combined_examples = combined_examples[:3]
    
    # Preserve group: if both themes have the same group, keep it; otherwise default to None
    group1 = theme1.get("group")
    group2 = theme2.get("group")
    merged_group = group1 if group1 == group2 else None

    return {
        "name": primary_name,
        "description": combined_description,
        "examples": combined_examples,
        "group": merged_group,
    }

###############################################################################
# 7. THEME CLASSIFICATION
###############################################################################
# ui
def show_classification_model_selection():
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    all_models = get_all_models()
    
    # Check if the currently saved model exists in available models
    current_model = st.session_state.get("classification_model_selected")
    index = 0  # Default to first model
    
    # Find the index of the current model if it exists in available models
    if current_model in all_models:
        index = all_models.index(current_model)

    selected_classification_model = st.selectbox("LLM Model for Classification:", all_models, index=index, key="classification_model_selected", help="Model used for classifying responses against themes (Step 2). Faster, cheaper models often work well here.")
    render_gpt5_ui_if_applicable(selected_classification_model, key_prefix="classification_")
    
    if selected_classification_model.startswith("Ollama: "):
        ollama_model = selected_classification_model.split("Ollama: ")[1]
        st.warning(f"Warning: To use the selected Ollama model, run locally with: `ollama run {ollama_model}`")
    
    if selected_classification_model not in MODELS_WITHOUT_TEMPERATURE:
        st.slider(
            "Temperature (creativity):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("classification_temperature"),
            step=0.1,
            help="Higher values make output more random, lower values more deterministic.",
            key="classification_temperature"
        )
    
    previous_batch_size = st.session_state.get("previous_batch_size", 3)
    st.number_input(
        "Batch size for classification calls",
        min_value=1, 
        max_value=100, 
        value=3,
        key="batch_size",
        help="Number of responses sent to the LLM in each classification call. Larger batches are faster but use more tokens per call and historically can show weaker performance."
    )
    batch_size = st.session_state.get("batch_size", 3)
    if batch_size != previous_batch_size:
        st.session_state["previous_batch_size"] = batch_size
        # Clear stored batches to force recalculation
        if "classification_batches" in st.session_state:
            del st.session_state["classification_batches"]
    
    # Calculate batches if needed
    if "classification_batches" not in st.session_state and st.session_state["reasoning"]["current_themes"]:
        valid_indices = df[pd.notna(df[question_col])].index.tolist()
        all_batches = []
        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start:start+batch_size]
            all_batches.append(batch_indices)
            
        st.session_state["classification_batches"] = all_batches

    st.text_area(
        "Additional instructions for the classification model (optional):",
        help="Provide any extra instructions for the model to consider during classification.",
        key="additional_classification_instructions"
    )

    st.toggle(
        "Include theme examples in classification context",
        help="When disabled, only theme names and descriptions are sent to the LLM.",
        key="include_examples_in_classification",
        value=True
    )
    
    # Show "Classify within groups only" toggle when theme set has group info
    _active_q = st.session_state["data"]["active_question"]
    _active_tsid = st.session_state["reasoning"]["active_theme_set_ids"].get(_active_q)
    _active_ts = {}
    if _active_tsid:
        _active_ts = st.session_state["reasoning"]["theme_sets_by_question"].get(_active_q, {}).get(_active_tsid, {})
    _has_groups = bool(_active_ts.get("valid_groups"))
    if _has_groups:
        st.toggle(
            "Classify within groups only",
            help="When enabled, each response is only classified against themes from its group (plus universal 'No group' themes). "
                 f"Grouping variable: {_active_ts.get('grouping_variable', 'N/A')}",
            key="classify_within_groups",
            value=True
        )
    
    if st.session_state.get("classification_batches") and st.session_state["reasoning"]["current_themes"]:
        themes = st.session_state["reasoning"]["current_themes"]

        # Use first batch for token usage estimation
        first_batch_indices = st.session_state["classification_batches"][0]
        # Ensure classification prompt preview uses cleaned responses
        first_batch_texts = [
            (str(x).strip() if pd.notna(x) else "")
            for x in df.loc[first_batch_indices, question_col].tolist()
        ]
            
        # Create config
        config = ClassificationConfig(st.session_state)
            
        # Get prompt for the first batch
        system_msg, user_msg = build_classification_prompt(
            first_batch_texts, 
            themes, 
            config
        )
            
        # Calculate token usage
        estimated_token_usage = calculate_token_usage(system_msg, user_msg, config.model_name, usage_type="classification")
            
        # Display token usage status
        show_token_usage_status(
            estimated_token_usage,
            description="Input token usage"
        )
    
    can_classify = bool(st.session_state.get("classification_batches") and st.session_state["reasoning"]["current_themes"])

    if st.button("Classify into Themes", disabled=not can_classify):
        if not st.session_state.get("include_examples_in_classification", True):
            st.info("ℹ️ Classifying using theme names and descriptions only (examples excluded)")
        with st.spinner('Classifying responses into themes...'):
            cfg_local = ClassificationConfig(st.session_state)
            classify_verbatims_in_batches(cfg_local)

# build prompt
def build_classification_prompt(response_texts: List[str],
                                themes: List[Dict],
                                config: ClassificationConfig
                                ) -> tuple[str, str]:
    """
    Builds a prompt for classifying survey responses into themes.
    
    Parameters:
        response_texts: List of responses to classify for this batch.
        themes: List of theme dicts to classify against.
        config: ClassificationConfig object with settings and instructions.

    Returns:
        A tuple containing:
            - system_msg_content: Complete system message for LLM
            - user_msg_content: Complete user message for LLM for the batch
    """
    batch_size = len(response_texts)
    additional_classification_instructions = config.additional_instructions

    # Build the prompt text using the current themes
    theme_definitions = ""
    for i, t in enumerate(themes, start=1):
        if i > 1:
            theme_definitions += "\n"
        
        theme_definitions += f"\nTheme {i}: {t['name']}\nDescription: {t['description']}"

        # Add examples if they exist
        if (config.include_examples and "examples" in t and t["examples"]):
            theme_definitions += "\nExamples:"
            for j, example in enumerate(t["examples"], start=1):
                theme_definitions += f"\n  {j}) {example}"

    prompt_text = f"""
    Your task is to classify each survey response into one or more themes. Think about each response individually and what themes apply.
    """
    if additional_classification_instructions:
        prompt_text += f"\n\nAdditional Instructions:\n{additional_classification_instructions}\n"

    prompt_text += f"""
    For each response, you should:
    1. For each theme, look for any substrings within the response that are good examples of that theme. These are 'supporting quotes'.
    2. Identify the themes with at least one supporting quote.
    
    \n    Use the exact theme names as listed below. Do not invent new names or renumber themes.
    \n    Return a valid JSON object with a key "root" whose value is an array of objects. Each object should follow the format below:
    \n\n
   [
        {{
            "index": 0,
            "themes": ["<THEME_NAME_1>", "<THEME_NAME_3>"],
            "justifications": {{
                "<THEME_NAME_1>": ["quote supporting theme1", "another supporting quote"],
                "<THEME_NAME_3>": ["quote supporting theme3"]
            }}
        }},
        ...
    ]
    \n\n    If no theme applies for a response, "themes" should be an empty array and "justifications" an empty object.
    
    \nWe have the following themes:
    {theme_definitions}

    \nBelow are survey responses, each labeled by index i (0-based within this list).
    """

    # Format response texts if requested
    formatted_responses = "\n".join(f"{i}) {txt}" for i, txt in enumerate(response_texts))
    user_msg_content = prompt_text + "\n\nResponses:\n" + formatted_responses

    # Build complete messages
    system_msg_content = "You are a highly skilled qualitative market researcher, trained in the nuanced interpretation and classification of verbatim responses into themes and the sensitive adherence to additional instructions provided by your client. You have been tasked with classifying verbatim responses into the provided themes. You must return valid JSON only."
    
    return system_msg_content, user_msg_content

def classify_verbatims_in_batches(config: ClassificationConfig, skip_rerun=False):
    """
    Classifies survey responses into themes in batches using the LLM and updates the classification results in session state.
    """
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    themes = st.session_state["reasoning"]["current_themes"]
    
    batch_size = st.session_state.get("batch_size", 10)

    model_name = config.model_name
    temperature = config.temperature
    
    # Get the active theme set ID
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question_col)
    if not active_theme_set_id:
        st.error("No active theme set found.")
        return

    unique_id_col = st.session_state["data"].get("unique_id_column")
    if not unique_id_col:
        df["unique_id"] = range(1, len(df) + 1)
        unique_id_col = "unique_id"

    result_df = df[[unique_id_col, question_col]].copy()
    theme_names = [t["name"] for t in themes]
    # reset all coding
    for theme_name in theme_names:
        result_df[theme_name] = 0

    theme_name_by_index = {i + 1: name for i, name in enumerate(theme_names)}
    def _normalize_label(label: str) -> str:
        normalized = re.sub(r"\s+", " ", str(label).strip().lower())
        return normalized

    normalized_theme_map = {_normalize_label(name): name for name in theme_names}

    def _resolve_theme_name(label):
        if label is None:
            return None
        if isinstance(label, int):
            if 1 <= label <= len(theme_names):
                return theme_name_by_index[label]
            return None
        text = str(label).strip()
        if text.isdigit():
            idx = int(text)
            if 1 <= idx <= len(theme_names):
                return theme_name_by_index[idx]
        match = re.search(r"\btheme\s*(\d+)\b", text, flags=re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= len(theme_names):
                return theme_name_by_index[idx]
        return normalized_theme_map.get(_normalize_label(text))

    # Clear existing justifications
    st.session_state["classification"]["justifications"] = {}
    # Store dataframe indices for later reference
    st.session_state["classification"]["df_indices"] = list(result_df.index)

    all_indices = list(result_df.index)

    # Determine if group-scoped classification is active
    classify_within_groups = st.session_state.get("classify_within_groups", False)
    active_ts = st.session_state["reasoning"]["theme_sets_by_question"].get(question_col, {}).get(active_theme_set_id, {})
    grouping_variable = active_ts.get("grouping_variable")
    valid_groups = active_ts.get("valid_groups", [])
    use_grouped_classification = classify_within_groups and grouping_variable and valid_groups

    # Build batches: list of (batch_indices, themes_for_batch)
    batches_with_themes = []
    if use_grouped_classification:
        # Partition indices by group
        universal_themes = [t for t in themes if t.get("group") is None]
        group_to_themes = {}
        for t in themes:
            g = t.get("group")
            if g is not None:
                group_to_themes.setdefault(g, []).append(t)
        
        group_to_indices = {}
        unmatched_indices = []
        for idx in all_indices:
            val = df.loc[idx, grouping_variable] if grouping_variable in df.columns else None
            if pd.notna(val) and val in valid_groups:
                group_to_indices.setdefault(val, []).append(idx)
            else:
                unmatched_indices.append(idx)
        
        # Create batches per group
        for group_val, g_indices in group_to_indices.items():
            group_themes = group_to_themes.get(group_val, []) + universal_themes
            if not group_themes:
                group_themes = themes  # Fallback to all themes if none match
            for start in range(0, len(g_indices), batch_size):
                batch_indices = g_indices[start:start+batch_size]
                batches_with_themes.append((batch_indices, group_themes))
        
        # Unmatched responses get universal themes only (or all themes if no universal)
        if unmatched_indices:
            fallback_themes = universal_themes if universal_themes else themes
            for start in range(0, len(unmatched_indices), batch_size):
                batch_indices = unmatched_indices[start:start+batch_size]
                batches_with_themes.append((batch_indices, fallback_themes))
    else:
        # Standard: all responses against all themes
        for start in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[start:start+batch_size]
            batches_with_themes.append((batch_indices, themes))

    # Initialize sparse structures
    sparse_results = {}
    sparse_justifications = {}

    # Always use parallel processing for classification
    total_batches = len(batches_with_themes)
    with st.spinner(f'Processing {total_batches} batches with parallel processing using model {model_name}...'):
        # Create progress tracking
        progress_bar = st.progress(0)
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_batches)) as executor:
            futures = []
            
            # Submit all batch processing jobs
            for batch_indices, batch_themes in batches_with_themes:
                batch_texts = result_df.loc[batch_indices, question_col].fillna("").tolist()
                future = executor.submit(
                    process_batch, 
                    batch_indices, 
                    batch_texts,
                    batch_themes,
                    config
                )
                futures.append(future)
            
            # Process results as they complete
            completed = 0
            nonempty_results = 0
            for future in as_completed(futures):
                completed += 1
                progress = completed / len(futures)
                progress_bar.progress(progress)
                
                result_tuple = future.result()
                # Support both old (indices, parsed_results) and new (indices, parsed_results, error_msg, raw)
                if len(result_tuple) == 2:
                    batch_indices, parsed_results = result_tuple
                    error_msg = None
                    raw_output = None
                else:
                    batch_indices, parsed_results, error_msg, raw_output = result_tuple

                if error_msg:
                    st.error(f"Batch error: {error_msg}")
                    if raw_output is not None:
                        with st.expander("Raw output for failed batch"):
                            st.write(raw_output)
                for item in parsed_results:
                    idx = item.index
                    if isinstance(idx, str) and idx.isdigit():
                        idx = int(idx)
                    real_index = None
                    if isinstance(idx, int):
                        if 0 <= idx < len(batch_indices):
                            real_index = batch_indices[idx]
                        elif idx in batch_indices:
                            real_index = idx
                    if real_index is None:
                        continue

                    # Only store if themes were found
                    if item.themes:
                        # Normalize theme names
                        normalized_themes = []
                        
                        for tname in item.themes:
                            resolved = _resolve_theme_name(tname)
                            if resolved:
                                normalized_themes.append(resolved)
                        
                        if normalized_themes:
                            sparse_results[real_index] = list(dict.fromkeys(normalized_themes))
                            nonempty_results += 1
                    
                            # Normalize justifications
                            normalized_justifications = {}
                            if hasattr(item, 'justifications') and isinstance(item.justifications, dict):
                                for theme_key, quotes in item.justifications.items():
                                    matched_theme = _resolve_theme_name(theme_key) or theme_key
                                    normalized_justifications[matched_theme] = quotes

                            sparse_justifications[real_index] = normalized_justifications
                        
    # Store justifications
    st.session_state["classification"]["justifications"] = sparse_justifications

    # Store sparse results in session state
    st.session_state["classification"]["sparse_results"] = sparse_results
    
    # Store results for the active theme set
    if question_col not in st.session_state["classification"]["results_by_question"]:
        st.session_state["classification"]["results_by_question"][question_col] = {}
    
    st.session_state["classification"]["results_by_question"][question_col][active_theme_set_id] = sparse_results
    
    update_theme_counts()
    
    # Store justifications for the active theme set
    if "justifications_by_theme_set" not in st.session_state["classification"]:
        st.session_state["classification"]["justifications_by_theme_set"] = {}
        
    if question_col not in st.session_state["classification"]["justifications_by_theme_set"]:
        st.session_state["classification"]["justifications_by_theme_set"][question_col] = {}

    st.session_state["classification"]["justifications_by_theme_set"][question_col][active_theme_set_id] = sparse_justifications
    
    # record last run summary
    st.session_state.setdefault("classification", {})
    st.session_state["classification"]["last_run_summary"] = {"batches": len(batches_with_themes), "nonempty_results": nonempty_results}

    if not skip_rerun:
        st.rerun()

def process_batch(batch_indices: List[int],
                  batch_texts: List[str],
                  themes: List[Dict],
                  config: ClassificationConfig
                  ) -> tuple[List[int], List[ClassificationResult], str | None, str | None]:
    """
    Process a batch of responses for classification.
    """
    
    system_content, user_content = build_classification_prompt(batch_texts, themes, config)

    system_msg = LLMMessage(role="system", content=system_content)
    user_msg = LLMMessage(role="user", content=user_content)

    # Calculate token usage before making the API call - only in main thread
    if threading.current_thread() is threading.main_thread():
        usage = calculate_token_usage(system_msg.content, user_msg.content, config.model_name, usage_type="classification")
        show_token_usage_status(usage, description="Input token usage")
        if usage["count"] >= usage["limit"] > 0:
            st.error(f"Prompt is too long ({usage['count']} tokens). Context limit is {usage['limit']} tokens.")
            return batch_indices, []

    model_name = config.model_name

    llm_request = LLMCall(
        messages=[system_msg, user_msg],
        model_name=model_name,
        temperature=config.get_effective_temperature(), 
        format=ClassificationResults
    )
    llm_response = call_llm(llm_request, config)

    error_msg = None
    raw_output = None
    if isinstance(llm_response.content, str) and llm_response.content.startswith("Error:"):
        error_msg = llm_response.content
        raw_output = llm_response.content
        return batch_indices, [], error_msg, raw_output

    try:
        parsed_results = ClassificationResults.model_validate_json(llm_response.content).root
        return batch_indices, parsed_results, None, None
    except Exception as e:
        error_msg = f"Structured output validation error in batch: {e}"
        raw_output = llm_response.content
        return batch_indices, [], error_msg, raw_output

def reclassify_themes():
    """
    Re-runs the classification process for all responses using the current themes.
    
    Called when themes have been edited and existing classification results need
    to be updated. Uses stored model settings from when the reclassification was
    requested rather than current settings, allowing users to adjust parameters
    without affecting a queued reclassification.
    
    Reuses the previously generated classification prompt instead of rebuilding it.
    
    No parameters or return values.
    """
    with st.spinner('Reclassifying responses into themes...'):
        config = ClassificationConfig(st.session_state)
        classify_verbatims_in_batches(config,
                                    skip_rerun=True)
    st.success("Reclassification complete! The results table has been updated.")
    st.session_state["reclassify_requested"] = False 

###############################################################################
# 8. VISUALISATION
###############################################################################

def show_classification_coverage():
    """
    Displays the percentage and count of responses classified with at least one theme.
    """
    sparse_results = st.session_state["classification"].get("sparse_results", {})
    
    classified_count = len(sparse_results)
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    total_count = len(df)
    percentage = (classified_count / total_count) * 100 if total_count > 0 else 0
    
    # Compute counts among responses with non-missing, non-whitespace-only values
    non_whitespace_mask = df[question_col].fillna("").astype(str).str.strip() != ""
    non_missing_total = int(non_whitespace_mask.sum())
    non_missing_pct = (classified_count / non_missing_total * 100) if non_missing_total > 0 else 0
    
    st.info(f"{classified_count} of {total_count} total responses ({percentage:.1f}%) were classified with at least one theme. ({classified_count} of {non_missing_total} responses ({non_missing_pct:.1f}%) that were not missing data or whitespace)")

def show_classification_results_table():
    # lazy load agrid if required
    global AgGrid, GridOptionsBuilder, GridUpdateMode
    if 'AgGrid' not in globals():
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    
    sparse_results = st.session_state["classification"].get("sparse_results", {})
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    theme_names = [t["name"] for t in st.session_state["reasoning"]["current_themes"]]
    unique_id_col = st.session_state["data"]["unique_id_column"]

    total_cells = len(df) * len(theme_names)
    AGGRID_THRESHOLD = 1_000_000  # 1M cells for AgGrid display

    if total_cells > AGGRID_THRESHOLD:
        show_large_dataset_summary(sparse_results)
    else:
        df_classified = sparse_to_dense_dataframe(
            sparse_results, df, question_col, theme_names, unique_id_col
        )
        # Create a DataFrame with theme counts
        theme_counts_data = {
            "Theme": theme_names,
            "Count": [sum(1 for themes in sparse_results.values() if theme in themes) for theme in theme_names]
        }
        theme_counts_df = pd.DataFrame(theme_counts_data)
        theme_counts_df = theme_counts_df.sort_values("Count", ascending=False)

        # Get the active theme set ID to use in the key
        active_question = st.session_state["data"]["active_question"]
        active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(active_question, "none")

        # Create a dynamic key for the AgGrid based on the active theme set
        grid_key = f"theme_grid_{active_question}_{active_theme_set_id}"

        # Make the dataframe interactive with a callback
        selected_indices = []
        df_display = df_classified.copy()

        # Get profiling columns
        profiling_cols = st.session_state["data"].get("profiling_columns", [])
        
        # Show filter options for profiling columns
        if profiling_cols:
            filter_applied = False
            
            # Create a copy to avoid modifying the df_display directly
            filtered_df = df_display.copy()
            
            # Create filter controls for each profiling column
            for col in profiling_cols:
                if col in st.session_state["data"]["survey"].columns:
                    # Add the profiling column to the display dataframe
                    filtered_df[col] = st.session_state["data"]["survey"].loc[filtered_df.index, col].values
                    
                    # Get unique values for the column, sorted
                    unique_values = sorted(filtered_df[col].dropna().unique())
                    
                    # Only show filter if fewer than 20 unique values (categorical data)
                    if len(unique_values) < 20:
                        selected_values = st.multiselect(
                            f"Filter by {col}:",
                            options=unique_values,
                            default=unique_values,
                            key=f"filter_profiling_{active_question}_{active_theme_set_id}_{col}"
                        )
                        
                        if selected_values and len(selected_values) < len(unique_values):
                            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                            filter_applied = True
            
            if filter_applied:
                # Update the display dataframe with the filtered rows
                # Keep only the original columns to avoid duplicating profiling columns
                original_cols = df_display.columns
                df_display = filtered_df[original_cols].copy()
                
                # Show message about filtering
                st.info(f"Showing {len(df_display)} of {len(df_classified)} responses after filtering")
        
        # Add profiling columns to display dataframe after filtering
        for col in profiling_cols:
            if col in st.session_state["data"]["survey"].columns:
                # Add profiling column to display dataframe
                df_display[col] = st.session_state["data"]["survey"].loc[df_display.index, col].values
        
        # Create a custom aggrid configuration that allows row selection
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_selection('single')

        # Set column order
        column_order = []
        if unique_id_col:
            column_order.append(unique_id_col)
        column_order.append(question_col)
        column_order.extend(theme_names)
        column_order.extend(profiling_cols)

        for col in column_order:
            if col in df_display.columns:
                if col == question_col:
                    gb.configure_column(col, headerName=col, maxWidth=400)
                else:
                    gb.configure_column(col, headerName=col)

        gridOptions = gb.build()
        
        grid_response = AgGrid(
            df_display,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=False,
            key=grid_key
        )

        # Rest of the function handling selection and justifications display...
        all_indices = st.session_state["classification"].get("df_indices", [])
        
        # Show theme justifications if a row is selected
        if 'selected_rows' in grid_response:
            selected_rows = grid_response['selected_rows']
            
            # Only proceed if we have a non-empty DataFrame
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                # Get the first selected row
                selected_row = selected_rows.iloc[0]
                
                # Find the row's position in the original dataframe by comparing values
                # We'll use the response text column to match
                if question_col in selected_row and question_col in df_classified:
                    selected_text = selected_row[question_col]
                    
                    # Look through all_indices to find a matching response
                    found_index = None
                    
                    for orig_idx in all_indices:
                        if orig_idx in df_classified.index and df_classified.at[orig_idx, question_col] == selected_text:
                            found_index = orig_idx
                            break
                    
                    # If we found a matching index
                    if found_index is not None:
                        justifications = st.session_state["classification"]["justifications"].get(found_index, {})

                        if justifications and len(justifications) > 0:
                            st.write("Theme Justifications:")
                            
                            # Get the theme names that are marked as 1 in the results for this row
                            active_themes = sparse_results.get(found_index, [])
                            
                            # Display justifications for active themes
                            for theme in active_themes:
                                # Try multiple variations of the theme name to find justifications
                                if theme in justifications:
                                    with st.expander(f"Theme: {theme}", expanded=True):
                                        for i, quote in enumerate(justifications[theme]):
                                            st.write(f"{i+1}. \"{quote}\"")
                                else:
                                    with st.expander(f"Theme: {theme}", expanded=True):
                                        st.write("No specific justification quotes available.")
                        else:
                            st.write("No justifications found for this response.")

def create_theme_count_chart(theme_names, title="Theme Counts", 
                           filter_key_prefix="filter", 
                           show_filters=True,
                           filter_columns=None,
                           sparse_results=None):
    """
    Create a standardized Altair bar chart for theme counts with filtering on any column.
    
    Parameters:
        theme_names: List of theme names to count
        title: Chart title
        filter_key_prefix: Prefix for Streamlit widget keys
        show_filters: Whether to show filtering controls
        filter_columns: List of column names to offer as filters
        sparse_results: Dict mapping indices to theme lists (defaults to session state)
    """
    # Lazy import altair
    global alt
    if 'alt' not in globals():
        import altair as alt
    
    if sparse_results is None:
        sparse_results = st.session_state["classification"].get("sparse_results", {})
    survey_df = st.session_state["data"]["survey"]

    theme_counts_dict = {}
    
    # Apply filtering if enabled
    if show_filters and filter_columns:    
        filter_applied = False
        filtered_indices = set(survey_df.index)
        
        # Get all columns that might be useful for filtering 
        # (exclude theme columns and the question text column)
        question_col = st.session_state["data"]["active_question"]
        
        # Create filter controls for each viable column
        for col in filter_columns:
            if col in survey_df.columns:
                unique_values = sorted(survey_df[col].dropna().unique())
            
                # Only show filter if fewer than 20 unique values (categorical data)
                if len(unique_values) < 20:
                    selected_values = st.multiselect(
                        f"Filter by {col}:",
                        options=unique_values,
                        default=unique_values,
                        key=f"{filter_key_prefix}_{col}"
                    )
                    
                    if selected_values and len(selected_values) < len(unique_values):
                        col_filtered = survey_df[survey_df[col].isin(selected_values)].index
                        filtered_indices &= set(col_filtered)
                        filter_applied = True
        
        # Display filter info
        if filter_applied:
            st.info(f"Showing {len(filtered_indices)} of {len(survey_df)} responses after filtering")

        # Count themes only for filtered indices
        theme_frequency = {}
        for idx, response_themes in sparse_results.items():
            if idx in filtered_indices:
                for theme_name in response_themes:
                    theme_frequency[theme_name] = theme_frequency.get(theme_name, 0) + 1
        
        # Convert to counts dict format
        total_filtered = len(filtered_indices)
        for theme in st.session_state["reasoning"]["current_themes"]:
            theme_name = theme["name"]
            count = theme_frequency.get(theme_name, 0)
            percentage = (count / total_filtered) * 100 if total_filtered > 0 else 0
            theme_counts_dict[theme_name] = {"count": count, "percentage": percentage}

    else:
        theme_frequency = {}
        for response_themes in sparse_results.values():
            for theme_name in response_themes:
                theme_frequency[theme_name] = theme_frequency.get(theme_name, 0) + 1

        # Convert to counts dict format
        total_responses = len(survey_df)
        for theme_name in theme_names:
            count = theme_frequency.get(theme_name, 0)
            percentage = (count / total_responses) * 100 if total_responses > 0 else 0
            theme_counts_dict[theme_name] = {"count": count, "percentage": percentage}
    
    # Create chart data
    theme_counts_data = {
        "Theme": theme_names,
        "Count": [theme_counts_dict.get(theme, {}).get("count", 0) for theme in theme_names]
    }
    theme_counts_df = pd.DataFrame(theme_counts_data).sort_values("Count", ascending=False)

    # Calculate domain based on data
    max_count = theme_counts_df["Count"].max()
    x_domain = [0, int(max_count) + 1] if max_count > 0 and not pd.isna(max_count) else [0, 1]

    # Create the bar chart
    bar = alt.Chart(theme_counts_df).mark_bar().encode(
        y=alt.Y("Theme:N", sort='-x', title="Theme", axis=alt.Axis(labelLimit=0)),
        x=alt.X("Count:Q", title="Response Count", scale=alt.Scale(domain=x_domain),
                axis=alt.Axis(tickMinStep=1, format="d")),
        tooltip=["Theme", "Count"]
    ).properties(
        title=title,
        width=600
    )
    
    # Create text labels
    text = alt.Chart(theme_counts_df).mark_text(
        align='left',
        dx=3,
        fontSize=16
    ).encode(
        y=alt.Y("Theme:N", sort='-x'),
        x=alt.X("Count:Q", scale=alt.Scale(domain=x_domain)),
        text=alt.Text("Count:Q", format="d")
    )
    
    return bar + text

def show_large_dataset_summary(sparse_results):
    """
    Show summary UI for datasets too large for AgGrid.
    """
    survey_df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    sparse_justifications = st.session_state["classification"].get("sparse_justifications", 
                                                                  st.session_state["classification"].get("justifications", {}))
    
    st.warning(f"⚠️ Dataset too large for interactive table ({len(survey_df):,} responses × {len(st.session_state['reasoning']['current_themes'])} themes). Showing summary view.")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total responses", f"{len(survey_df):,}")
    with col2:
        st.metric("Responses with themes", f"{len(sparse_results):,}")
    with col3:
        st.metric("Responses with no themes", f"{len(survey_df) - len(sparse_results):,}")
    
    # Sample viewer with search
    st.subheader("Response Viewer")
    
    search_term = st.text_input("Search responses:", key="sparse_search")
    
    # Filter samples based on search
    if search_term:
        matching_indices = []
        for idx in sparse_results.keys():
            response_text = str(survey_df.loc[idx, question_col])
            if search_term.lower() in response_text.lower():
                matching_indices.append(idx)
        sample_indices = matching_indices[:20]
        st.caption(f"Showing {len(sample_indices)} of {len(matching_indices)} matching responses")
    else:
        sample_indices = list(sparse_results.keys())[:20]
        st.caption(f"Showing first {len(sample_indices)} classified responses")
    
    # Display samples
    for idx in sample_indices:
        response_text = survey_df.loc[idx, question_col]
        themes = sparse_results[idx]
        
        with st.expander(f"Response {idx}: {response_text[:100]}..."):
            st.write("**Full response:**")
            st.write(response_text)
            
            st.write("**Classified themes:**")
            for theme in themes:
                st.write(f"• {theme}")
                
                # Show justifications if available
                if idx in sparse_justifications and theme in sparse_justifications[idx]:
                    quotes = sparse_justifications[idx][theme]
                    if quotes:
                        st.write("  Justifications:")
                        for quote in quotes:
                            st.write(f"  - \"{quote}\"")

def show_themes_counts_chart():
    """
    Shows theme count visualizations with profiling column selection and filtering.
    """
    
    # Get the active theme set ID and question for use in keys
    active_question = st.session_state["data"]["active_question"]
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(active_question, "none")
    
    # Use a specific state key for chart profiling columns
    chart_profiling_key = f"chart_profiling_{active_question}_{active_theme_set_id}"
    
    # Add the profiling column selection UI for charts
    survey_df = st.session_state["data"]["survey"]
    if survey_df is not None:
        # Get all available columns excluding the active question and unique ID
        question_col = st.session_state["data"]["active_question"]
        unique_id_col = st.session_state["data"]["unique_id_column"]
        
        available_cols = [col for col in survey_df.columns 
                        if col != question_col and col != unique_id_col]
        
        # Filter to categorical columns (fewer than 20 unique values)
        categorical_cols = []
        for col in available_cols:
            if len(survey_df[col].dropna().unique()) < 20:
                categorical_cols.append(col)
        
        # Only show filters checkbox if there are filters available
        show_filters_checkbox = False
        if categorical_cols:
            show_filters_checkbox = st.checkbox(
                "Show filters",
                key=f"show_filters_chart_{active_question}_{active_theme_set_id}"
            )
        
        if show_filters_checkbox:
            # Get previously selected columns
            current_profiling_cols = st.session_state.get(chart_profiling_key, [])
            
            # Filter to only include valid columns that still exist in categorical_cols
            valid_current_cols = [col for col in current_profiling_cols if col in categorical_cols]
            
            # Show multiselect with previously selected columns as default
            selected_cols = st.multiselect(
                "Select columns for filtering:",
                options=categorical_cols,
                default=valid_current_cols,
                key=f"filter_cols_select_chart_{active_question}_{active_theme_set_id}"
            )
            
            # Update the stored selection (no button needed)
            st.session_state[chart_profiling_key] = selected_cols
            
            # Set profiling_cols for chart
            profiling_cols = selected_cols
        else:
            # If checkbox is unchecked, no columns for filtering
            profiling_cols = []
    else:
        # If no categorical columns, no filtering
        profiling_cols = []
        show_filters_checkbox = False
    
    theme_names = [t["name"] for t in st.session_state["reasoning"]["current_themes"]]
    
    # Get the profiling columns for this chart
    profiling_cols = st.session_state.get(chart_profiling_key, [])
    
    # Chart, with filtering
    filter_key_prefix = f"filter_chart_{active_question}_{active_theme_set_id}"
    chart = create_theme_count_chart(
        theme_names=theme_names,
        title="Count of Responses Coded as Each Theme",
        filter_key_prefix=filter_key_prefix,
        show_filters=show_filters_checkbox,
        filter_columns=profiling_cols,
        sparse_results=None
    )
    
    st.altair_chart(chart, use_container_width=True)

def show_all_themes_summary():
    """
    Shows summary visualizations for all processed questions.
    """
    # Create tabs for different summary views
    tab1, tab2 = st.tabs(["By Question", "All Themes"])
    
    with tab1:
        for question in st.session_state["data"]["processed_questions"]:
            with st.expander(f"{question}", expanded=True):
                # Get all theme sets for this question
                theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
                
                if not theme_sets:
                    st.write("No theme sets available.")
                    continue
                
                # For each theme set, create a visualization
                for theme_set_id, theme_set in theme_sets.items():
                    theme_set_name = theme_set["name"]
                    themes = theme_set["themes"]
                    
                    if not themes:
                        continue
                    
                    theme_names = [t["name"] for t in themes]
                    
                    # Create a bar chart showing theme frequencies for this theme set
                    if (question in st.session_state["classification"]["results_by_question"] and 
                        theme_set_id in st.session_state["classification"]["results_by_question"][question]):
                        sparse_results = st.session_state["classification"]["results_by_question"][question][theme_set_id]
                        
                        # Create a unique state key for this theme set's filter selection
                        filter_cols_key = f"filter_cols_{question}_{theme_set_id}"
                        
                        # Get available columns for filters (excluding themes and question)
                        survey_df = st.session_state["data"]["survey"]
                        available_filter_cols = [col for col in survey_df.columns 
                                               if col not in theme_names 
                                               and col != question
                                               and col != st.session_state["data"].get("unique_id_column")]
                        
                        # Only keep categorical columns (fewer than 20 unique values)
                        categorical_cols = []
                        for col in available_filter_cols:
                            if len(survey_df[col].dropna().unique()) < 20:
                                categorical_cols.append(col)

                        # Only show filters checkbox if there are filters available
                        show_filters = False
                        if categorical_cols:
                            show_filters = st.checkbox(
                                f"Show filters", 
                                key=f"show_filters_{question}_{theme_set_id}"
                            )
                        
                        if show_filters:
                            selected_filter_cols = st.multiselect(
                                "Select columns for filtering:",
                                options=categorical_cols,
                                default=st.session_state.get(filter_cols_key, []),
                                key=f"filter_cols_select_{question}_{theme_set_id}"
                            )

                            st.session_state[filter_cols_key] = selected_filter_cols
                        else:
                            selected_filter_cols = []

                        # Create filter key prefix
                        filter_key_prefix = f"filter_summary_{question}_{theme_set_id}"
                        
                        # Create chart with conditional filtering
                        chart = create_theme_count_chart(
                            theme_names=theme_names,
                            title=f"Themes for {question} - {theme_set_name}",
                            filter_key_prefix=filter_key_prefix,
                            show_filters=show_filters,
                            filter_columns=selected_filter_cols,
                            sparse_results=sparse_results
                        )
                        
                        st.altair_chart(chart, use_container_width=True, key=f"chart_{question}_{theme_set_id}")
    
    with tab2:
        # Show all themes across all questions and theme sets
        all_themes_data = []
        
        for question in st.session_state["data"]["processed_questions"]:
            theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
            
            for theme_set_id, theme_set in theme_sets.items():
                theme_set_name = theme_set["name"]
                themes = theme_set["themes"]
                
                if not themes:
                    continue
                
                sparse_results = None
                if (question in st.session_state["classification"]["results_by_question"] and 
                    theme_set_id in st.session_state["classification"]["results_by_question"][question]):
                    sparse_results = st.session_state["classification"]["results_by_question"][question][theme_set_id]
                
                if sparse_results is not None:
                    theme_frequency = {}
                    for response_themes in sparse_results.values():
                        for theme_name in response_themes:
                            theme_frequency[theme_name] = theme_frequency.get(theme_name, 0) + 1

                    for theme in themes:
                        theme_name = theme["name"]
                        count = theme_frequency.get(theme_name, 0)
                        
                        theme_examples = []
                        if "examples" in theme and isinstance(theme["examples"], list):
                            theme_examples = theme["examples"]
                        elif "example" in theme and isinstance(theme["example"], dict) and "examples" in theme["example"]:
                            # Handle alternate structure where examples might be nested
                            theme_examples = theme["example"]["examples"]

                        # Format examples as a single string
                        examples_str = "\n".join([f"• {ex}" for ex in theme_examples]) if theme_examples else ""

                        all_themes_data.append({
                            "Question": question,
                            "Theme Set": theme_set_name,
                            "Theme": theme_name,
                            "Description": theme["description"],
                            "Examples": examples_str,
                            "Count": count
                        })
        
        if all_themes_data:
            all_themes_df = pd.DataFrame(all_themes_data)
            st.dataframe(all_themes_df, use_container_width=True)

def sparse_to_dense_dataframe(sparse_results, df, question_col, theme_names, unique_id_col=None):
    """
    Convert sparse results to dense DataFrame for display or export.
    
    Parameters:
        sparse_results: Dict mapping indices to list of theme names
        df: Original survey DataFrame
        question_col: Name of the question column
        theme_names: List of all theme names
        unique_id_col: Optional unique ID column name
        
    Returns:
        DataFrame with binary theme columns
    """
    # Start with required columns
    result_df = pd.DataFrame(index=df.index)
    
    if unique_id_col:
        result_df[unique_id_col] = df[unique_id_col]
    
    result_df[question_col] = df[question_col]
    
    # Initialize all theme columns to 0
    for theme_name in theme_names:
        result_df[theme_name] = 0
    
    # Fill in the 1s from sparse results
    for idx, themes in sparse_results.items():
        for theme in themes:
            if theme in theme_names:
                result_df.at[idx, theme] = 1
    
    return result_df

###############################################################################
# 9. SHOW THEMES IN SIDEBAR
###############################################################################

##### fragments #####
def _theme_editor_widget_key(theme_set_id: str | None, prefix: str, idx: int) -> str:
    """
    Create theme-editor widget keys that are scoped to a theme set, so switching
    theme sets/questions doesn't leak widget state across contexts.
    """
    ts = theme_set_id or "no_theme_set"
    return f"{prefix}_{ts}_{idx}"

def _normalize_theme_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())

def _parse_examples_text_to_list(text: str) -> list[str]:
    # Accept numbered or bullet lists; normalize whitespace.
    lines = [line.strip() for line in str(text or "").split("\n") if line.strip()]
    cleaned = [re.sub(r"^\s*(?:\d+\)|[-*•])\s*", "", line).strip() for line in lines]
    return [_normalize_theme_text(x) for x in cleaned if _normalize_theme_text(x)]

def is_theme_editor_dirty() -> bool:
    """
    Detect whether the visible theme editor widgets contain unsaved edits relative
    to `st.session_state['reasoning']['current_themes']` for the active theme set.
    """
    question = st.session_state.get("data", {}).get("active_question")
    if not question:
        return False

    current_themes = st.session_state.get("reasoning", {}).get("current_themes") or []
    if not current_themes:
        return False

    active_theme_set_id = st.session_state.get("reasoning", {}).get("active_theme_set_ids", {}).get(question)
    # Deletion marks are an explicit unsaved change.
    if st.session_state.get("themes_to_delete"):
        return True

    for i, theme in enumerate(current_themes):
        name_key = _theme_editor_widget_key(active_theme_set_id, "theme_name", i)
        desc_key = _theme_editor_widget_key(active_theme_set_id, "theme_description", i)
        ex_key = _theme_editor_widget_key(active_theme_set_id, "theme_examples", i)
        group_key = _theme_editor_widget_key(active_theme_set_id, "theme_group", i)

        if name_key in st.session_state:
            if _normalize_theme_text(st.session_state.get(name_key)) != _normalize_theme_text(theme.get("name", "")):
                return True
        if desc_key in st.session_state:
            if _normalize_theme_text(st.session_state.get(desc_key)) != _normalize_theme_text(theme.get("description", "")):
                return True
        if ex_key in st.session_state:
            widget_examples = _parse_examples_text_to_list(st.session_state.get(ex_key, ""))
            theme_examples = [_normalize_theme_text(x) for x in (theme.get("examples") or []) if _normalize_theme_text(x)]
            if widget_examples != theme_examples:
                return True
        if group_key in st.session_state:
            widget_group = st.session_state.get(group_key)
            # "No group" in widget corresponds to None on the theme
            widget_group_normalized = None if widget_group == "No group" else widget_group
            theme_group = theme.get("group")
            if widget_group_normalized != theme_group:
                return True

    return False

def request_theme_editor_sync():
    """Request that theme editor widget state be synced on next render."""
    st.session_state["theme_editor_sync_requested"] = True

def sync_theme_editor_widget_state(theme_set_id: str | None):
    """
    Force the theme editor widgets to reflect `reasoning.current_themes`.
    Call this only when edits have been programmatically applied (LLM edits,
    deduplication, etc.), not when the user has unsaved manual edits.
    """
    themes = st.session_state.get("reasoning", {}).get("current_themes") or []
    tsid = theme_set_id or "no_theme_set"
    for i, theme in enumerate(themes):
        st.session_state[_theme_editor_widget_key(tsid, "theme_name", i)] = theme.get("name", "")
        st.session_state[_theme_editor_widget_key(tsid, "theme_description", i)] = theme.get("description", "")
        examples = theme.get("examples") or []
        numbered_examples = [f"{j+1}) {ex}" for j, ex in enumerate(examples) if ex]
        st.session_state[_theme_editor_widget_key(tsid, "theme_examples", i)] = "\n".join(numbered_examples)
        # Sync group widget
        group_val = theme.get("group") or "No group"
        st.session_state[_theme_editor_widget_key(tsid, "theme_group", i)] = group_val

@st.fragment
def render_theme_editor(theme_set_id, theme_index, theme, theme_count_data, valid_groups=None):
    """Isolated fragment for editing a single theme."""
    i = theme_index
    percentage = theme_count_data.get("percentage", 0)
    
    base_title = theme["name"] if theme["name"].strip() else f"Theme {i+1}"
    is_marked_for_deletion = i in st.session_state.get("themes_to_delete", set())
    deletion_indicator = " ❌" if is_marked_for_deletion else ""
    # Show group label in expander title when groups are active
    group_label = ""
    if valid_groups and theme.get("group"):
        group_label = f" [{theme['group']}]"
    elif valid_groups and not theme.get("group"):
        group_label = " [No group]"
    expander_title = f"{base_title}{group_label} ({percentage:.0f}%){deletion_indicator}"
    
    with st.expander(expander_title):
        st.text_input(
            "Name",
            value=theme["name"],
            key=_theme_editor_widget_key(theme_set_id, "theme_name", i),
        )
        st.text_area(
            "Description",
            value=theme["description"],
            key=_theme_editor_widget_key(theme_set_id, "theme_description", i),
        )
        
        # Group selectbox (only shown when theme set has valid_groups)
        if valid_groups:
            group_options = ["No group"] + list(valid_groups)
            current_group = theme.get("group") or "No group"
            default_index = group_options.index(current_group) if current_group in group_options else 0
            st.selectbox(
                "Group",
                options=group_options,
                index=default_index,
                key=_theme_editor_widget_key(theme_set_id, "theme_group", i),
            )
        
        # Handle examples
        theme_examples = []
        if "examples" in theme and isinstance(theme["examples"], list):
            theme_examples = theme["examples"]
        
        numbered_examples = [f"{j+1}) {ex}" for j, ex in enumerate(theme_examples) if ex]
        examples_text = "\n".join(numbered_examples) if numbered_examples else ""
        
        st.text_area(
            "Examples",
            value=examples_text,
            key=_theme_editor_widget_key(theme_set_id, "theme_examples", i),
        )
        
        # Find another example button
        if st.button("Find another example", key=_theme_editor_widget_key(theme_set_id, "find_example", i), help="Uses the LLM to search the dataset for another response that matches this theme and adds it to the examples list."):
            find_and_add_example(i)
        
        # Delete button
        if st.button(f"Delete Theme {i+1}", key=_theme_editor_widget_key(theme_set_id, "delete_theme", i)):
            if "themes_to_delete" not in st.session_state:
                st.session_state["themes_to_delete"] = set()
            st.session_state["themes_to_delete"].add(i)
        
        # Display count
        st.write(f"Count: {theme_count_data['count']} ({theme_count_data['percentage']:.2f}%)")

def find_and_add_example(theme_index):
    """Helper to find and add example without triggering full rerun."""
    theme = st.session_state["reasoning"]["current_themes"][theme_index]
    df = st.session_state["data"]["survey"]
    question_col = st.session_state["data"]["active_question"]
    
    if "examples" in theme and isinstance(theme["examples"], list):
        current_examples = theme["examples"]
    else:
        current_examples = []
    
    # Sample some responses
    sample_size = st.session_state["sample_size"]
    # Use only non-missing, non-whitespace-only responses
    available_rows = df.index[df[question_col].fillna("").astype(str).str.strip() != ""].tolist()
    
    if available_rows:
        sample_indices = random.sample(available_rows, min(sample_size, len(available_rows)))
        sample_texts = df.loc[sample_indices, question_col].tolist()
        
        with st.spinner("Finding a new example..."):
            config = ReasoningConfig(st.session_state)
            new_example = find_example_for_theme(theme, sample_texts, current_examples, config, max_retries=5)
        
        if new_example and new_example not in current_examples:
            current_examples.append(new_example)
            st.session_state["reasoning"]["current_themes"][theme_index]["examples"] = current_examples
            # Keep the examples widget in sync for this theme set
            _q = st.session_state.get("data", {}).get("active_question")
            _tsid = None
            if _q:
                _tsid = st.session_state.get("reasoning", {}).get("active_theme_set_ids", {}).get(_q)
            _examples_key = _theme_editor_widget_key(_tsid, "theme_examples", theme_index)
            st.session_state[_examples_key] = "\n".join([f"{j+1}) {ex}" for j, ex in enumerate(current_examples) if ex])
            
            st.success(f"Added new example: \"{new_example}\"")
            st.rerun(scope="fragment")

def save_all_theme_changes():
    """Process all theme changes at once."""
    new_themes = []
    themes_to_delete = st.session_state.get("themes_to_delete", set())
    question = st.session_state.get("data", {}).get("active_question")
    active_theme_set_id = None
    if question:
        active_theme_set_id = st.session_state.get("reasoning", {}).get("active_theme_set_ids", {}).get(question)
    
    for i, theme in enumerate(st.session_state["reasoning"]["current_themes"]):
        if i in themes_to_delete:
            continue
            
        theme_name = st.session_state.get(
            _theme_editor_widget_key(active_theme_set_id, "theme_name", i),
            theme["name"],
        )
        theme_description = st.session_state.get(
            _theme_editor_widget_key(active_theme_set_id, "theme_description", i),
            theme["description"],
        )
        theme_examples_text = st.session_state.get(
            _theme_editor_widget_key(active_theme_set_id, "theme_examples", i),
            "",
        )
        
        # Read group from widget (if present), otherwise preserve existing
        group_key = _theme_editor_widget_key(active_theme_set_id, "theme_group", i)
        if group_key in st.session_state:
            widget_group = st.session_state.get(group_key)
            theme_group = None if widget_group == "No group" else widget_group
        else:
            theme_group = theme.get("group")
        
        # Parse examples
        example_lines = [line.strip() for line in theme_examples_text.split('\n') if line.strip()]
        example_lines = [re.sub(r'^\d+\)\s*', '', line) for line in example_lines]
        
        new_themes.append({
            "name": theme_name,
            "description": theme_description,
            "examples": example_lines,
            "group": theme_group,
        })
    
    # Update session state
    st.session_state["reasoning"]["current_themes"] = new_themes
    st.session_state["themes_to_delete"] = set()  # Clear deletion marks
    
    # Update theme set
    question = st.session_state["data"]["active_question"]
    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if active_theme_set_id:
        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
        if active_theme_set_id in theme_sets:
            theme_sets[active_theme_set_id]["themes"] = new_themes.copy()
    
    st.success("Themes updated!")
    st.rerun()

@st.fragment
def render_deletion_checkbox(item):
    """Isolated fragment for each deletion confirmation checkbox."""
    idx = item["index"]
    theme = item["theme"]
    reason = item["reason"]
    
    # Get default value based on flags
    default_value = True
    if st.session_state.get("select_all_deletions_flag", False):
        default_value = True
    elif st.session_state.get("deselect_all_deletions_flag", False):
        default_value = False
    
    col1, col2 = st.columns([1, 4])
    with col1:
        checkbox_key = f"delete_confirm_{idx}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = True  # Default to selected

        st.checkbox("Delete", key=checkbox_key)
    
    with col2:
        st.write(f"**{theme['name']}**")
        st.caption(f"Reason: {reason}")

@st.fragment
def render_deletion_controls(pending_deletions):
    """Fragment for select/deselect all controls."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Select All", key="select_all_deletions"):
            for item in pending_deletions:
                st.session_state[f"delete_confirm_{item['index']}"] = True
    
    with col2:
        if st.button("Deselect All", key="deselect_all_deletions"):
            for item in pending_deletions:
                st.session_state[f"delete_confirm_{item['index']}"] = False

##### end fragments #####

def show_editable_themes():
    """
    Displays the generated themes in the sidebar and allows users to edit, delete, or add new themes.
    Also provides buttons to save changes, revert changes, and reclassify responses.
    """
    with st.sidebar:
        # Get the active question and theme set
        question = st.session_state["data"]["active_question"]
        active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
        
        # Display the current theme set name in the header
        if question and active_theme_set_id:
            theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
            active_theme_set = theme_sets.get(active_theme_set_id, {})
            theme_set_name = active_theme_set.get("name", "")
            st.divider()
            st.subheader(f"Current Theme Set: {theme_set_name}")
        else:
            st.subheader("Current Themes")

        st.caption("Note: 'Reclassify Responses' will use the model settings for the classification stage in the main pane, all other actions will use the model settings from the 'Identify Themes' stage.")

        # If themes were programmatically updated (LLM edits/dedup/etc.), sync widget state.
        if st.session_state.pop("theme_editor_sync_requested", False):
            sync_theme_editor_widget_state(active_theme_set_id)

        # Acceptance guidance (keep control only in main pane)
        if question and active_theme_set_id and not is_current_theme_set_accepted():
            st.info("To proceed to classification, accept these themes in the main pane under Step 1 (Identify Themes).")

        # Unsaved edits indicator
        if is_theme_editor_dirty():
            st.warning("Unsaved theme edits. Save Changes or Revert Changes before switching questions or theme sets.")

        # Iterate through the current themes and display editable widgets.
        if st.session_state["classification"].get("sparse_results"):
            update_theme_counts()

        # Get valid_groups from the active theme set (for group selectbox in editor)
        _valid_groups = []
        if question and active_theme_set_id:
            _ts = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {}).get(active_theme_set_id, {})
            _valid_groups = _ts.get("valid_groups", [])

        for i, theme in enumerate(st.session_state["reasoning"]["current_themes"]):
            if i in st.session_state.get("themes_to_delete", set()):
                continue
            # Get count data
            count_data = st.session_state["reasoning"]["theme_counts"].get(
                theme["name"], {"count": 0, "percentage": 0}
            )
            # Render in isolated fragment (scope widget keys to theme set)
            render_theme_editor(active_theme_set_id, i, theme, count_data, _valid_groups)

        savebutton, revertchangesbutton, addthemebutton = st.columns(3)
        with savebutton:
            if st.button("Save Changes"):
                save_all_theme_changes()
                st.success("Themes updated")
                
        with revertchangesbutton:
            if st.button("Revert Changes"):
                st.session_state["reasoning"]["current_themes"] = st.session_state.get("backup_themes", st.session_state["reasoning"]["current_themes"]).copy()
                st.session_state["themes_to_delete"] = set()
                request_theme_editor_sync()
                st.toast("Reverted to previous themes")
                st.rerun()

        with addthemebutton:
            if st.button("Add Theme", key="add_theme_button"):
                st.session_state["reasoning"]["current_themes"].append({"name": "", "description": "", "examples": [], "group": None})
                st.rerun()
        
        with st.expander("Ask LLM for more themes"):
            # Default to active theme set's saved instructions, falling back to global reasoning instructions
            _q = st.session_state["data"]["active_question"]
            _tsid = st.session_state["reasoning"]["active_theme_set_ids"].get(_q)
            _ts_instructions = ""
            if _tsid:
                _ts_instructions = (
                    st.session_state["reasoning"]["theme_sets_by_question"].get(_q, {})
                    .get(_tsid, {})
                    .get("additional_instructions", "")
                )
            _default_instructions = _ts_instructions or st.session_state.get("additional_theme_generation_instructions", "")

            additional_instructions = st.text_area(
                "Additional instructions:",
                help="Single LLM call using reasoning model settings. Samples survey responses (same count as 'Sample size per batch' setting) to find additional themes. Provide any extra instructions for the model to consider during theme generation.",
                value=_default_instructions,
                key="get_more_additional_instructions"
            )
            if st.button("Ask LLM for more themes"):
                get_more_themes(additional_instructions)
                request_theme_editor_sync()
                st.rerun()

        with st.expander("Ask LLM to split/merge themes"):
            split_merge_instructions = st.text_area(
                "Instructions for splits/merges:",
                help="Single LLM call using reasoning model settings. Analyzes existing theme data only, no new response sampling. Provide instructions on how themes should be split or merged.",
                key="split_merge_instructions"
            )
            if st.button("Suggest Theme Splits/Merges"):
                # Save current themes as backup first
                st.session_state["backup_themes"] = st.session_state["reasoning"]["current_themes"].copy()
                suggest_theme_splits_merges(split_merge_instructions)
                request_theme_editor_sync()
                st.rerun()

        with st.expander("Ask LLM to edit themes"):
            edit_instructions = st.text_area(
                "Instructions for editing themes:",
                help="Edits themes in parallel batches using reasoning model settings. Batch size controls how many themes per call. No new response sampling. Provide instructions on how themes should be edited. ",
                key="theme_edit_instructions"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input(
                    "Batch size:",
                    min_value=1,
                    max_value=200,
                    value=20,
                    help="Number of themes to process in each parallel batch",
                    key="theme_edit_batch_size"
                )
            
            with col2:
                include_validated_examples = st.checkbox(
                    "Include validated examples only",
                    value=True,
                    help="If checked, examples will remain unchanged. If unchecked, LLM can edit examples (they may become illustrative rather than actual quotes).",
                    key="include_validated_examples"
                )
            
            if st.button("Edit Themes with LLM"):
                # Save current themes as backup first
                st.session_state["backup_themes"] = st.session_state["reasoning"]["current_themes"].copy()
                edit_all_themes(edit_instructions, batch_size, include_validated_examples)
                request_theme_editor_sync()
                st.rerun()
            
        # Handle deletion confirmations if there are pending deletions
        if "themes_pending_deletion" in st.session_state and st.session_state["themes_pending_deletion"]:
            st.divider()
            st.subheader("Confirm Theme Deletions")
            st.write("Note: themes edits are always applied. Use Revert Changes to revert.")

            # Initialize checkbox states if not present
            for item in st.session_state["themes_pending_deletion"]:
                checkbox_key = f"delete_confirm_{item['index']}"
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = True

            # Render controls in their own fragment
            render_deletion_controls(st.session_state["themes_pending_deletion"])

            # Render each checkbox in its own fragment
            for item in st.session_state["themes_pending_deletion"]:
                render_deletion_checkbox(item)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Deletions"):
                    # Create a set of indices that were actually marked for deletion by LLM
                    deletion_candidates = {item["index"] for item in st.session_state["themes_pending_deletion"]}
                    
                    edited_themes = st.session_state["edited_themes_result"]
                    final_themes = []
                    
                    for i, theme in enumerate(edited_themes):
                        # Only check deletion checkbox if this theme was marked for deletion by LLM
                        if i in deletion_candidates:
                            checkbox_key = f"delete_confirm_{i}"
                            should_delete = st.session_state.get(checkbox_key, True)
                            
                            if should_delete:
                                continue  # Skip this theme (delete it)
                        
                        # Keep this theme - remove deletion markers
                        clean_theme = theme.copy()
                        if "mark_for_deletion" in clean_theme:
                            del clean_theme["mark_for_deletion"]
                        if "deletion_reason" in clean_theme:
                            del clean_theme["deletion_reason"]
                            
                        final_themes.append(clean_theme)
                    
                    # Rest of the update logic remains the same...
                    st.session_state["reasoning"]["current_themes"] = final_themes
                    
                    # Update active theme set
                    question = st.session_state["data"]["active_question"]
                    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
                    
                    if active_theme_set_id:
                        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
                        if active_theme_set_id in theme_sets:
                            theme_sets[active_theme_set_id]["themes"] = final_themes.copy()
                    
                    # Clear temporary state and checkbox keys
                    del st.session_state["themes_pending_deletion"]
                    del st.session_state["edited_themes_result"]
                    
                    # Clear all deletion confirmation checkbox keys
                    keys_to_remove = [key for key in st.session_state.keys() if key.startswith("delete_confirm_")]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    
                    st.success(f"Theme edits applied. {len(edited_themes) - len(final_themes)} themes deleted.")
                    request_theme_editor_sync()
                    st.rerun()

            with col2:
                if st.button("Apply edits without deletions"):
                    # Apply edits WITHOUT deletions
                    edited_themes = st.session_state["edited_themes_result"]
                    final_themes = []
                    
                    for theme in edited_themes:
                        # Keep all themes - just remove deletion markers
                        clean_theme = theme.copy()
                        if "mark_for_deletion" in clean_theme:
                            del clean_theme["mark_for_deletion"]
                        if "deletion_reason" in clean_theme:
                            del clean_theme["deletion_reason"]
                        final_themes.append(clean_theme)
                    
                    # Update themes
                    st.session_state["reasoning"]["current_themes"] = final_themes
                    
                    # Update active theme set
                    question = st.session_state["data"]["active_question"]
                    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
                    
                    if active_theme_set_id:
                        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
                        if active_theme_set_id in theme_sets:
                            theme_sets[active_theme_set_id]["themes"] = final_themes.copy()
                    
                    # Clear temporary state
                    del st.session_state["themes_pending_deletion"]
                    del st.session_state["edited_themes_result"]
                    
                    # Clear all deletion confirmation checkbox keys
                    keys_to_remove = [key for key in st.session_state.keys() if key.startswith("delete_confirm_")]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    
                    st.success(f"Theme edits applied without deletions. All {len(final_themes)} themes kept.")
                    request_theme_editor_sync()
                    st.rerun()

        with st.expander("Deduplicate Similar Themes"):
            similarity_threshold = st.slider(
                "Name similarity threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=0.9,
                step=0.05,
                help="Slider determines which theme pairs get sent to the LLM for semantic comparison. Approx. 1.0 for exact matches only, 0.9 for typos or slight variations, 0.6 for word replacements."
            )
            show_debug = st.checkbox("Show deduplication details")
            
            custom_config = ClassificationConfig(st.session_state)
            
            if st.button("Deduplicate Themes"):
                # Save current themes as backup first
                st.session_state["backup_themes"] = st.session_state["reasoning"]["current_themes"].copy()
                
                with st.spinner("Deduplicating themes..."):
                    deduplicated, stats = deduplicate_themes(
                        st.session_state["reasoning"]["current_themes"],
                        similarity_threshold=similarity_threshold,
                        config=custom_config,
                        debug=show_debug
                    )
                    
                if deduplicated:
                    # Update themes
                    st.session_state["reasoning"]["current_themes"] = deduplicated
                    
                    # Update the active theme set
                    question = st.session_state["data"]["active_question"]
                    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
                    
                    if active_theme_set_id:
                        theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
                        if active_theme_set_id in theme_sets:
                            theme_sets[active_theme_set_id]["themes"] = deduplicated.copy()
                    request_theme_editor_sync()
                    st.rerun()
                
            if show_debug and "dedup_debug_info" in st.session_state:
                display_deduplication_debug_info()

        if st.button("Auto-Consolidate Similar Themes", help="Uses the LLM to identify and merge semantically similar themes. Recommended after parallel batch processing."):
            # Save current themes as backup first
            st.session_state["backup_themes"] = st.session_state["reasoning"]["current_themes"].copy()
            consolidate_themes_after_sampling()
            request_theme_editor_sync()
            st.rerun()

        if st.button("Reclassify Responses", help="Re-runs classification of all responses against the current themes. Use after editing, adding, or removing themes."):        
            st.session_state["stored_model_for_reclassify"] = st.session_state.get("classification_model_selected")
            st.session_state["stored_temperature_for_reclassify"] = st.session_state.get("classification_temperature")
            st.session_state["stored_batch_size_for_reclassify"] = st.session_state.get("batch_size", 3)
            st.session_state["reclassify_requested"] = True
            st.rerun()
        
def update_theme_counts():
    """
    Calculates and updates the count and percentage of responses for each theme.
    
    Processes the classification results dataframe to determine how many responses
    match each theme and what percentage of the total dataset they represent.
    Updates the theme_counts dictionary in session state, which is used for
    displaying statistics in the theme editing interface and visualizations.
    
    No parameters or return values.
    """
    sparse_results = st.session_state["classification"].get("sparse_results")
    
    if sparse_results is None:
        return
    
    total_responses = len(st.session_state["data"]["survey"])

    # Count occurrences of each theme
    theme_frequency = {}
    for response_themes in sparse_results.values():
        for theme_name in response_themes:
            theme_frequency[theme_name] = theme_frequency.get(theme_name, 0) + 1

    theme_counts = {}
    # Work with the current themes
    for theme in st.session_state["reasoning"]["current_themes"]:
        theme_name = theme["name"]
        count = theme_frequency.get(theme_name, 0)
        percentage = (count / total_responses) * 100 if total_responses > 0 else 0
        theme_counts[theme_name] = {"count": count, "percentage": percentage}
    
    st.session_state["reasoning"]["theme_counts"] = theme_counts

###############################################################################
# 10. DATA MANAGEMENT
###############################################################################
def save_current_question_results():
    """
    Saves the current question's themes and classification results to the final results.
    Prefixes theme column names with the question name and theme set name to avoid conflicts.
    """
    current_question = st.session_state["data"]["active_question"]
    
    if current_question is None:
        return False
    
    # Get all theme sets for this question
    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(current_question, {})
    
    if not theme_sets:
        st.error("No theme sets found for this question.")
        return False
    
    # Start with base dataframe containing unique ID column
    unique_id_col = st.session_state["data"]["unique_id_column"]
    if unique_id_col:
        if st.session_state["data"]["final_results"] is None:
            st.session_state["data"]["final_results"] = st.session_state["data"]["survey"][[unique_id_col]].copy()
    else:
        # Create a dataframe with just the index
        if st.session_state["data"]["final_results"] is None:
            st.session_state["data"]["final_results"] = pd.DataFrame(index=st.session_state["data"]["survey"].index)
            st.session_state["data"]["final_results"]["row_id"] = range(1, len(st.session_state["data"]["final_results"]) + 1)

    final_results = st.session_state["data"]["final_results"]
    final_index = final_results.index
    new_columns = {
        current_question: st.session_state["data"]["survey"][current_question].reindex(final_index)
    }
    
    # Process each theme set
    for theme_set_id, theme_set in theme_sets.items():
        theme_set_name = theme_set["name"]
        themes = theme_set["themes"]
        
        # Skip theme sets with no themes
        if not themes:
            continue
        
        # Get classification results for this theme set
        sparse_results = None
        if (current_question in st.session_state["classification"]["results_by_question"] and 
            theme_set_id in st.session_state["classification"]["results_by_question"][current_question]):
            sparse_results = st.session_state["classification"]["results_by_question"][current_question][theme_set_id]
        
        if sparse_results is not None:
            theme_names = [theme["name"] for theme in themes]
            theme_to_indices = {name: [] for name in theme_names}
            theme_name_set = set(theme_to_indices)

            for idx, response_themes in sparse_results.items():
                if idx not in final_index:
                    continue
                for theme_name in response_themes:
                    if theme_name in theme_name_set:
                        theme_to_indices[theme_name].append(idx)

            # Build theme columns with qualified names (question_themeset_theme)
            for theme in themes:
                theme_name = theme["name"]
                qualified_name = f"{current_question}_{theme_set_name}_{theme_name}"

                column_values = pd.Series(0, index=final_index)
                if theme_to_indices[theme_name]:
                    column_values.loc[theme_to_indices[theme_name]] = 1
                new_columns[qualified_name] = column_values

    # Avoid fragmented DataFrame by concatenating new columns once
    if new_columns:
        new_columns_df = pd.DataFrame(new_columns, index=final_index)
        existing_cols = list(final_results.columns)
        ordered_columns = []
        for col in existing_cols:
            if col in new_columns_df.columns:
                ordered_columns.append(new_columns_df[col])
            else:
                ordered_columns.append(final_results[col])
        for col in new_columns_df.columns:
            if col not in existing_cols:
                ordered_columns.append(new_columns_df[col])

        st.session_state["data"]["final_results"] = pd.concat(ordered_columns, axis=1)
    
    # Mark question as processed if not already
    if current_question not in st.session_state["data"]["processed_questions"]:
        st.session_state["data"]["processed_questions"].append(current_question)
    
    return True

def reset_for_new_question():
    """
    Resets state for analyzing a new question while preserving full survey data and processed questions.
    Clears all keys related to the currently active question and resets widget state.
    """
    # Clear the active question; processed_questions remains intact.
    st.session_state["data"]["active_question"] = None

    # Clear pointers for the active theme set.
    st.session_state["reasoning"]["active_theme_set_ids"] = {}  # clear current active theme set pointers
    st.session_state["reasoning"]["current_themes"] = []
    st.session_state["reasoning"]["theme_counts"] = {}

    # Clear classification state related to the active question.
    st.session_state["classification"]["sparse_results"] = {}
    st.session_state["classification"]["justifications"] = {}
    st.session_state["classification"]["df_indices"] = []

    # Clear UI action flags and temporary reclassification parameters.
    st.session_state["themes_to_delete"] = set()
    st.session_state["backup_themes"] = None
    st.session_state["reclassify_requested"] = False
    st.session_state["stored_model_for_reclassify"] = None
    st.session_state["stored_temperature_for_reclassify"] = None
    st.session_state["stored_batch_size_for_reclassify"] = None

    # Clear previous theme set selection tracking.
    if "previous_theme_set_selection" in st.session_state:
        del st.session_state["previous_theme_set_selection"]

    # Clear tracking flow flags
    st.session_state.pop("newly_created_theme_set", None)
    st.session_state.pop("just_created_theme_set", None)

    # Clear UI mode selections
    st.session_state.pop("grouping_variable", None)

    # Clear profiling columns
    st.session_state["data"]["profiling_columns"] = []

    # Reset question select dropdown
    st.session_state.pop("question_select", None)

    # Clear widget states to force fresh rendering.
    keys_to_remove = [key for key in st.session_state.keys() 
                      if (key.startswith("theme_set_selector_") or
                         key.startswith("theme_set_rename_input") or
                         key.startswith("filter_") or
                         key.startswith("show_filters_"))]
    for key in keys_to_remove:
        del st.session_state[key]

def save_theme_results():
    """
    Saves the current question's theme set results.
    """
    current_question = st.session_state["data"]["active_question"]

    # Always show a single "Save theme results" button.
    if st.button("Save theme results", help="Saves the current classification results for this question and theme set. Required before analysing a new question."):
        if save_current_question_results():
            st.success("Results saved for this theme set!")
    if st.button("Analyse new question", disabled=is_theme_editor_dirty()):
        st.session_state["reasoning"]["current_themes"] = []
        reset_for_new_question()
        st.rerun()

def run_yolo_mode():
    """
    Generates themes and immediately runs classification with default YOLO settings.
    """
    question = st.session_state["data"].get("active_question")
    if not question:
        st.error("No active question selected.")
        return

    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
    if not theme_sets:
        create_theme_set(question)

    with st.spinner("Generating themes (YOLO mode)..."):
        generate_themes()

    if not st.session_state["reasoning"]["current_themes"]:
        st.error("No themes were generated. YOLO mode stopped.")
        return

    active_theme_set_id = st.session_state["reasoning"]["active_theme_set_ids"].get(question)
    if not active_theme_set_id:
        st.error("No active theme set found after generation.")
        return

    theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(question, {})
    active_theme_set = theme_sets.get(active_theme_set_id, {})
    current_name = active_theme_set.get("name")
    theme_sets[active_theme_set_id]["name"] = current_name if current_name else "Default Theme Set"
    theme_sets[active_theme_set_id]["accepted"] = True

    st.session_state["classification_model_selected"] = st.session_state.get("reasoning_model_selected")
    st.session_state["classification_temperature"] = st.session_state.get("reasoning_temperature")

    reasoning_effort_choice = st.session_state.get("reasoning_reasoning_effort_select")
    reasoning_verbosity_choice = st.session_state.get("reasoning_verbosity_select")
    if reasoning_effort_choice is not None:
        st.session_state["classification_reasoning_effort_select"] = reasoning_effort_choice
    if reasoning_verbosity_choice is not None:
        st.session_state["classification_verbosity_select"] = reasoning_verbosity_choice

    st.session_state["batch_size"] = 15
    st.session_state["include_examples_in_classification"] = True

    if "classification_batches" in st.session_state:
        del st.session_state["classification_batches"]

    with st.spinner("Classifying responses into themes (YOLO mode)..."):
        cfg_local = ClassificationConfig(st.session_state)
        classify_verbatims_in_batches(cfg_local)

###############################################################################
# 11. MISC HELPER FUNCTIONS
###############################################################################

def is_main_thread():
    """
    Helper function to determine if current code is running in main thread.
    Use this to decide when it's safe to access session state or update UI.
    
    Returns:
        Boolean indicating if current execution is in main thread
    """
    return threading.current_thread() is threading.main_thread()

###############################################################################
# MAIN STREAMLIT APP
###############################################################################

def main():
    ### Main Streamlit App ###
    st.set_page_config(page_title="LLM Theme Categorization", layout="wide")

    # Initialize session
    initialise_session_states()

    # Setup sidebar (this is the first ui shown)
    # Includes:
    # - state load and save
    # - openai key (sets st.session_state["openai_client"])
    # - load_csv_file() (sets st.session_state["data"]["active_question"] and st.session_state["shuffled_indices"])
    # - optionally set unique id (st.session_state["data"]["unique_id_column"])
    # - shows already processed questions and sets st.session_state["reasoning"]["theme_sets_by_question"] based on st.session_state["data"]["processed_questions"]
    # - select_question_to_analyse() (sets st.session_state["data"]["active_question"] and st.session_state["initial_theme_generation_sample"])
    # - theme set management
    setup_sidebar()

    st.write("") # padding at top of main UI page to align with sidebar
    st.write("")

    # If data is loaded, show sample data and active question
    if st.session_state["data"]["survey"] is not None:
        show_sample_data()
        if not st.session_state["data"]["active_question"]:
            _render_info_card(
                '<p style="font-size:1.2em; margin:0;text-align:center;">Please select a column to analyse from the sidebar</p>'
            )

    # 1) REASONING STEP. If an active question, show the theme generation UI
    if st.session_state["data"]["active_question"]:
        st.subheader("1) Identify Themes")

        # Show the theme generation model selection
        # Includes UI to set:
        # - st.session_state["reasoning_model_selected"]
        # - st.session_state["reasoning_temperature"]
        # - st.session_state["batching_approach"]
        # - st.session_state["grouping_variable"]
        # - st.session_state["sample_size"]
        # - st.session_state["max_reasoning_batches"]
        # - st.session_state["reasoning_processing_mode"]
        # - st.session_state["additional_theme_generation_instructions"]
        # - st.session_state["enable_evaluate_and_improve_examples"]
        # - st.session_state["enable_consolidate_themes"]
        # (note that model_name, temperature, and additional_instructions sufficient for config object)
        show_reasoning_model_selection()

        if st.button("Generate Themes"):
            # Check if the active question has theme sets
            theme_sets = st.session_state["reasoning"]["theme_sets_by_question"].get(st.session_state["data"]["active_question"], {})
            # Create the theme set if it doesn't exist
            if not theme_sets:
                create_theme_set(st.session_state["data"]["active_question"])
            # get prompt
            with st.expander("See full model prompt:"):
                system_msg, user_msg = build_generate_themes_prompt(st.session_state["initial_theme_generation_sample"], config = ReasoningConfig(st.session_state))
            # generate themes (updates st.session_state["reasoning"]["current_themes"])
            with st.spinner('Generating themes...'):
                # sets st.session_state["reasoning"]["current_themes"], with process_serial_batches() or process_parallel_batches()
                # and then calls evaluate_and_improve_theme_examples() and consolidate_themes_after_sampling() if enabled
                generate_themes()

            st.success("Success! See the generated themes in the sidebar and then accept if you are ready to continue.")

        if st.button("YOLO mode"):
            run_yolo_mode()
        st.caption("YOLO mode identifies themes and then proceeds immediately to classification with default settings.")

        # Once we have themes, show them in the sidebar and if not yet accepted, show naming and acceptance in main ui
        if st.session_state["reasoning"]["current_themes"]:
            show_editable_themes()

            # If not yet accepted, show naming and acceptance button
            if not is_current_theme_set_accepted():
                handle_theme_naming_and_acceptance()
                st.divider()

    # 2) CLASSIFICATION STEP. If an active question and themes are accepted, show the classification UI
    if st.session_state["data"]["active_question"] and is_current_theme_set_accepted():
        st.subheader("2) Classify Responses")
        show_classification_model_selection()

        # If reclassification requested from the sidebar we show status in the main ui for consistency with initial theme classification
        if st.session_state.get("reclassify_requested", False):
            reclassify_themes()

    # If there are classification results, update theme counts in sidebar and show the results
    if st.session_state["classification"].get("sparse_results"):
        st.divider()
        
        update_theme_counts()  # Ensure theme counts in sidebar are up to date
        st.write("### Result of theme classification")
        st.write("_Select a row to see justifications_")
        
        show_classification_coverage() 
        show_classification_results_table()

        st.divider()
        st.write("### Responses Coded by Theme")
        
        show_themes_counts_chart()

        # 3) SAVE AND CONTINUE
        st.divider()
        st.subheader("3) Save theme results")
        
        save_theme_results()

    # Show export option if any questions have been processed
    if st.session_state["data"]["processed_questions"]:
        st.divider()
        st.subheader("Final Results")
        
        # Show summary of all themes across questions
        show_all_themes_summary()
        
        if st.button("Export Raw Data with Themes to CSV"):
            csv_data = export_to_csv()
            
            if csv_data is None:
                # Check what kind of helpful message to show
                processed_questions = st.session_state["data"].get("processed_questions", [])
                if not processed_questions:
                    st.error("No processed data available for export. Please analyze and classify some questions first.")
                else:
                    st.error("No saved results found. Please use 'Save theme results' for each question before exporting.")
            else:
                st.download_button(
                    label="Download Complete CSV",
                    data=csv_data,
                    file_name="all_classified_themes.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()