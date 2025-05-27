# CodeBaseAi.py
import streamlit as st
import os
import faiss
import numpy as np
import pickle
from google import generativeai as genai
import traceback
from typing import List, Dict # Added Dict
import tempfile # Added for temporary directory
import zipfile  # Added for ZIP file handling

# Attempt to import from reader.py
try:
    from reader import generate_markdown_for_file, should_skip_folder, should_skip_file, print_tree
except ImportError:
    st.error("Critical Error: Could not import functions from reader.py. Ensure reader.py is in the same directory as this script and that reader.py itself has no import errors.")
    # Provide dummy functions so the app can at least try to load UI elements.
    # Functionality will be severely impacted.
    def generate_markdown_for_file(file_path: str, output_dir: str = "markdown_files", save_content_to_file: bool = False) -> str:
        st.warning(f"Dummy generate_markdown_for_file called for {file_path}. Functionality impaired.")
        return ""
    def should_skip_folder(folder_name: str, skip_folders_list: List[str]) -> bool:
        st.warning(f"Dummy should_skip_folder called for {folder_name}. Functionality impaired.")
        return folder_name in skip_folders_list
    def should_skip_file(filename: str, skip_exts_list: List[str]) -> bool:
        st.warning(f"Dummy should_skip_file called for {filename}. Functionality impaired.")
        file_ext = os.path.splitext(filename)[1]
        return file_ext in skip_exts_list
    def print_tree(start_dir: str, skip_folders_list: List[str], skip_exts_list: List[str], prefix: str = "") -> str:
        st.warning(f"Dummy print_tree called for {start_dir} (skip_exts: {skip_exts_list}). Functionality impaired.")
        return "Error: print_tree function (from reader.py) not loaded."
    # st.stop() # You might want to conditionally stop or allow minimal UI to load

# --- Global Constants ---
FAISS_INDEX_FILE = "codebase_vector_db.faiss"
TEXT_CHUNKS_FILE = "codebase_text_chunks.pkl"
OVERVIEW_FILE_NAME = "codebase_overview.md" # Added for consistency
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768
LLM_MODEL_NAME = "gemini-1.5-flash"

# --- Core Logic Functions ---

def get_google_embedding(text_to_embed: str, task_type="RETRIEVAL_DOCUMENT"):
    if not text_to_embed.strip():
        print("Warning (get_google_embedding): Attempting to embed empty or whitespace-only text.")
        return [0.0] * EMBEDDING_DIMENSION
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text_to_embed,
            task_type=task_type,
        )
        return result['embedding']
    except Exception as e:
        print(f"Error (get_google_embedding) generating embedding: '{text_to_embed[:100].replace('\n', ' ')}...': {e}")
        # Optionally, you could re-raise or return a specific error indicator
        return None

def build_and_save_faiss_db(
    markdown_file_items: list[str],
    faiss_db_path: str, # Made required
    text_meta_path: str, # Made required
    status_callback=None
):
    log_msg = f"Building FAISS vector database from {len(markdown_file_items)} documents..."
    if status_callback: status_callback(log_msg, "info")
    else: print(log_msg)

    all_db_text_chunks = []
    all_db_embeddings = []

    for item_idx, md_content_item in enumerate(markdown_file_items):
        if not md_content_item or "```" not in md_content_item : # Basic check for code block presence
            msg = f"Skipping item {item_idx} due to malformed structure or no code block: '{md_content_item[:100].replace('\n',' ')}...'"
            if status_callback: status_callback(msg, "warning")
            else: print(msg)
            continue
        
        all_db_text_chunks.append(md_content_item)
        embedding = get_google_embedding(md_content_item, task_type="RETRIEVAL_DOCUMENT")
        
        if embedding and len(embedding) == EMBEDDING_DIMENSION:
            all_db_embeddings.append(embedding)
        else:
            all_db_text_chunks.pop() # Remove corresponding text chunk if embedding failed
            err_msg = f"Failed to embed or got wrong dimension for item {item_idx}. Item skipped. Embedding: {embedding}"
            if status_callback: status_callback(err_msg, "warning")
            else: print(err_msg)

    if not all_db_embeddings or not all_db_text_chunks: # Ensure both lists are non-empty
        msg = "No valid data to build FAISS DB. All items failed embedding or were invalid."
        if status_callback: status_callback(msg, "error")
        else: print(msg)
        return False

    embeddings_np = np.array(all_db_embeddings).astype('float32')
    if embeddings_np.ndim != 2 or embeddings_np.shape[1] != EMBEDDING_DIMENSION:
        msg = f"Embeddings array error. Expected shape (N, {EMBEDDING_DIMENSION}), got {embeddings_np.shape}. FAISS DB not built."
        if status_callback: status_callback(msg, "error")
        else: print(msg)
        return False

    try:
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        index.add(embeddings_np)
        faiss.write_index(index, faiss_db_path)
        msg_idx = f"FAISS index ({index.ntotal} vectors) saved to {faiss_db_path}"
        if status_callback: status_callback(msg_idx, "success")
        else: print(msg_idx)
        
        with open(text_meta_path, 'wb') as f:
            pickle.dump(all_db_text_chunks, f)
        msg_meta = f"Text chunks ({len(all_db_text_chunks)}) saved to {text_meta_path}"
        if status_callback: status_callback(msg_meta, "success")
        else: print(msg_meta)
        return True
    except Exception as e:
        msg = f"Error saving FAISS DB/metadata: {e}"
        if status_callback: status_callback(msg, "error")
        else: print(msg)
        traceback.print_exc()
        return False

def load_faiss_db_and_metadata(faiss_db_path: str, text_meta_path: str, status_callback=None):
    # ... (rest of the function remains the same)
    if status_callback:
        status_callback(f"Loading FAISS index from: {faiss_db_path}", "info")
        status_callback(f"Loading text chunks metadata from: {text_meta_path}", "info")
    else:
        print(f"\nLoading FAISS index from: {faiss_db_path}")
        print(f"Loading text chunks metadata from: {text_meta_path}")
    try:
        if not os.path.exists(faiss_db_path):
            msg = f"Error: FAISS index file not found: {faiss_db_path}"
            if status_callback: status_callback(msg, "error")
            else: print(msg)
            return None, None
        if not os.path.exists(text_meta_path):
            msg = f"Error: Text chunks metadata file not found: {text_meta_path}"
            if status_callback: status_callback(msg, "error")
            else: print(msg)
            return None, None
        
        index = faiss.read_index(faiss_db_path)
        with open(text_meta_path, 'rb') as f:
            text_chunks = pickle.load(f)
        
        msg = f"Loaded FAISS index with {index.ntotal} vectors and {len(text_chunks)} text chunks."
        if status_callback: status_callback(msg, "success")
        else: print(msg)
        
        if index.ntotal == 0:
            msg_warn = "Warning: Loaded FAISS index is empty."
            if status_callback: status_callback(msg_warn, "warning")
            else: print(msg_warn)
        return index, text_chunks
    except Exception as e:
        msg = f"Error loading FAISS DB or metadata: {e}"
        if status_callback: status_callback(msg, "error")
        else: print(msg)
        traceback.print_exc()
        return None, None

def perform_similarity_search(query_text: str, faiss_index, text_chunks_list: list, top_k: int = 3) -> list[str]:
    # ... (rest of the function remains the same)
    if faiss_index is None or text_chunks_list is None or faiss_index.ntotal == 0:
        print("Search cannot be performed: FAISS index or text chunks not loaded or index is empty.")
        return []
    
    query_embedding = get_google_embedding(query_text, task_type="RETRIEVAL_QUERY")
    if query_embedding is None or len(query_embedding) != EMBEDDING_DIMENSION:
        print("Failed to get valid query embedding during search.")
        return []
    
    query_vector_np = np.array([query_embedding]).astype('float32')
    retrieved_texts = []
    try:
        actual_k = min(top_k, faiss_index.ntotal)
        if actual_k == 0: return []
        
        distances, indices = faiss_index.search(query_vector_np, k=actual_k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    if indices.size == 0 or (indices.ndim > 1 and len(indices[0]) > 0 and indices[0][0] == -1): # Check for -1 index
        print("No relevant documents found or invalid indices returned.")
        return []
        
    for i in range(indices.shape[1]): # Iterate through columns of indices
        idx = indices[0][i]
        if idx == -1: continue # Skip if index is -1 (no document found for this position)
        if 0 <= idx < len(text_chunks_list):
            retrieved_texts.append(text_chunks_list[idx])
        else:
            print(f"Warning: Index {idx} from FAISS search is out of bounds for text_chunks_list (size {len(text_chunks_list)}).")
    return retrieved_texts

def get_llm_codebase_overview(directory_tree: str, aggregated_markdown_content: str, client, model_name: str) -> str:
    # ... (rest of the function remains the same)
    LLM_FAILURE_MESSAGES = [
        "Failed to generate codebase overview from LLM.",
        "Cannot generate overview: Directory tree and markdown content are empty.",
    ]
    if not directory_tree and not aggregated_markdown_content:
        return LLM_FAILURE_MESSAGES[1]
    
    effective_markdown_content = aggregated_markdown_content if aggregated_markdown_content else 'No code snippets were processed or available for aggregation.'
    
    prompt = (
        "You are an expert code analyst. Based on the following directory structure and aggregated code snippets "
        "from various files, please provide a comprehensive high-level overview of the entire codebase. "
        "Describe the project's overall purpose, its main components or modules, how different parts might generally interact "
        "(based on the structure and any specific code hints), and any key architectural patterns, languages, or technologies you can infer. "
        "Focus on the overall structure and functionality, and the relationships between components.\n\n"
        f"Directory Structure:\n```\n{directory_tree if directory_tree else 'Not available.'}\n```\n\n"
        f"Aggregated Code Snippets (from various files):\n{effective_markdown_content}"
    )
    try:
        if client is None:
            print("Error (get_llm_codebase_overview): LLM client is not initialized.")
            return "LLM client not initialized. Cannot generate overview."
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling LLM for overview: {e}")
        traceback.print_exc()
        return LLM_FAILURE_MESSAGES[0]

def get_llm_answer_with_context(
    user_query: str,
    context_chunks: List[str],
    chat_history: List[Dict[str, str]],
    client: genai.GenerativeModel,
    model_name: str
) -> str:
    # ... (rest of the function remains the same)
    if client is None:
        print("Error (get_llm_answer_with_context): LLM client is not initialized.")
        return "LLM client not initialized. Cannot generate answer."

    context_str = "\n\n---\n\n".join(context_chunks) if context_chunks else "No specific codebase context was retrieved for this query."
    
    MAX_CONTEXT_CHAR_LEN = 15000
    if len(context_str) > MAX_CONTEXT_CHAR_LEN:
        print(f"Warning: Truncating FAISS context from {len(context_str)} to {MAX_CONTEXT_CHAR_LEN} characters.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LEN] + "\n... (FAISS context truncated)"

    current_turn_user_content = (
        f"User Query: \"{user_query}\"\n\n"
        f"If the query is about the codebase, use the following retrieved context. Otherwise, rely on our conversation history.\n"
        f"Retrieved Codebase Context:\n\"\"\"\n{context_str}\n\"\"\"\n\n"
        f"Please answer the user's query based on our conversation history and the provided codebase context (if relevant to the query)."
    )
    
    api_messages = []
    if len(chat_history) > 1: 
        for message in chat_history[:-1]: 
            role = "user" if message["role"] == "user" else "model"
            api_messages.append({'role': role, 'parts': [message["content"]]})
    
    api_messages.append({'role': 'user', 'parts': [current_turn_user_content]})
    
    MAX_API_HISTORY_ITEMS = 11 
    if len(api_messages) > MAX_API_HISTORY_ITEMS:
        api_messages = api_messages[-MAX_API_HISTORY_ITEMS:]
        print(f"Truncated API chat history to the last {len(api_messages)} items.")

    try:
        response = client.generate_content(api_messages)
        return response.text
    except Exception as e:
        print(f"Error calling LLM for Q&A with history: {e}")
        traceback.print_exc()
        if "API key not valid" in str(e):
            return "Error: The Gemini API key is not valid. Please check your configuration."
        if "400" in str(e) and ("userLocation" in str(e) or "blocked" in str(e).lower()):
             return "Error: API request blocked, possibly due to safety settings, regional restrictions, or an issue with the query/context provided."
        return "Sorry, I encountered an error while trying to generate an answer. The context or history might have been too long, or an API issue occurred."

# --- Streamlit App UI and Logic ---
st.set_page_config(page_title="Codebase AI Analyst", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 Codebase AI Analyst")
st.caption(f"Analyze codebases, generate overviews, and ask questions using Gemini ({LLM_MODEL_NAME} & {EMBEDDING_MODEL_NAME}).")

# Initialize session state variables
default_values = {
    "api_key": os.getenv("GOOGLE_API_KEY", ""),
    "genai_initialized": False,
    "llm_client": None,
    "faiss_index": None,
    "text_chunks": None,
    "processed_target_dir": None, # This will be the path to the extracted codebase in temp dir
    "codebase_overview": "",
    "directory_tree": "",
    "messages": [],
    "status_log": []
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

def log_status(message, level="info"):
    # ... (rest of the function remains the same)
    print(f"UI_LOG ({level.upper()}): {message}")
    log_entry = f"[{level.upper()}] {message}"
    # Avoid duplicate consecutive messages
    if not st.session_state.status_log or st.session_state.status_log[-1] != log_entry:
        st.session_state.status_log.append(log_entry)
    MAX_LOG_ENTRIES = 100 # Limit log size
    if len(st.session_state.status_log) > MAX_LOG_ENTRIES:
        st.session_state.status_log = st.session_state.status_log[-MAX_LOG_ENTRIES:]
    
    # Show toast notifications
    if level == "success": st.toast(message, icon="✅")
    elif level == "error": st.toast(message, icon="❌")
    elif level == "warning": st.toast(message, icon="⚠️")


with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get your API key from Google AI Studio.",
        key="api_key_input_widget"
    )
    if api_key_input != st.session_state.api_key: # If key changed in input field
        st.session_state.api_key = api_key_input
        st.session_state.genai_initialized = False # Require re-initialization
        st.session_state.llm_client = None

    if st.button("Apply API Key & Initialize", use_container_width=True, type="primary", key="apply_api_key_button"):
        if st.session_state.api_key:
            try:
                with st.spinner("Initializing Gemini..."):
                    genai.configure(api_key=st.session_state.api_key)
                    st.session_state.llm_client = genai.GenerativeModel(LLM_MODEL_NAME) # Initialize the model
                    st.session_state.genai_initialized = True
                log_status("Gemini API Key applied and GenAI initialized successfully!", "success")
            except Exception as e:
                st.session_state.genai_initialized = False
                st.session_state.llm_client = None
                log_status(f"Failed to initialize Gemini: {str(e)}", "error")
                st.error(f"API Key Initialization Failed: {str(e)}")
        else:
            log_status("Please enter an API Key.", "warning")

    if st.session_state.genai_initialized:
        st.sidebar.success(f"Gemini Initialized ({LLM_MODEL_NAME})")
    else:
        st.sidebar.warning("Gemini not initialized. Enter API Key.")
    st.divider()

    st.subheader("1. Process New Codebase")
    
    # --- MODIFIED SECTION for File Upload ---
    uploaded_codebase_zip = st.file_uploader(
        "Upload Codebase (ZIP file)",
        type=["zip"],
        key="codebase_zip_uploader",
        help="Upload a ZIP file containing the codebase to analyze."
    )
    # --- END MODIFIED SECTION ---

    proc_skip_folders = st.text_input("Folders to Skip (comma-separated)", value=".git,venv,node_modules,__pycache__,build,dist,target,docs,.DS_Store,.vscode,.idea", key="proc_skip_folders_input")
    proc_skip_exts = st.text_input("Extensions to Skip (comma-separated, with dot)", value=".log,.tmp,.bak,.lock,.env,.gz,.zip,.rar,.7z,.tar,.class,.obj,.o,.pyc,.pdb,.exe,.dll,.so,.dylib,.DS_Store,.ipynb,.md,.txt,.json,.xml,.yaml,.yml,.csv,.tsv,.gif,.png,.jpg,.jpeg,.svg,.ico,.webmanifest,.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx", key="proc_skip_exts_input")

    if st.button("Analyze & Build Database", disabled=not st.session_state.genai_initialized, use_container_width=True, key="analyze_db_button"):
        if not uploaded_codebase_zip: # Check if a file was uploaded
            log_status("Please upload a codebase ZIP file.", "warning")
        else:
            # Use a temporary directory that will be cleaned up automatically
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    st.session_state.status_log = [] # Clear previous logs for this run
                    current_task_status_placeholder = st.empty()
                    current_task_status_placeholder.info(f"Processing '{uploaded_codebase_zip.name}'...")

                    # Save and extract the zip file
                    path_to_zip_file = os.path.join(temp_dir, uploaded_codebase_zip.name)
                    with open(path_to_zip_file, "wb") as f:
                        f.write(uploaded_codebase_zip.getbuffer())
                    log_status(f"Uploaded '{uploaded_codebase_zip.name}' to temporary location.", "info")

                    extracted_codebase_path = os.path.join(temp_dir, "extracted_codebase")
                    os.makedirs(extracted_codebase_path, exist_ok=True)
                    
                    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extracted_codebase_path)
                    log_status(f"Codebase extracted to: {extracted_codebase_path}", "info")
                    
                    # This extracted_codebase_path is now our target directory for analysis
                    proc_target_dir_for_analysis = extracted_codebase_path
                    
                    current_task_status_placeholder.info("Initializing processing of extracted codebase...")
                    _skip_folders_list = [f.strip() for f in proc_skip_folders.split(',') if f.strip()]
                    _skip_exts_list = [e.strip() for e in proc_skip_exts.split(',') if e.strip()]

                    current_task_status_placeholder.info("Generating directory tree...")
                    tree_str = print_tree(proc_target_dir_for_analysis, _skip_folders_list, _skip_exts_list)
                    st.session_state.directory_tree = tree_str
                    if "Error: print_tree function not loaded" in tree_str: # Check against dummy output
                        log_status("Directory tree generation failed (dummy function used or 'reader.py' error).", "error")
                    else:
                        log_status("Directory tree generated.", "info")
                    
                    current_task_status_placeholder.info("Scanning files and generating markdown content...")
                    _all_markdown_file_items = []
                    _processed_files_count = 0
                    _skipped_files_count = 0
                    for dirpath, dirnames, filenames in os.walk(proc_target_dir_for_analysis, topdown=True):
                        dirnames[:] = [d for d in dirnames if not should_skip_folder(d, _skip_folders_list)]
                        for filename in filenames:
                            if should_skip_file(filename, _skip_exts_list):
                                _skipped_files_count +=1
                                continue
                            full_file_path = os.path.join(dirpath, filename)
                            markdown_code_block = generate_markdown_for_file(full_file_path, save_content_to_file=False) 
                            if markdown_code_block and markdown_code_block.strip() and markdown_code_block.strip() != "``````" and markdown_code_block.strip() != "```\n```": # Basic checks
                                relative_file_path = os.path.relpath(full_file_path, proc_target_dir_for_analysis)
                                full_item_for_list = f"--- File: {relative_file_path} ---\n{markdown_code_block.strip()}"
                                _all_markdown_file_items.append(full_item_for_list)
                                _processed_files_count += 1
                            else:
                                _skipped_files_count +=1 
                    log_status(f"File scan complete. Processed: {_processed_files_count}, Skipped/Empty: {_skipped_files_count}.", "info")

                    if _processed_files_count > 0 or (tree_str and "Error" not in tree_str):
                        current_task_status_placeholder.info("Requesting LLM for codebase overview...")
                        aggregated_markdown_for_overview = "\n\n".join(_all_markdown_file_items) if _all_markdown_file_items else ""
                        overview = get_llm_codebase_overview(st.session_state.directory_tree, aggregated_markdown_for_overview, st.session_state.llm_client, LLM_MODEL_NAME)
                        st.session_state.codebase_overview = overview
                        log_status("Codebase overview generated by LLM.", "info")
                        
                        # Save overview within the temporary extracted codebase directory
                        overview_file_path = os.path.join(proc_target_dir_for_analysis, OVERVIEW_FILE_NAME)
                        try:
                            with open(overview_file_path, 'w', encoding='utf-8') as ov_file:
                                ov_file.write(f"# Codebase Overview for uploaded: {uploaded_codebase_zip.name}\n") # Use zip name
                                ov_file.write(f"## Processed Path (temporary): {os.path.abspath(proc_target_dir_for_analysis)}\n")
                                ov_file.write(f"## Generated by: {LLM_MODEL_NAME}\n\n")
                                if st.session_state.directory_tree: 
                                    ov_file.write(f"## Directory Structure\n```\n{st.session_state.directory_tree}\n```\n\n")
                                ov_file.write(f"## AI Generated Overview\n{st.session_state.codebase_overview}\n")
                            log_status(f"Codebase overview saved to temporary location: {overview_file_path}", "info")
                        except Exception as e_save:
                            log_status(f"Warning: Error saving overview file: {str(e_save)}", "warning")
                    else:
                        st.session_state.codebase_overview = "Not enough data (no processed files or valid tree) to generate overview."
                        log_status("Skipped LLM overview generation due to lack of data.", "warning")

                    if _all_markdown_file_items:
                        current_task_status_placeholder.info("Building FAISS vector database...")
                        # Save FAISS DB files within the temporary extracted codebase directory
                        _faiss_db_file_path = os.path.join(proc_target_dir_for_analysis, FAISS_INDEX_FILE)
                        _text_chunks_meta_file_path = os.path.join(proc_target_dir_for_analysis, TEXT_CHUNKS_FILE)
                        
                        build_success = build_and_save_faiss_db(_all_markdown_file_items, _faiss_db_file_path, _text_chunks_meta_file_path, status_callback=log_status)
                        if build_success:
                            current_task_status_placeholder.info("Loading newly built database...")
                            index, chunks = load_faiss_db_and_metadata(_faiss_db_file_path, _text_chunks_meta_file_path, status_callback=log_status)
                            if index is not None and chunks is not None:
                                st.session_state.faiss_index = index
                                st.session_state.text_chunks = chunks
                                st.session_state.processed_target_dir = proc_target_dir_for_analysis # Critical: update session state
                                log_status(f"Successfully built and loaded database for '{uploaded_codebase_zip.name}'.", "success")
                            else:
                                log_status("Database built, but failed to auto-load it. Try loading manually (if files were persisted).", "error")
                        else:
                            log_status("Failed to build FAISS database. Check logs for details.", "error")
                    else:
                        log_status("No markdown content generated from files, skipping FAISS DB creation.", "warning")
                        st.session_state.faiss_index = None # Ensure clean state
                        st.session_state.text_chunks = None
                        st.session_state.processed_target_dir = proc_target_dir_for_analysis # Still set to show context if overview was generated
                    
                    current_task_status_placeholder.empty()
                    st.success(f"Processing complete for '{uploaded_codebase_zip.name}'!")

                except zipfile.BadZipFile:
                    log_status(f"Error: Uploaded file '{uploaded_codebase_zip.name}' is not a valid ZIP file or is corrupted.", "error")
                    st.error(f"Uploaded file '{uploaded_codebase_zip.name}' is not a valid ZIP file or is corrupted.")
                    current_task_status_placeholder.empty()
                except Exception as e:
                    current_task_status_placeholder.empty()
                    log_status(f"An unexpected error occurred during processing: {str(e)}", "error")
                    st.error(f"Processing Failed: {str(e)}")
                    traceback.print_exc()
            # Temporary directory 'temp_dir' and its contents are automatically cleaned up here.
            # The FAISS index and chunks are in st.session_state (in memory).
            st.rerun() # Rerun to update UI based on new session state

    st.divider()
    st.subheader("2. Load Existing Vector DB")
    st.info("To load an existing DB, ensure DB files (.faiss, .pkl, overview.md) are part of your app's repository or upload them separately (feature not yet implemented here). Then provide the relative path to the directory containing these files on the server.")
    db_dir_load = st.text_input("Directory of Existing DB (server path)", key="db_dir_load_input", help=f"E.g., 'data/my_db' if DB files are in a 'data/my_db' subfolder of your app repo.")

    if st.button("Load Vector Database", disabled=not st.session_state.genai_initialized, use_container_width=True, key="load_db_button"):
        if not db_dir_load:
            log_status("Please provide the directory of the database (relative server path).", "warning")
        elif not os.path.isdir(db_dir_load): # This checks path on the server
            log_status(f"Database directory not found on server: {db_dir_load}. Ensure it's a valid relative path to files in your app repository.", "error")
        else:
            _faiss_db_file_path = os.path.join(db_dir_load, FAISS_INDEX_FILE)
            _text_chunks_meta_file_path = os.path.join(db_dir_load, TEXT_CHUNKS_FILE)
            overview_file_path_load = os.path.join(db_dir_load, OVERVIEW_FILE_NAME)

            if not os.path.exists(_faiss_db_file_path) or not os.path.exists(_text_chunks_meta_file_path):
                log_status(f"DB files ('{FAISS_INDEX_FILE}', '{TEXT_CHUNKS_FILE}') not found in '{db_dir_load}'.", "error")
            else:
                with st.spinner(f"Loading database from '{db_dir_load}'..."):
                    index, chunks = load_faiss_db_and_metadata(_faiss_db_file_path, _text_chunks_meta_file_path, status_callback=log_status)
                    if index is not None and chunks is not None:
                        st.session_state.faiss_index = index
                        st.session_state.text_chunks = chunks
                        st.session_state.processed_target_dir = db_dir_load # Update context to this loaded DB path
                        
                        # Load overview and tree if overview file exists
                        tree_from_overview = ""
                        overview_content = ""
                        if os.path.exists(overview_file_path_load):
                            try:
                                with open(overview_file_path_load, 'r', encoding='utf-8') as f:
                                    full_overview_doc = f.read()
                                # Extract tree
                                tree_start_marker = "## Directory Structure\n```\n"
                                tree_end_marker = "\n```\n\n## AI Generated Overview\n"
                                if tree_start_marker in full_overview_doc and tree_end_marker in full_overview_doc:
                                    start_tree = full_overview_doc.find(tree_start_marker) + len(tree_start_marker)
                                    end_tree = full_overview_doc.find(tree_end_marker, start_tree)
                                    if end_tree != -1: tree_from_overview = full_overview_doc[start_tree:end_tree]
                                # Extract overview
                                overview_start_marker = "## AI Generated Overview\n"
                                if overview_start_marker in full_overview_doc:
                                    start_overview = full_overview_doc.find(overview_start_marker) + len(overview_start_marker)
                                    overview_content = full_overview_doc[start_overview:]
                                
                                st.session_state.directory_tree = tree_from_overview.strip() if tree_from_overview else "Directory tree not found in overview file."
                                st.session_state.codebase_overview = overview_content.strip() if overview_content else "Overview not found in file."
                                log_status("Overview and tree loaded from existing file.", "info")
                            except Exception as e_read_ov:
                                log_status(f"Could not read overview file '{overview_file_path_load}': {e_read_ov}", "warning")
                                st.session_state.codebase_overview = "Could not load overview file."
                                st.session_state.directory_tree = "Could not load tree from overview file."
                        else:
                            st.session_state.codebase_overview = f"Overview file ({OVERVIEW_FILE_NAME}) not found in DB directory '{db_dir_load}'."
                            st.session_state.directory_tree = "Tree not available (overview file missing)."
                        log_status(f"Successfully loaded database from '{db_dir_load}'.", "success")
                    else:
                        log_status("Failed to load vector database. Check logs.", "error")
                st.rerun()

# Main page content
if st.session_state.processed_target_dir:
    context_name = os.path.basename(st.session_state.processed_target_dir)
    if st.session_state.processed_target_dir.endswith("extracted_codebase") and uploaded_codebase_zip is not None: # Check if it was from an upload
         context_name = f"uploaded '{uploaded_codebase_zip.name}' (processed in temp)" # More descriptive
    elif st.session_state.processed_target_dir.endswith("extracted_codebase"): # Fallback if widget state is lost but path indicates extraction
         context_name = "uploaded codebase (processed in temp)"

    st.success(f"**Active Codebase Context:** `{context_name}` ({st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0} vectors)")
else:
    st.info("No codebase processed or database loaded. Use sidebar to Analyze a new codebase (via ZIP upload) or Load an existing database (from app repository).")

if st.session_state.directory_tree or st.session_state.codebase_overview:
    tab_overview, tab_tree = st.tabs(["✨ Codebase Overview", "🌳 Directory Tree"])
    with tab_overview:
        if st.session_state.codebase_overview:
            st.markdown("### AI Generated Codebase Overview")
            st.markdown(st.session_state.codebase_overview)
        else:
            st.caption("Codebase overview not available or not generated yet.")
    with tab_tree:
        if st.session_state.directory_tree:
            st.markdown("### Directory Tree")
            st.code(st.session_state.directory_tree, language="") # Removed language='markdown' for plain tree
        else:
            st.caption("Directory tree not available or not generated yet.")
else:
    if st.session_state.processed_target_dir: # If a dir was processed but somehow no overview/tree
        st.caption("Directory tree and codebase overview are not available for the current context, though a directory was processed.")

st.divider()

st.header("💬 Ask Questions About the Codebase")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
q_disabled = not (st.session_state.genai_initialized and st.session_state.faiss_index is not None and st.session_state.faiss_index.ntotal > 0)
q_disabled_reason_text = ""
if not st.session_state.genai_initialized:
    q_disabled_reason_text = " (Initialize API Key in sidebar)"
elif st.session_state.faiss_index is None or st.session_state.faiss_index.ntotal == 0: # Check ntotal explicitly
    q_disabled_reason_text = " (Process a codebase or Load a non-empty DB in sidebar)"

if q_disabled_reason_text:
    st.caption(f"Question input disabled{q_disabled_reason_text}.")

if prompt := st.chat_input("Your question...", disabled=q_disabled, key="chat_input_user_prompt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching context and generating answer..."):
            log_status(f"User query: {prompt}", "info") # Log the query
            # Ensure index and chunks are available (redundant if q_disabled works, but good practice)
            if st.session_state.faiss_index and st.session_state.text_chunks:
                retrieved_texts = perform_similarity_search(prompt, st.session_state.faiss_index, st.session_state.text_chunks, top_k=5) # Increased top_k for more context
                
                if not retrieved_texts:
                    log_status("No relevant context found in the database for the query.", "info")
                else:
                    log_status(f"Retrieved {len(retrieved_texts)} context chunks for the query.", "info")

                full_response = get_llm_answer_with_context(
                    prompt, 
                    retrieved_texts, 
                    st.session_state.messages, # Pass full history
                    st.session_state.llm_client, 
                    LLM_MODEL_NAME
                )
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                log_status("LLM response generated and displayed.", "info")
            else:
                # This case should ideally be prevented by q_disabled
                err_msg = "Cannot process query: FAISS index or text chunks are not loaded."
                message_placeholder.error(err_msg)
                log_status(err_msg, "error")
                st.session_state.messages.append({"role": "assistant", "content": err_msg}) # Add error to history
    st.rerun() # Rerun to ensure chat input is cleared/updated properly

# Display status log in an expander (optional)
with st.expander("Show Processing Logs", expanded=False):
    if st.session_state.status_log:
        st.text_area("Logs:", "\n".join(st.session_state.status_log), height=200, disabled=True)
    else:
        st.caption("No logs yet for this session.")
