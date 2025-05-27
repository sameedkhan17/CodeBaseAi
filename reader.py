#!/usr/bin/env python3
import os
import logging
from typing import List, Set

BASE_DIR = ""  # will be set to the user’s directory (if reader.py is run directly)

# --- Utility Functions for Skipping ---

def should_skip_folder(folder_name: str, skip_folders_list: List[str]) -> bool:
    """Checks if a folder name is in the list of folders to skip."""
    return folder_name in skip_folders_list

def should_skip_file(filename: str, skip_exts_list: List[str]) -> bool:
    """
    Checks if a file should be skipped based on its extension.
    Comparison is case-insensitive and handles leading dots.
    """
    if not filename or not skip_exts_list:
        return False
    
    file_extension = os.path.splitext(filename)[1]
    if not file_extension:  # No extension
        return False

    # Normalize skip_exts to ensure they have a leading dot and are lowercase
    normalized_skip_exts: Set[str] = {
        ext.lower() if ext.startswith('.') else '.' + ext.lower() 
        for ext in skip_exts_list
    }
    return file_extension.lower() in normalized_skip_exts

# --- Markdown Generation ---

def generate_markdown_for_file(file_path: str, output_dir: str = "markdown_files", save_content_to_file: bool = False) -> str:
    """
    Reads a file, creates a Markdown formatted string with its content (as a code block).
    Optionally saves it to a .md file in the specified output directory.

    Args:
        file_path (str): The path to the file to read.
        output_dir (str): The directory where the .md file will be saved (if save_content_to_file is True).
        save_content_to_file (bool): Whether to save the markdown to a file.

    Returns:
        str: The Markdown formatted string (code block), or an empty string if an error occurs or file is empty.
    """
    try:
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            # print(f"Warning (generate_markdown_for_file): File not found or is a directory: {file_path}")
            return ""
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
        
        if not file_content.strip(): # If file is empty or only whitespace
            return ""

    except FileNotFoundError:
        print(f"Error (generate_markdown_for_file): File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error (generate_markdown_for_file): Reading file {file_path}: {e}")
        return ""

    _, ext = os.path.splitext(file_path)
    language = ext[1:] if ext else ""

    markdown_string = f"```{language}\n{file_content}\n```"

    if save_content_to_file:
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except Exception as e:
            print(f"Error (generate_markdown_for_file): Creating output directory {output_dir}: {e}")
            # Continue, but saving might fail

        base_filename = os.path.basename(file_path)
        md_filename_stem = os.path.splitext(base_filename)[0]
        # Ensure unique names if multiple files have same stem (e.g., by appending path hash or full rel path)
        # For simplicity here, just using stem.md
        md_filename = f"{md_filename_stem}.md"
        md_file_path = os.path.join(output_dir, md_filename)

        try:
            with open(md_file_path, 'w', encoding='utf-8') as md_file:
                md_file.write(f"## File: {os.path.abspath(file_path)}\n\n")
                md_file.write(markdown_string)
            # print(f"Markdown file saved: {md_file_path}")
        except Exception as e:
            print(f"Error (generate_markdown_for_file): Writing markdown file {md_file_path}: {e}")

    return markdown_string


# --- Directory Tree Printing ---

def print_tree(start_dir: str, skip_folders_list: List[str], skip_exts_list: List[str], prefix: str = "") -> str:
    """
    Generates a string representation of the directory tree,
    skipping specified folders and file extensions.
    """
    output_lines = []
    
    try:
        # Get all entries and sort them
        entries = sorted(os.listdir(start_dir))
    except PermissionError:
        msg = f"Permission denied: {start_dir}"
        if logging.getLogger().hasHandlers() and logging.getLogger().handlers:
            logging.warning(msg)
        else:
            print(f"Warning: {msg}")
        return ""

    # Filter entries: exclude skipped folders and skipped file types
    displayable_entries = []
    for entry_name in entries:
        full_path = os.path.join(start_dir, entry_name)
        if os.path.isdir(full_path):
            if not should_skip_folder(entry_name, skip_folders_list):
                displayable_entries.append(entry_name)
        else: # It's a file
            if not should_skip_file(entry_name, skip_exts_list):
                displayable_entries.append(entry_name)
    
    for i, entry_name in enumerate(displayable_entries):
        full_path = os.path.join(start_dir, entry_name) # Reconstruct full_path for current entry
        is_last = (i == len(displayable_entries) - 1)
        connector = "└── " if is_last else "├── "
        
        if os.path.isdir(full_path):
            output_lines.append(f"{prefix}{connector}{entry_name}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            # Recursively call print_tree for subdirectories
            output_lines.append(print_tree(full_path, skip_folders_list, skip_exts_list, new_prefix))
        else: # It's a file
            output_lines.append(f"{prefix}{connector}{entry_name}")
            
    return "\n".join(output_lines) # filter(None, ...) removed as empty strings from recursion are less likely with this structure


# --- File Content Reading and Processing ---

def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler() # Also print to console
        ]
    )

def read_and_print_file_content(file_path: str, num_lines: int, current_base_dir_for_relpath: str):
    """Reads and prints a specified number of lines from a file."""
    # Use current_base_dir_for_relpath for relpath calculation
    relative_path = os.path.relpath(file_path, current_base_dir_for_relpath)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            print(f"\n--- {relative_path} (first {num_lines} lines) ---")
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip('\n'))
        if logging.getLogger().hasHandlers() and logging.getLogger().handlers:
            logging.info(f"Read file: {relative_path}")
    except Exception as e:
        msg = f"Failed to read {relative_path}: {e}"
        if logging.getLogger().hasHandlers() and logging.getLogger().handlers:
            logging.error(msg)
        else:
            print(f"Error: {msg}")


def traverse_and_process_files(
    root_scan_path: str, 
    folders_to_skip: List[str], 
    extensions_to_skip: List[str], 
    lines_to_print: int, 
    base_dir_for_relpath: str
):
    """
    Traverses a directory, skipping specified folders and file extensions,
    and calls read_and_print_file_content for non-skipped files.
    """
    for dirpath, dirnames, filenames in os.walk(root_scan_path, topdown=True):
        # Filter out directories to skip from further traversal
        original_dirs = list(dirnames) 
        dirnames[:] = [d for d in dirnames if not should_skip_folder(d, folders_to_skip)]
        
        # Log skipped directories (optional)
        if logging.getLogger().hasHandlers() and logging.getLogger().handlers:
            for skipped_dir_name in set(original_dirs) - set(dirnames):
                rel_skipped_dir_path = os.path.relpath(os.path.join(dirpath, skipped_dir_name), base_dir_for_relpath)
                logging.info(f"Skipped folder (during traversal): {rel_skipped_dir_path}/")

        for fname in filenames:
            full_file_path = os.path.join(dirpath, fname)
            if should_skip_file(fname, extensions_to_skip):
                if logging.getLogger().hasHandlers() and logging.getLogger().handlers:
                    rel_file_path = os.path.relpath(full_file_path, base_dir_for_relpath)
                    logging.info(f"Skipped file (by extension): {rel_file_path}")
                continue
            
            # If not skipped, process the file
            read_and_print_file_content(full_file_path, lines_to_print, base_dir_for_relpath)

# --- Main CLI Execution ---

def main():
    global BASE_DIR # Set the global BASE_DIR for context if needed by other functions directly
    
    BASE_DIR = input("Enter directory path to scan: ").strip()
    if not os.path.isdir(BASE_DIR):
        print(f"Error: Directory not found: {BASE_DIR}")
        return

    # --- Configuration for Tree Printing ---
    skip_tree_folders_str = input("Enter folder names to skip in tree (comma-separated, e.g. .git,venv): ").strip()
    skip_tree_folders_list = [f.strip() for f in skip_tree_folders_str.split(',') if f.strip()]

    # --- Configuration for File Processing (traverse_and_process) ---
    # By default, use the same folder skip list for processing, or allow a different one
    skip_processing_folders_str = input(f"Enter folder names to skip during file content processing (comma-separated, default '{skip_tree_folders_str}'): ").strip()
    if skip_processing_folders_str:
        skip_processing_folders_list = [f.strip() for f in skip_processing_folders_str.split(',') if f.strip()]
    else:
        skip_processing_folders_list = skip_tree_folders_list
    
    skip_exts_str = input("Enter file extensions to skip for BOTH tree and processing (comma-separated, include dot, e.g. .log,.tmp,.pyc): ").strip()
    skip_exts_list = [e.strip() for e in skip_exts_str.split(',') if e.strip()]

    num_lines_str = input("Enter number of lines to print from each file (default 5): ").strip()
    log_file_path_str = input("Enter log file path (default 'scan_directory.log'): ").strip()

    num_lines = int(num_lines_str) if num_lines_str.isdigit() and int(num_lines_str) > 0 else 5
    log_file_path = log_file_path_str if log_file_path_str else 'scan_directory.log'

    # --- Setup Logging ---
    setup_logging(log_file_path)
    logging.info(f"--- Initiating Scan ---")
    logging.info(f"Scanning Directory: {os.path.abspath(BASE_DIR)}")
    logging.info(f"Skipping folders in Tree: {skip_tree_folders_list}")
    logging.info(f"Skipping folders for Processing: {skip_processing_folders_list}")
    logging.info(f"Skipping extensions (Tree & Processing): {skip_exts_list}")
    logging.info(f"Lines to print per file: {num_lines}")
    logging.info(f"Log file: {os.path.abspath(log_file_path)}")


    # --- Generate and Print Directory Tree ---
    print(f"\n--- Directory Tree for {BASE_DIR} ---")
    print(f"(Skipping folders: {skip_tree_folders_list}, Skipping extensions: {skip_exts_list})")
    tree_structure = print_tree(BASE_DIR, skip_tree_folders_list, skip_exts_list)
    print(tree_structure)
    logging.info(f"Directory tree generated for {BASE_DIR}")


    # --- Traverse and Process Files ---
    print(f"\n--- Processing Files in {BASE_DIR} ---")
    print(f"(Skipping folders: {skip_processing_folders_list}, Skipping extensions: {skip_exts_list})")
    traverse_and_process_files(BASE_DIR, skip_processing_folders_list, skip_exts_list, num_lines, BASE_DIR)
    
    logging.info("--- Scan Completed ---")
    print(f"\nScan complete. Log saved to {os.path.abspath(log_file_path)}")

if __name__ == '__main__':
    main()