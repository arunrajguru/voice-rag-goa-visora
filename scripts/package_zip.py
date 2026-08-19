"""
ZIP Packaging Utility for HH Goa 2026 Task 2 Submission

Creates voice-rag-goa-2026.zip containing the complete project directory structure,
excluding secrets, build artifacts, virtualenvs, node_modules, and cache files.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "voice-rag-goa-2026.zip"
OUTPUT_ZIP_PATH = PROJECT_ROOT.parent / ZIP_NAME

EXCLUDE_DIRS = {
    ".venv", "venv", "node_modules", "__pycache__", ".git", ".idea", ".vscode",
    "data/index", "evaluation/results"
}
EXCLUDE_FILES = {
    ".env", ZIP_NAME, ".DS_Store", "faiss.index", "bm25.pkl"
}

def create_zip():
    print("=" * 60)
    print(f"CREATING SUBMISSION ZIP ARCHIVE: {ZIP_NAME}")
    print("=" * 60)
    
    zip_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            
            for file in files:
                if file in EXCLUDE_FILES or file.endswith('.pyc') or file.endswith('.zip'):
                    continue
                    
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT.parent)
                zipf.write(file_path, arcname=str(rel_path))
                zip_count += 1

    print(f"Successfully packaged {zip_count} files into:")
    print(f"-> {OUTPUT_ZIP_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    create_zip()
