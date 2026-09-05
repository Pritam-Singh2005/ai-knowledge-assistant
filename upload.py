import os
import shutil
from pathlib import Path

DOCUMENT_DIR = Path("data_ml/documents")

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_uploaded_file(uploaded_file):

    file_path = DOCUMENT_DIR / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)