"""Utility functions for PDF text extraction."""
import pdfplumber
from typing import List, Union
from io import BytesIO
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import streamlit as st

def extract_text(files: List[Union[BytesIO, str]]) -> str:
    """Return concatenated text of uploaded PDFs ('' if none).
    
    Args:
        files: List of uploaded PDF file objects or paths
        
    Returns:
        String containing all extracted text, or empty string if no files
    """
    if not files:
        return ""
    
    all_text = []
    
    for file in files:
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    all_text.append(text)
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            
    return "\n\n".join(all_text)

