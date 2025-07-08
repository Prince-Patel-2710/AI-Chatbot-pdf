from fastapi import APIRouter, HTTPException
import os
import fitz  # PyMuPDF
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dotenv import load_dotenv
import torch
import json

load_dotenv()

summarize_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

PDF_DIR = os.path.abspath("pdfs")
SUMMARY_FILE = os.path.join("summarize", "last_summary.json")
os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)

def get_latest_pdf_file():
    pdf_files = sorted(
        [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")],
        key=lambda x: os.path.getmtime(os.path.join(PDF_DIR, x)),
        reverse=True
    )
    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")
    return os.path.join(PDF_DIR, pdf_files[0])

def extract_pdf_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def split_into_chunks(text, max_chunk_words=250):
    words = text.split()
    return [" ".join(words[i:i+max_chunk_words]) for i in range(0, len(words), max_chunk_words)]

def summarize_chunks(text):
    if not text.strip():
        return "No text content to summarize."

    chunks = split_into_chunks(text)
    summaries = []

    for chunk in chunks[:15]:
        if len(chunk.strip()) < 50:
            continue
        try:
            result = summarizer(
                chunk,
                max_length=min(150, int(len(chunk.split()) * 0.8)),
                min_length=40,
                truncation=True,
                do_sample=False
            )
            if isinstance(result, list) and "summary_text" in result[0]:
                summaries.append(result[0]["summary_text"])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            continue

    return summaries if summaries else "Unable to generate summary."

async def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args))

@summarize_router.post("/summarize")
async def summarize_pdf():
    try:
        pdf_path = await run_in_threadpool(get_latest_pdf_file)
        raw_text = await run_in_threadpool(extract_pdf_text, pdf_path)
        summaries = await run_in_threadpool(summarize_chunks, raw_text)

        if isinstance(summaries, str):
            summaries = [summaries]

        with open(SUMMARY_FILE, "w") as f:
            json.dump(summaries, f)

        return {"pdf": os.path.basename(pdf_path), "summaries": summaries}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@summarize_router.get("/status")
def get_status():
    return {
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
