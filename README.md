# IncidentMemoryAI

IncidentMemoryAI is a Python portfolio project for incident-response decision support.

It accepts a free-text incident alert, searches synthetic historical incident email threads, and produces an Incident Investigation Report with recommended next steps.

## Problem It Solves

Incident responders often need to quickly answer:

- Has something like this happened before?
- Which team handled it?
- What was the root cause?
- What action should we take now?
- Is this a real escalation or expected maintenance behavior?

IncidentMemoryAI helps operators compare a new alert against historical incident context and turn that context into an action-oriented investigation report.

## Features

- Free-text incident alert input
- Local JSON incident email thread dataset
- Keyword search fallback
- Optional semantic search with sentence-transformers and ChromaDB
- OpenAI-powered Incident Investigation Report when `OPENAI_API_KEY` is configured
- Rule-based fallback when OpenAI is unavailable
- Time-aware maintenance-window detection
- CLI workflow
- Streamlit web UI
- Similar historical incidents shown below the recommendation

## Tech Stack

- Python
- Streamlit
- OpenAI Python SDK
- python-dotenv
- sentence-transformers
- ChromaDB
- JSON demo data

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`:

```bash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

The app works without an OpenAI key by using the rule-based fallback.

## Run The CLI

```bash
python app.py
```

On Windows with the included virtual environment:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Run The Streamlit UI

Recommended:

```bash
python run.py
```

On Windows with the included virtual environment:

```powershell
.\.venv\Scripts\python.exe run.py
```

You can also run Streamlit directly:

```bash
streamlit run ui.py
```

Or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui.py
```

## Demo Data

All incident emails in `data/email_threads.json` are synthetic demo data. They are not real customer, merchant, provider, or internal company emails.

## Future Roadmap

- Add structured incident severity and impact fields
- Add richer timestamp parsing for natural-language alert times
- Cache semantic embeddings instead of rebuilding the index on each run
- Add tests for search, maintenance detection, and fallback reports
- Add export to Markdown or PDF
- Add support for uploading new historical incident threads
- Add evaluation examples for recommendation quality
