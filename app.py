import json
import os
import re
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

DATA_FILE = Path("data/email_threads.json")
CHROMA_DB_DIR = Path("chroma_db")
CHROMA_COLLECTION_NAME = "email_incident_threads"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATETIME_FORMAT = "%Y-%m-%d %H:%M"
DATETIME_PATTERN = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"
DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
TIME_PATTERN = r"\d{2}:\d{2}"


def load_threads(file_path):
    """Load historical incident email threads from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_keywords(alert_text):
    """Turn the alert text into simple searchable keywords."""
    words = alert_text.lower().replace(",", " ").replace(".", " ").split()
    return [word for word in words if len(word) > 2]


def thread_text(thread):
    """Combine all message text in a thread into one searchable string."""
    messages = thread.get("messages", [])
    combined_messages = " ".join(message.get("body", "") for message in messages)
    return combined_messages.lower()


def search_threads(alert_text, threads):
    """Find threads where alert keywords appear in the historical messages."""
    keywords = get_keywords(alert_text)
    matches = []

    for thread in threads:
        searchable_text = thread_text(thread)
        matched_keywords = [word for word in keywords if word in searchable_text]

        if matched_keywords:
            matches.append(
                {
                    "thread": thread,
                    "score": len(set(matched_keywords)),
                }
            )

    # Show the threads with the most keyword matches first.
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches


def thread_document(thread):
    """Create one searchable text document for an email thread."""
    message_bodies = [message.get("body", "") for message in thread.get("messages", [])]

    parts = [
        thread.get("subject", ""),
        thread.get("owner_team", ""),
        thread.get("root_cause", ""),
        thread.get("resolution", ""),
        " ".join(message_bodies),
    ]

    return "\n".join(parts)


def load_embedding_model():
    """Load the local sentence-transformers model, if it is installed."""
    try:
        from sentence_transformers import SentenceTransformer

        # Use the local cache only. If the model is not available locally,
        # fall back to keyword search instead of blocking the CLI or UI.
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception as error:
        error_type = type(error).__name__
        print(f"Semantic search unavailable: {error_type}: {error}")
        print("Using keyword search fallback.")
        print()
        return None


def build_or_refresh_chroma_index(threads):
    """Build a local ChromaDB index from the incident email threads."""
    try:
        import chromadb
    except Exception as error:
        error_type = type(error).__name__
        print(f"Semantic search unavailable: {error_type}: {error}")
        print("Using keyword search fallback.")
        print()
        return None, None

    model = load_embedding_model()

    if model is None:
        return None, None

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        # Rebuild the small local index on each run so it always matches the JSON.
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
        documents = [thread_document(thread) for thread in threads]
        ids = [str(thread["id"]) for thread in threads]
        embeddings = model.encode(documents).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
        )

        return collection, model
    except Exception as error:
        error_type = type(error).__name__
        print(f"Semantic search unavailable: {error_type}: {error}")
        print("Using keyword search fallback.")
        print()
        return None, None


def search_chroma_index(alert_text, threads, collection, model):
    """Search ChromaDB using the user alert text and return the top 3 threads."""
    if collection is None or model is None:
        return None

    try:
        query_embedding = model.encode([alert_text]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

        thread_by_id = {str(thread["id"]): thread for thread in threads}
        result_ids = results.get("ids", [[]])[0]
        matches = []

        for index, thread_id in enumerate(result_ids):
            thread = thread_by_id.get(thread_id)

            if thread:
                matches.append(
                    {
                        "thread": thread,
                        # Keep a simple score so the existing fallback confidence works.
                        "score": 3 - index,
                    }
                )

        return matches
    except Exception as error:
        error_type = type(error).__name__
        print(f"Semantic search unavailable: {error_type}: {error}")
        print("Using keyword search fallback.")
        print()
        return None


def find_matching_threads(alert_text, threads):
    """Find similar threads with semantic search, falling back to keyword search."""
    collection, model = build_or_refresh_chroma_index(threads)
    matches = search_chroma_index(alert_text, threads, collection, model)

    if matches is None:
        matches = search_threads(alert_text, threads)

    return matches


def parse_datetime(value):
    """Parse a timestamp in YYYY-MM-DD HH:mm format."""
    return datetime.strptime(value, DATETIME_FORMAT)


def parse_maintenance_windows(thread):
    """Find maintenance windows mentioned in a historical thread."""
    windows = []
    searchable_text = thread_document(thread)
    pattern = rf"from ({DATETIME_PATTERN}) to ({DATETIME_PATTERN})"

    for match in re.finditer(pattern, searchable_text, flags=re.IGNORECASE):
        start_time = parse_datetime(match.group(1))
        end_time = parse_datetime(match.group(2))
        windows.append(
            {
                "thread": thread,
                "start": start_time,
                "end": end_time,
                "source_text": match.group(0),
            }
        )

    return windows


def extract_alert_times(alert_text):
    """Extract timestamps from the current alert when they are available."""
    alert_times = []

    for value in re.findall(DATETIME_PATTERN, alert_text):
        alert_times.append(parse_datetime(value))

    # Also support alerts that mention one date and one or more times separately.
    dates = re.findall(DATE_PATTERN, alert_text)
    times = re.findall(TIME_PATTERN, alert_text)

    if dates:
        alert_date = dates[0]

        for alert_time in times:
            value = f"{alert_date} {alert_time}"
            parsed_time = parse_datetime(value)

            if parsed_time not in alert_times:
                alert_times.append(parsed_time)

    return alert_times


def find_maintenance_match(alert_text, matches):
    """Check whether the alert time falls inside a matched maintenance window."""
    alert_times = extract_alert_times(alert_text)

    if not alert_times:
        return None

    for match in matches:
        thread = match["thread"]

        for window in parse_maintenance_windows(thread):
            for alert_time in alert_times:
                if window["start"] <= alert_time <= window["end"]:
                    return {
                        "thread": thread,
                        "alert_time": alert_time,
                        "window": window,
                    }

    return None


def maintenance_analysis_text(maintenance_match):
    """Format maintenance analysis for the AI prompt."""
    if maintenance_match is None:
        return "No matching maintenance window was detected from the current alert time."

    thread = maintenance_match["thread"]
    window = maintenance_match["window"]

    return (
        "The current alert time falls inside a known historical maintenance window.\n"
        f"Alert time: {maintenance_match['alert_time'].strftime(DATETIME_FORMAT)}\n"
        f"Maintenance window: {window['start'].strftime(DATETIME_FORMAT)} to "
        f"{window['end'].strftime(DATETIME_FORMAT)}\n"
        f"Historical maintenance thread: {thread['subject']}\n"
        "Required classification: Scheduled Maintenance\n"
        "Required action type: Monitor\n"
        "Required confidence: high\n"
        "Required recommended action: No escalation required unless impact exceeds "
        "expected maintenance behavior."
    )


def timeline_summary(thread):
    """Create a short timeline summary from a thread's messages."""
    timeline_lines = []

    for message in thread.get("messages", []):
        timeline_lines.append(
            f"- {message['timestamp']} | {message['from']}: {message['body']}"
        )

    return "\n".join(timeline_lines)


def thread_prompt_summary(match):
    """Format one matching thread for the AI prompt."""
    thread = match["thread"]

    return (
        f"Subject: {thread['subject']}\n"
        f"Owner team: {thread['owner_team']}\n"
        f"Match score: {match['score']}\n"
        f"Root cause: {thread['root_cause']}\n"
        f"Resolution: {thread['resolution']}\n"
        f"Timeline:\n{timeline_summary(thread)}"
    )


def build_ai_prompt(alert_text, matches):
    """Build the prompt sent to the LLM."""
    top_matches = matches[:3]
    maintenance_match = find_maintenance_match(alert_text, top_matches)

    if top_matches:
        historical_context = "\n\n---\n\n".join(
            thread_prompt_summary(match) for match in top_matches
        )
    else:
        historical_context = "No historical incidents were found for this alert."

    return f"""
You are helping an incident manager review a new production alert.

Current alert:
{alert_text}

Historical incident context:
{historical_context}

Time-aware maintenance analysis:
{maintenance_analysis_text(maintenance_match)}

Use the historical incident details as evidence. Focus on practical incident
response workflow: classify the alert, explain confidence, identify the most
relevant historical incident, recommend action, and list investigation steps.
If the time-aware maintenance analysis says the alert is inside a maintenance
window, classify it as Scheduled Maintenance, use confidence high, and recommend
monitoring instead of escalation unless impact exceeds expected maintenance behavior.

Return a concise Incident Investigation Report using exactly this format:

Incident Classification: <classification>

Confidence:
<high/medium/low>

Most Relevant Historical Incident: <subject or none found>

Evidence:

* <evidence from historical thread or current alert>
* <evidence from historical thread or current alert>
* <evidence from historical thread or current alert>

Recommended Action:
<clear operational action>

Suggested Investigation Steps:

1. <first operational check>
2. <second operational check>
3. <third operational check>

Suggested Message:
<short email or Slack message to send>
""".strip()


def classification_for_thread(thread):
    """Create a simple fallback incident classification."""
    return f"{thread['owner_team']} incident"


def evidence_for_thread(thread):
    """Build evidence bullets from the best matching historical thread."""
    messages = thread.get("messages", [])
    evidence = [
        f"Historical owner team was {thread['owner_team']}.",
        f"Previous root cause: {thread['root_cause']}",
        f"Previous resolution: {thread['resolution']}",
    ]

    if messages:
        evidence.append(f"Timeline signal: {messages[0]['body']}")

    return evidence[:3]


def print_maintenance_report(maintenance_match):
    """Print a deterministic report for alerts inside maintenance windows."""
    thread = maintenance_match["thread"]
    window = maintenance_match["window"]
    alert_time = maintenance_match["alert_time"]

    print("Incident Classification: Scheduled Maintenance")
    print()
    print("Confidence:")
    print("high")
    print()
    print(f"Most Relevant Historical Incident: {thread['subject']}")
    print()
    print("Evidence:")
    print()
    print(
        f"* Current alert time {alert_time.strftime(DATETIME_FORMAT)} falls inside "
        f"the known maintenance window from {window['start'].strftime(DATETIME_FORMAT)} "
        f"to {window['end'].strftime(DATETIME_FORMAT)}."
    )
    print(f"* Historical root cause: {thread['root_cause']}")
    print(f"* Historical resolution: {thread['resolution']}")
    print()
    print("Recommended Action:")
    print("Action Type: Monitor")
    print("No escalation required unless impact exceeds expected maintenance behavior.")
    print()
    print("Suggested Investigation Steps:")
    print()
    print("1. Confirm the alert volume and affected merchants match the maintenance notice.")
    print("2. Monitor provider status until the maintenance window ends.")
    print("3. Escalate only if failures continue after the window or impact is higher than expected.")
    print()
    print("Suggested Message:")
    print(
        "Hi Provider Operations, the current alert falls within the known provider "
        "maintenance window. We will monitor impact and update merchants; please "
        "confirm if behavior exceeds the expected maintenance impact."
    )


def investigation_steps_for_thread(thread):
    """Create simple operational investigation steps from the owning team."""
    owner_team = thread["owner_team"]

    if owner_team == "Payments Platform":
        return [
            "Check current payment provider connectivity and error rates.",
            "Confirm whether backup routing is available and healthy.",
            "Ask Payments Platform to compare the alert with the previous incident timeline.",
        ]

    if owner_team == "Database Operations":
        return [
            "Check database replication lag and replica health.",
            "Review payment status update delays and worker queue depth.",
            "Ask Database Operations to confirm whether replay or restart is needed.",
        ]

    if owner_team == "Card Processing":
        return [
            "Check current 3DS provider error responses.",
            "Compare authentication failure rates with normal baseline.",
            "Prepare a merchant update while monitoring provider recovery.",
        ]

    if owner_team == "Provider Operations":
        return [
            "Check whether the alert time falls inside a provider maintenance window.",
            "Confirm expected provider impact and affected payment methods.",
            "Monitor recovery and send a merchant update if delays continue.",
        ]

    return [
        f"Contact {owner_team} and request an initial health check.",
        "Compare current alert timing and symptoms with the historical thread.",
        "Document findings and decide whether escalation is needed.",
    ]


def generate_ai_recommendation(alert_text, matches):
    """Generate the final recommendation with OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client = client.with_options(timeout=30.0, max_retries=2)
        prompt = build_ai_prompt(alert_text, matches)

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )

        return response.output_text
    except Exception as error:
        error_type = type(error).__name__
        error_message = str(error).splitlines()[0]
        print(f"AI recommendation unavailable: {error_type}: {error_message}")
        print("Using rule-based fallback recommendation.")
        print()
        return None


def openai_is_configured():
    """Check whether an OpenAI API key was loaded from the environment."""
    return bool(os.getenv("OPENAI_API_KEY"))


def recommendation_for_team(owner_team):
    """Return a simple rule-based action for the team that owns the top match."""
    if owner_team == "Payments Platform":
        return "Contact Payments Platform and check provider connectivity or backup routing."

    if owner_team == "Database Operations":
        return "Contact Database Operations and check replication lag or status update delays."

    if owner_team == "Card Processing":
        return "Contact Card Processing and check 3DS provider errors."

    if owner_team == "Provider Operations":
        return "Contact Provider Operations, confirm the maintenance window, monitor impact, and update merchants."

    return "Contact the owning team and compare the current alert with the previous incident."


def action_type_for_team(owner_team):
    """Return the action type for the team that owns the top match."""
    if owner_team == "Payments Platform":
        return "Escalate"

    if owner_team == "Database Operations":
        return "Escalate"

    if owner_team == "Card Processing":
        return "Monitor + Update Merchant"

    return "Manual Investigation"


def confidence_for_score(score):
    """Return a confidence label based on the number of keyword matches."""
    if score >= 3:
        return "High"

    if score == 2:
        return "Medium"

    return "Low"


def suggested_message(thread):
    """Create a short message that can be sent to the relevant team."""
    return (
        f"Hi {thread['owner_team']}, we are seeing an alert similar to "
        f"'{thread['subject']}'. Can you please check this area and confirm "
        "whether the previous root cause or resolution applies?"
    )


def print_recommendation(matches, alert_text=None):
    """Print a fallback Incident Investigation Report from the best match."""
    if alert_text:
        maintenance_match = find_maintenance_match(alert_text, matches)

        if maintenance_match:
            print_maintenance_report(maintenance_match)
            return

    if not matches:
        print("Incident Classification: Unknown incident")
        print()
        print("Confidence:")
        print("low")
        print()
        print("Most Relevant Historical Incident: None found")
        print()
        print("Evidence:")
        print()
        print("* No similar historical incidents were found.")
        print("* The current alert needs manual triage.")
        print("* There is no previous root cause or resolution to compare against.")
        print()
        print("Recommended Action:")
        print("Start a manual investigation because no similar historical threads were found.")
        print()
        print("Suggested Investigation Steps:")
        print()
        print("1. Confirm the alert source, affected service, and customer impact.")
        print("2. Check recent deployments, provider notices, and service health dashboards.")
        print("3. Assign an owner team and collect first findings before escalation.")
        print()
        print("Suggested Message:")
        print(
            "Hi team, we have a new incident alert with no close historical match. "
            "Please begin manual investigation and share initial findings."
        )
        return

    top_match = matches[0]
    top_thread = top_match["thread"]

    print(f"Incident Classification: {classification_for_thread(top_thread)}")
    print()
    print("Confidence:")
    print(confidence_for_score(top_match["score"]).lower())
    print()
    print(f"Most Relevant Historical Incident: {top_thread['subject']}")
    print()
    print("Evidence:")
    print()
    for evidence in evidence_for_thread(top_thread):
        print(f"* {evidence}")
    print()
    print("Recommended Action:")
    print(recommendation_for_team(top_thread["owner_team"]))
    print()
    print("Suggested Investigation Steps:")
    print()
    for index, step in enumerate(investigation_steps_for_thread(top_thread), start=1):
        print(f"{index}. {step}")
    print()
    print("Suggested Message:")
    print(suggested_message(top_thread))


def print_incident_summary(thread):
    """Print one matching thread as a structured incident summary."""
    print(f"Subject: {thread['subject']}")
    print()
    print("Owner Team:")
    print(thread["owner_team"])
    print()
    print("Root Cause:")
    print(thread["root_cause"])
    print()
    print(f"Resolution: {thread['resolution']}")
    print()
    print("Timeline:")
    print()

    for message in thread.get("messages", []):
        print(f"* {message['timestamp']} | {message['from']}")
        print(f"  {message['body']}")


def print_matches(matches):
    """Print structured incident summaries for all matching threads."""
    if not matches:
        print("No similar incidents found.")
        return

    for index, match in enumerate(matches, start=1):
        thread = match["thread"]
        print("=" * 60)
        print(f"Match {index}")
        print("=" * 60)
        print()
        print_incident_summary(thread)
        print()


def capture_printed_output(print_function, *args):
    """Capture output from an existing print-based function as text."""
    output = StringIO()

    with redirect_stdout(output):
        print_function(*args)

    return output.getvalue().strip()


def create_incident_report(alert_text, matches):
    """Create an AI report if possible, otherwise use the rule-based fallback."""
    ai_recommendation = generate_ai_recommendation(alert_text, matches)

    if ai_recommendation:
        return ai_recommendation

    return capture_printed_output(print_recommendation, matches, alert_text)


def create_similar_incidents_text(matches):
    """Create the detailed matching output as text."""
    return capture_printed_output(print_matches, matches)


def main():
    if openai_is_configured():
        print("OpenAI available")
    else:
        print("OpenAI unavailable - using fallback")

    threads = load_threads(DATA_FILE)

    alert_text = input("Paste incident alert text: ")
    matches = find_matching_threads(alert_text, threads)

    print()
    print(create_incident_report(alert_text, matches))

    print("\nSimilar historical incidents:\n")
    print(create_similar_incidents_text(matches))


if __name__ == "__main__":
    main()
