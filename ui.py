import re

import streamlit as st

from app import (
    DATA_FILE,
    create_incident_report,
    find_matching_threads,
    load_threads,
    openai_is_configured,
)


def section_text(report, heading):
    """Extract the text under a report heading."""
    pattern = rf"{re.escape(heading)}:\s*\n?(.*?)(?=\n[A-Z][A-Za-z ]+:\s*|\Z)"
    match = re.search(pattern, report, flags=re.DOTALL)

    if not match:
        return ""

    return match.group(1).strip()


def first_line(value, default):
    """Return the first non-empty line from a block of text."""
    for line in value.splitlines():
        clean_line = line.strip()

        if clean_line:
            return clean_line

    return default


def infer_action_type(report, matches):
    """Infer an operator action type from the report and best match."""
    report_lower = report.lower()

    if "action type: monitor" in report_lower or "scheduled maintenance" in report_lower:
        return "Monitor"

    if not matches:
        return "Manual Investigation"

    owner_team = matches[0]["thread"].get("owner_team", "")

    if owner_team in ["Payments Platform", "Database Operations"]:
        return "Escalate"

    if owner_team == "Card Processing":
        return "Update Merchant"

    if owner_team == "Provider Operations":
        return "Monitor"

    return "Manual Investigation"


def action_required_for_type(action_type):
    """Convert the action type into a simple operator decision."""
    if action_type == "Monitor":
        return "Monitor Only"

    if action_type == "Ignore":
        return "No"

    return "Yes"


def reason_from_report(report):
    """Use evidence as the short reason in the main action panel."""
    evidence = section_text(report, "Evidence")

    if evidence:
        bullets = [
            line.strip("* ").strip()
            for line in evidence.splitlines()
            if line.strip().startswith("*")
        ]

        if bullets:
            return bullets[0]

    classification = section_text(report, "Incident Classification")
    return first_line(classification, "Review the incident report for details.")


def build_action_panel(report, matches):
    """Build the practical operator summary shown at the top of the UI."""
    action_type = infer_action_type(report, matches)
    recommended_action = section_text(report, "Recommended Action")
    suggested_message = section_text(report, "Suggested Message")

    # Fallback reports may include "Action Type: Monitor" as the first line.
    recommended_action = recommended_action.replace("Action Type: Monitor", "").strip()

    return {
        "action_required": action_required_for_type(action_type),
        "action_type": action_type,
        "next_step": first_line(recommended_action, "Review the incident report and assign an owner."),
        "reason": reason_from_report(report),
        "suggested_message": suggested_message or "No suggested message available.",
    }


def show_timeline(thread):
    """Render the thread timeline inside an expander."""
    for message in thread.get("messages", []):
        st.markdown(f"**{message['timestamp']} | {message['from']}**")
        st.write(message["body"])


def show_historical_incidents(matches):
    """Render matching incidents as collapsed Streamlit expanders."""
    if not matches:
        st.info("No similar historical incidents found.")
        return

    for match in matches:
        thread = match["thread"]

        with st.expander(thread["subject"]):
            st.markdown(f"**Owner Team:** {thread['owner_team']}")
            st.markdown(f"**Root Cause:** {thread['root_cause']}")
            st.markdown(f"**Resolution:** {thread['resolution']}")
            st.markdown("**Timeline**")
            show_timeline(thread)


def main():
    """Render the Streamlit app."""
    st.set_page_config(page_title="IncidentMemoryAI")

    st.title("IncidentMemoryAI")
    st.caption("Operator decision support for incident alerts")

    if openai_is_configured():
        st.caption("OpenAI available")
    else:
        st.caption("OpenAI unavailable - using fallback")

    alert_text = st.text_area(
        "Incident alert",
        height=180,
        placeholder="Paste a free-text incident alert here...",
    )

    if st.button("Analyze", type="primary"):
        if not alert_text.strip():
            st.warning("Paste an incident alert before analyzing.")
        else:
            with st.spinner("Analyzing incident..."):
                threads = load_threads(DATA_FILE)
                matches = find_matching_threads(alert_text, threads)
                report = create_incident_report(alert_text, matches)
                action_panel = build_action_panel(report, matches)

            st.subheader("Main Action Panel")

            col1, col2 = st.columns(2)
            col1.metric("Action Required", action_panel["action_required"])
            col2.metric("Action Type", action_panel["action_type"])

            st.markdown("**Recommended Next Step**")
            st.write(action_panel["next_step"])

            st.markdown("**Reason**")
            st.write(action_panel["reason"])

            st.markdown("**Suggested Message**")
            st.code(action_panel["suggested_message"], language="text")

            with st.expander("Full Incident Investigation Report"):
                st.markdown(report)

            st.subheader("Similar Historical Incidents")
            show_historical_incidents(matches)


if __name__ == "__main__":
    main()
