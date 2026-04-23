import requests
import streamlit as st


API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Clinical Triage Assistant", page_icon="🩺", layout="centered")
st.title("🩺 AI Clinical Triage Assistant (Local v1)")
st.caption("Chat + form intake. Safety-first triage assistant (no diagnosis, no treatment suggestions).")

if "intake_state" not in st.session_state:
    st.session_state.intake_state = {
        "name": "",
        "age": None,
        "gender": "",
        "duration": "",
        "conditions": "",
        "severity": None,
        "symptoms": "",
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("assistant", "Hi, I can help with intake. Share your name, age, symptoms, duration, and severity (0-10).")
    ]


def run_triage(payload: dict) -> None:
    response = requests.post(f"{API_BASE}/triage", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    urgency = data["urgency"]
    if urgency == "emergency":
        st.error(f"Urgency: {urgency.upper()}")
    elif urgency == "urgent":
        st.warning(f"Urgency: {urgency.upper()}")
    else:
        st.success(f"Urgency: {urgency.upper()}")

    st.write(f"**Department:** {data['department']}")
    st.write(f"**Confidence:** {data['confidence']:.2f}")
    st.write(f"**Summary:** {data['summary']}")
    st.write("**Red Flags:**")
    for rf in data["red_flags"] or []:
        st.markdown(f"- {rf}")
    if not data["red_flags"]:
        st.write("None")
    st.write(f"**Rule Triggered:** {data.get('rule_triggered') or 'none'}")


tab1, tab2 = st.tabs(["Conversational Intake", "Manual Form"])

with tab1:
    st.subheader("Chat-first Intake")
    for role, text in st.session_state.chat_history:
        st.chat_message(role).write(text)

    user_text = st.chat_input("Type message for intake...")
    if user_text:
        st.session_state.chat_history.append(("user", user_text))
        try:
            turn_resp = requests.post(
                f"{API_BASE}/intake/turn",
                json={"message": user_text, "state": st.session_state.intake_state},
                timeout=60,
            )
            turn_resp.raise_for_status()
            data = turn_resp.json()
            st.session_state.intake_state = data["state"]
            st.session_state.chat_history.append(("assistant", data["assistant_reply"]))
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not process intake turn: {exc}")

    st.write("### Current Extracted Fields")
    st.json(st.session_state.intake_state)
    if st.button("Run Triage from Chat Intake"):
        state = st.session_state.intake_state
        required_ok = all(
            state.get(k) not in (None, "")
            for k in ["age", "gender", "duration", "severity", "symptoms"]
        )
        if not required_ok:
            st.warning("Please continue chat intake until all required fields are captured.")
        else:
            payload = {
                "name": state.get("name") or "Patient",
                "age": state["age"],
                "gender": state["gender"],
                "duration": state["duration"],
                "conditions": state.get("conditions") or "",
                "severity": state["severity"],
                "symptoms": state["symptoms"],
            }
            try:
                run_triage(payload)
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

    st.write("### Optional Voice Input")
    audio = st.audio_input("Record a voice message for transcription (optional)")
    if audio is not None and st.button("Transcribe Voice"):
        files = {"audio": ("voice.wav", audio.getvalue(), "audio/wav")}
        try:
            resp = requests.post(f"{API_BASE}/voice/transcribe", files=files, timeout=180)
            if resp.status_code == 501:
                st.info(resp.json().get("detail", "Voice transcription unavailable."))
            else:
                resp.raise_for_status()
                transcript = resp.json().get("transcript", "")
                st.success("Transcription complete. Paste/edit into chat.")
                st.text_area("Transcript", value=transcript, height=120)
        except requests.RequestException as exc:
            st.error(f"Voice transcription failed: {exc}")

with tab2:
    st.subheader("Manual Form Intake")
    with st.form("triage_form"):
        name = st.text_input("Name", value="Patient")
        symptoms = st.text_area("Symptoms", placeholder="Describe symptoms...")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=130, value=30)
            duration = st.text_input("Duration", value="1 day")
            severity = st.slider("Severity (0-10)", min_value=0, max_value=10, value=5)
        with col2:
            gender = st.selectbox("Gender", options=["female", "male", "other"])
            conditions = st.text_input("Known Conditions", value="")
        submitted = st.form_submit_button("Run Triage")

    if submitted:
        if not symptoms.strip():
            st.error("Please enter symptoms.")
        else:
            payload = {
                "name": name,
                "age": age,
                "gender": gender,
                "duration": duration,
                "conditions": conditions,
                "severity": severity,
                "symptoms": symptoms,
            }
            try:
                run_triage(payload)
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")
