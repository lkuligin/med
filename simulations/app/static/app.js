(() => {
  "use strict";

  const state = {
    userEmail: "",
    storageBackend: "gcs",
    patients: [],
    session: null,
    isBusy: false,
  };

  const els = {
    doctorEmail: document.getElementById("doctor-email"),
    ehrSelect: document.getElementById("ehr-select"),
    startChatBtn: document.getElementById("start-chat-btn"),
    endChatBtn: document.getElementById("end-chat-btn"),
    sessionId: document.getElementById("session-id"),
    turnCount: document.getElementById("turn-count"),
    saveStatus: document.getElementById("save-status"),
    endedNotice: document.getElementById("ended-notice"),
    errorBanner: document.getElementById("error-banner"),
    chatPane: document.getElementById("chat-pane"),
    emptyState: document.getElementById("empty-state"),
    messages: document.getElementById("messages"),
    typingIndicator: document.getElementById("typing-indicator"),
    chatForm: document.getElementById("chat-form"),
    messageInput: document.getElementById("message-input"),
    sendBtn: document.getElementById("send-btn"),
  };

  function showError(message) {
    if (!message) {
      els.errorBanner.textContent = "";
      els.errorBanner.classList.add("hidden");
      return;
    }
    els.errorBanner.textContent = message;
    els.errorBanner.classList.remove("hidden");
  }

  function savedLabel() {
    return state.storageBackend === "gcs" ? "Saved to GCS" : "Saved locally";
  }

  function setSaveStatus(kind, text) {
    els.saveStatus.textContent = text;
    els.saveStatus.classList.remove("saved", "saving");
    if (kind === "saved") {
      els.saveStatus.classList.add("saved");
    } else if (kind === "saving") {
      els.saveStatus.classList.add("saving");
    }
  }

  async function apiFetch(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    });

    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      const detail = data && data.detail ? data.detail : `Request failed (${response.status})`;
      throw new Error(detail);
    }
    return data;
  }

  function formatTimestamp(isoString) {
    if (!isoString) return "";
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return isoString;
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function scrollChatToBottom() {
    els.chatPane.scrollTop = els.chatPane.scrollHeight;
  }

  function createMessageNode(turn) {
    const role = turn.role === "doctor" ? "doctor" : "patient";
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = turn.text || "";

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const speakerLabel = role === "doctor" ? "Doctor" : "Patient";
    const timeText = formatTimestamp(turn.timestamp);
    meta.textContent = timeText ? `${speakerLabel} · ${timeText}` : speakerLabel;

    row.appendChild(bubble);
    row.appendChild(meta);
    return row;
  }

  function renderSession() {
    const session = state.session;
    els.messages.innerHTML = "";

    if (!session) {
      els.emptyState.classList.remove("hidden");
      els.sessionId.textContent = "—";
      els.turnCount.textContent = "0";
      els.endedNotice.classList.add("hidden");
      setSaveStatus("idle", "Not started");
      updateControls();
      return;
    }

    els.emptyState.classList.add("hidden");
    const turns = Array.isArray(session.turns) ? session.turns : [];
    for (const turn of turns) {
      els.messages.appendChild(createMessageNode(turn));
    }

    els.sessionId.textContent = session.session_id || "—";
    els.turnCount.textContent = String(turns.length);

    const isEnded = session.status !== "active";
    els.endedNotice.classList.toggle("hidden", !isEnded);
    setSaveStatus("saved", savedLabel());
    updateControls();
    scrollChatToBottom();
  }

  function updateControls() {
    const hasPatients = state.patients.length > 0;
    const selectedPatient = getSelectedPatient();
    const hasProfiles = Boolean(selectedPatient && selectedPatient.profiles.length > 0);
    const hasActiveSession = Boolean(state.session && state.session.status === "active");

    els.ehrSelect.disabled = !hasPatients || state.isBusy;
    els.startChatBtn.disabled = !hasProfiles || state.isBusy;
    els.endChatBtn.disabled = !hasActiveSession || state.isBusy;

    els.messageInput.disabled = !hasActiveSession || state.isBusy;
    els.sendBtn.disabled = !hasActiveSession || state.isBusy;
    els.typingIndicator.classList.toggle("hidden", !state.isBusy || !hasActiveSession);
  }

  function getSelectedPatient() {
    const ehrId = els.ehrSelect.value;
    return state.patients.find((p) => p.ehr_id === ehrId) || null;
  }

  function getSelectedProfile() {
    const patient = getSelectedPatient();
    return patient && patient.profiles && patient.profiles.length > 0
      ? patient.profiles[0]
      : null;
  }

  function syncSessionUrl(sessionId) {
    const url = new URL(window.location.href);
    if (sessionId) {
      url.searchParams.set("session_id", sessionId);
    } else {
      url.searchParams.delete("session_id");
    }
    window.history.replaceState({}, "", url.toString());
  }

  async function handleStartChat() {
    const patient = getSelectedPatient();
    const profile = getSelectedProfile();
    if (!patient || !profile) return;

    showError("");
    state.isBusy = true;
    setSaveStatus("saving", "Creating session...");
    updateControls();

    try {
      const session = await apiFetch("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          ehr_id: patient.ehr_id,
          profile_id: profile.profile_id,
          use_seed_question: true,
        }),
      });
      state.session = session;
      syncSessionUrl(session.session_id);
      renderSession();
      els.messageInput.focus();
    } catch (err) {
      showError(err.message);
      setSaveStatus("idle", state.session ? savedLabel() : "Not started");
    } finally {
      state.isBusy = false;
      updateControls();
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault();
    if (!state.session || state.session.status !== "active" || state.isBusy) return;

    const text = els.messageInput.value.trim();
    if (!text) return;

    showError("");
    els.messageInput.value = "";

    // Render doctor's turn immediately while awaiting the simulated patient.
    const optimisticTurn = {
      role: "doctor",
      text,
      timestamp: new Date().toISOString(),
    };
    const optimisticNode = createMessageNode(optimisticTurn);
    els.messages.appendChild(optimisticNode);
    els.turnCount.textContent = String((state.session.turns || []).length + 1);

    state.isBusy = true;
    setSaveStatus("saving", "Saving...");
    updateControls();
    scrollChatToBottom();

    try {
      const result = await apiFetch(`/api/sessions/${encodeURIComponent(state.session.session_id)}/messages`, {
        method: "POST",
        body: JSON.stringify({ text }),
      });

      state.session.turns.push(result.doctor_turn, result.patient_turn);
      state.session.status = result.status;
      state.session.updated_at = result.updated_at;
      renderSession();
    } catch (err) {
      optimisticNode.remove();
      els.messageInput.value = text;
      showError(err.message);
      setSaveStatus("saved", savedLabel());
    } finally {
      state.isBusy = false;
      updateControls();
      if (state.session && state.session.status === "active") {
        els.messageInput.focus();
      }
    }
  }

  async function handleEndChat() {
    if (!state.session || state.session.status !== "active" || state.isBusy) return;

    showError("");
    state.isBusy = true;
    setSaveStatus("saving", "Ending chat...");
    updateControls();

    try {
      const result = await apiFetch(`/api/sessions/${encodeURIComponent(state.session.session_id)}/end`, {
        method: "POST",
      });
      state.session.status = result.status;
      state.session.updated_at = result.updated_at;
      renderSession();
    } catch (err) {
      showError(err.message);
      setSaveStatus("saved", savedLabel());
    } finally {
      state.isBusy = false;
      updateControls();
    }
  }

  async function init() {
    els.ehrSelect.addEventListener("change", updateControls);
    els.startChatBtn.addEventListener("click", handleStartChat);
    els.endChatBtn.addEventListener("click", handleEndChat);
    els.chatForm.addEventListener("submit", handleSendMessage);

    try {
      const [me, catalogData] = await Promise.all([
        apiFetch("/api/me"),
        apiFetch("/api/catalog"),
      ]);

      state.userEmail = me.email || "";
      state.storageBackend = me.storage_backend || "gcs";
      els.doctorEmail.textContent = state.userEmail || "Unknown";

      state.patients = Array.isArray(catalogData.patients) ? catalogData.patients : [];
      els.ehrSelect.innerHTML = "";

      if (state.patients.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No patients found";
        els.ehrSelect.appendChild(opt);
      } else {
        for (const patient of state.patients) {
          const opt = document.createElement("option");
          opt.value = patient.ehr_id;
          opt.textContent =
            patient.display_label && patient.display_label !== patient.ehr_id
              ? `${patient.display_label} (${patient.ehr_id})`
              : patient.ehr_id;
          els.ehrSelect.appendChild(opt);
        }
      }

      updateControls();

      // Reload an existing session if present in URL query string.
      const existingSessionId = new URLSearchParams(window.location.search).get("session_id");
      if (existingSessionId) {
        try {
          state.session = await apiFetch(`/api/sessions/${encodeURIComponent(existingSessionId)}`);
          if (state.session.patient && state.session.patient.ehr_id) {
            els.ehrSelect.value = state.session.patient.ehr_id;
          }
          renderSession();
        } catch (err) {
          showError(`Could not reload session: ${err.message}`);
          syncSessionUrl(null);
        }
      }
    } catch (err) {
      els.doctorEmail.textContent = "Not authenticated";
      showError(err.message);
    } finally {
      updateControls();
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
