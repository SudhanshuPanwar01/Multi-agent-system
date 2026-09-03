import re
import time

import streamlit as st

from agent import fast_chain, writer_chain, critic_chain
from tools import web_search, scrape_url
from auth import login, sign_up
from history import save_report, load_history


st.set_page_config(
    page_title="Multi AI Agent System",
    layout="wide",
)

# ── GREY THEME ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0c, #141418, #1d1d23, #0a0a0c);
        background-size: 400% 400%;
        animation: gradientShift 12s ease infinite;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1 {
        background: linear-gradient(90deg, #ffffff, #9ca3af, #e5e7eb, #ffffff);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        margin-bottom: 0;
    }
    @keyframes shine { to { background-position: 300% center; } }

    .logo-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.8s ease both;
    }
    .logo-svg {
        filter: drop-shadow(0 0 12px rgba(200,200,210,0.45));
        animation: logoPulse 3s ease infinite;
    }
    @keyframes logoPulse {
        0%, 100% { filter: drop-shadow(0 0 8px rgba(200,200,210,0.25)); }
        50%      { filter: drop-shadow(0 0 18px rgba(220,220,230,0.55)); }
    }
    .tagline { color: #8a8f98; font-size: 0.85rem; }

    .step-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        background: rgba(255,255,255,0.03);
        transition: all 0.4s ease;
        animation: fadeInUp 0.6s ease both;
    }
    .step-card:hover {
        transform: translateX(6px);
        border-color: rgba(255,255,255,0.35);
        box-shadow: 0 0 18px rgba(200,200,210,0.12);
    }
    .step-card.done {
        border-color: #9ca3af;
        background: rgba(255,255,255,0.06);
        animation: glowPulse 2.5s ease infinite;
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 8px rgba(200,200,210,0.10); }
        50%      { box-shadow: 0 0 20px rgba(200,200,210,0.28); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .step-title { font-weight: 700; color: #e5e7eb; letter-spacing: 0.03em; }
    .status-done    { color: #d1d5db; font-size: 0.8rem; font-weight: 700; }
    .status-waiting { color: #6b7280; font-size: 0.8rem; }
    .step-desc { font-size: 0.78rem; color: #8a8f98; margin-top: 2px; }

    .hist-card {
        border: 1px solid rgba(255,255,255,0.14);
        border-left: 4px solid #9ca3af;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        background: rgba(255,255,255,0.04);
        font-family: monospace;
        font-size: 0.8rem;
        color: #e5e7eb;
        animation: fadeInUp 0.5s ease both;
        transition: all 0.3s ease;
    }
    .hist-card:hover {
        background: rgba(255,255,255,0.08);
        box-shadow: 0 0 14px rgba(200,200,210,0.15);
    }
    .hist-time { color: #6b7280; font-size: 0.7rem; }

    .stButton > button {
        background: linear-gradient(90deg, #4b5563, #9ca3af) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 22px rgba(200,200,210,0.45) !important;
        transform: scale(1.02) !important;
    }
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #f3f4f6 !important;
        border-radius: 10px !important;
    }
    .stTextInput input:focus {
        border-color: #d1d5db !important;
        box-shadow: 0 0 12px rgba(200,200,210,0.25) !important;
    }
    .result-block { animation: fadeInUp 0.8s ease both; }
</style>
""", unsafe_allow_html=True)


# ── AI LOGO (neural network SVG) ─────────────────────────────
AI_LOGO = """
<svg class="logo-svg" width="64" height="64" viewBox="0 0 100 100" fill="none">
  <defs>
    <radialGradient id="ng" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#6b7280"/>
    </radialGradient>
  </defs>
  <circle cx="50" cy="50" r="44" stroke="#4b5563" stroke-width="2" fill="rgba(255,255,255,0.02)"/>
  <line x1="50" y1="26" x2="50" y2="50" stroke="#9ca3af" stroke-width="2"/>
  <line x1="26" y1="62" x2="50" y2="50" stroke="#9ca3af" stroke-width="2"/>
  <line x1="74" y1="62" x2="50" y2="50" stroke="#9ca3af" stroke-width="2"/>
  <line x1="26" y1="62" x2="74" y2="62" stroke="#6b7280" stroke-width="1.5"/>
  <circle cx="50" cy="26" r="6" fill="url(#ng)"/>
  <circle cx="26" cy="62" r="6" fill="url(#ng)"/>
  <circle cx="74" cy="62" r="6" fill="url(#ng)"/>
  <circle cx="50" cy="50" r="9" fill="#e5e7eb"/>
</svg>
"""


# ── Helpers ───────────────────────────────────────────────────
def extract_urls(text: str, max_urls: int = 3) -> list:
    urls = re.findall(r"https?://[^\s\"')\],]+", text)
    clean = []
    for u in urls:
        if not u.lower().endswith(".pdf") and u not in clean:
            clean.append(u)
        if len(clean) == max_urls:
            break
    return clean


def safe_run(func, step_name: str, wait: int = 20):
    try:
        return func()
    except Exception:
        st.warning(f"{step_name} rate-limited. Waiting {wait}s...")
        time.sleep(wait)
        return func()


def step_card(num, title, state, desc=""):
    label = {"done": "DONE", "waiting": "WAITING"}.get(state, "")
    cls = "done" if state == "done" else ""
    st.markdown(
        f'<div class="step-card {cls}">'
        f'<span class="step-title">{num} · {title}</span> '
        f'<span class="status-{state}">{label}</span>'
        f'<div class="step-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )


for key, default in (("logged_in", False), ("username", ""), ("results", {})):
    if key not in st.session_state:
        st.session_state[key] = default


# ── AUTH ─────────────────────────────────────────────────────
def show_auth():
    st.markdown(
        f'<div class="logo-row">{AI_LOGO}'
        f'<div><h1>Multi AI Agent System</h1>'
        f'<div class="tagline">Please log in to continue</div></div></div>',
        unsafe_allow_html=True,
    )

    tab_login, tab_signup, tab_guest = st.tabs(["Login", "Sign Up", "Guest"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            ok, msg = login(u, p)
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        with st.form("signup_form"):
            u = st.text_input("Choose a username")
            p = st.text_input("Choose a password", type="password")
            p2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if p != p2:
                st.error("Passwords do not match.")
            else:
                ok, msg = sign_up(u, p)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with tab_guest:
        st.info("Continue without an account. Reports saved under 'guest'.")
        if st.button("Enter as Guest", use_container_width=True, key="guest_btn"):
            st.session_state.logged_in = True
            st.session_state.username = "guest"
            st.rerun()


if not st.session_state.logged_in:
    show_auth()
    st.stop()


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {st.session_state.username}")
    if st.button("Logout", use_container_width=True, key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.results = {}
        st.rerun()

    st.divider()
    st.markdown("### Report History")

    hist = load_history(st.session_state.username)
    if not hist:
        st.caption("No saved reports yet.")

    for i, h in enumerate(hist[:10]):
        st.markdown(
            f'<div class="hist-card">'
            f'{h["topic"][:40]}<br>'
            f'<span class="hist-time">{h["time"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"View {i + 1}", expanded=False):
            st.markdown(h["report"])
            if h.get("feedback"):
                st.markdown("**Critic:**")
                st.markdown(h["feedback"])


# ── HEADER WITH LOGO ─────────────────────────────────────────
st.markdown(
    f'<div class="logo-row">{AI_LOGO}'
    f'<div><h1>Multi AI Agent System</h1>'
    f'<div class="tagline">Fast Chat + Full Research | Powered by Gemini and Groq</div></div></div>',
    unsafe_allow_html=True,
)

mode = st.radio(
    "Select Mode:",
    ["Fast Response (1-3 Seconds)",
     "Full Research Pipeline (Deep Search & Critic)"],
    horizontal=True,
    key="mode_radio",
)

col_input, col_pipeline = st.columns([5, 4])

with col_input:
    topic = st.text_input(
        "Enter Research Topic or Question:",
        placeholder="e.g. best skill in 2026",
        key="topic_input",
    )
    # RENAMED BUTTON
    run_btn = st.button("Run Agent", use_container_width=True, key="run_btn")

with col_pipeline:
    r = st.session_state.results
    if mode.startswith("Fast"):
        step_card("01", "Fast AI Response", "done" if "writer" in r else "waiting",
                  "Direct structured response")
    else:
        step_card("01", "Search", "done" if "search" in r else "waiting", "Web search")
        step_card("02", "Reader", "done" if "reader" in r else "waiting", "Scrape source")
        step_card("03", "Writer", "done" if "writer" in r else "waiting", "Generate report")
        step_card("04", "Critic", "done" if "critic" in r else "waiting", "Review report")


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    results = {}
    start = time.time()

    if mode.startswith("Fast"):
        with st.spinner("Agent is thinking..."):
            results["writer"] = safe_run(
                lambda: fast_chain.invoke({"topic": topic}), "Fast Chat"
            )
        results["critic"] = ""
    else:
        with st.spinner("Searching the web..."):
            results["search"] = web_search.invoke(topic)
        st.session_state.results = dict(results)

        with st.spinner("Scraping source..."):
            scraped = "No page scraped."
            for url in extract_urls(results["search"]):
                out = scrape_url.invoke(url)
                if not out.startswith("Could not scrape") and not out.startswith("Skipped"):
                    scraped = out
                    break
        results["reader"] = scraped
        st.session_state.results = dict(results)

        with st.spinner("Writing report..."):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = safe_run(
                lambda: writer_chain.invoke({"topic": topic, "research": research_combined}),
                "Writer"
            )
        st.session_state.results = dict(results)

        with st.spinner("Critic reviewing..."):
            results["critic"] = safe_run(
                lambda: critic_chain.invoke({"report": results["writer"]}),
                "Critic"
            )

    elapsed = round(time.time() - start, 2)
    st.session_state.results = dict(results)
    save_report(st.session_state.username, topic, results["writer"], results.get("critic", ""))
    st.success(f"Done in {elapsed} seconds. Saved to history.")
    st.rerun()


r = st.session_state.results

if r:
    st.divider()
    st.markdown('<div class="result-block">', unsafe_allow_html=True)

    if "search" in r:
        with st.expander("Search Results (raw)", expanded=False):
            st.text(r["search"][:4000])

    if "reader" in r:
        with st.expander("Scraped Content (raw)", expanded=False):
            st.text(r["reader"][:4000])

    if "writer" in r:
        st.markdown("### AI Response")
        st.markdown(r["writer"])
        st.download_button(
            "Download (.md)",
            data=r["writer"],
            file_name=f"response_{int(time.time())}.md",
            mime="text/markdown",
            key="download_btn",
        )

    if r.get("critic"):
        st.markdown("### Critic Feedback")
        st.markdown(r["critic"])

    st.markdown('</div>', unsafe_allow_html=True)
