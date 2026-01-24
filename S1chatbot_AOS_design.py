# =========================
# Style Loom Chatbot Experiment
# Dropdown + Auto Switch (Option C) + KB-grounded Answers (LangChain) + GPT Fallback
# Supabase (REQUIRED): save ONLY at end (sessions + transcripts + satisfaction)
#
# Folder requirement:
#   ./data/  (md/json knowledge files)
#
# Streamlit Secrets required:
#   OPENAI_API_KEY
#   SUPABASE_URL
#   SUPABASE_ANON_KEY
#
# Supabase tables (must exist):
#   public.sessions(session_id, ts_start, ts_end, identity_option, brand_type,
#                   name_present, picture_present, scenario, user_turns, bot_turns)
#   public.transcripts(id, session_id, ts, transcript_text)
#   public.satisfaction(id, session_id, ts, rating)
# =========================

import os
import re
import uuid
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import streamlit as st
from openai import OpenAI
from supabase import create_client  # Supabase is REQUIRED

# LangChain / Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Style Loom Chatbot Experiment", layout="centered")


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# -------------------------
# Experiment constants
# -------------------------
MODEL_CHAT = "gpt-4o-mini"
MODEL_EMBED = "text-embedding-3-small"
MIN_USER_TURNS = 5

# Supabase tables (match your SQL exactly)
TBL_SESSIONS = "sessions"
TBL_TRANSCRIPTS = "transcripts"
TBL_SATISFACTION = "satisfaction"


# -------------------------
# OpenAI client
# -------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Please configure it in environment variables or st.secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)


# -------------------------
# Supabase client (REQUIRED)
# -------------------------
SUPA_URL = st.secrets.get("SUPABASE_URL", None)
SUPA_KEY = st.secrets.get("SUPABASE_ANON_KEY", None)

if not SUPA_URL or not SUPA_KEY:
    st.error("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY in st.secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_supabase():
    return create_client(SUPA_URL, SUPA_KEY)

supabase = get_supabase()


# -------------------------
# Session state
# -------------------------
defaults = {
    "chat_history": [],                       # List[Tuple[speaker, message]]
    "session_id": uuid.uuid4().hex[:10],
    "greeted_once": False,
    "ended": False,
    "rating_saved": False,
    "user_turns": 0,
    "bot_turns": 0,
    "last_user_selected_scenario": "— Select a scenario —",
    "active_scenario": None,                  # scenario used to answer (can auto-switch)
    "switch_log": [],                         # list of dicts
    "session_started_logged": False,           # ensure sessions.ts_start is logged once
    "last_kb_context": "",                     # for follow-up continuity
    "last_intent_used": None,                  # for follow-up continuity
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------------
# Branding and identity (Study condition)
# -------------------------
identity_option = "With name and image"
show_name = True
show_picture = True
CHATBOT_NAME = "Skyler"
CHATBOT_PICTURE = "https://i.imgur.com/4uLz4FZ.png"

# IMPORTANT: keep this consistent with your greeting manipulation
brand_type = "Mass-market Brand"

def chatbot_speaker() -> str:
    return CHATBOT_NAME if show_name else "Assistant"


# -------------------------
# Header UI (only place where image/name appear; chat itself is text-only)
# -------------------------
st.markdown(
    "<div style='display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;'>"
    "<div style='font-weight:700;font-size:20px;letter-spacing:0.3px;'>Style Loom</div>"
    "</div>",
    unsafe_allow_html=True,
)
if show_picture:
    try:
        st.image(CHATBOT_PICTURE, width=84)
    except Exception:
        pass


# -------------------------
# Scenarios (dropdown)
# -------------------------
SCENARIOS = [
    "— Select a scenario —",
    "Check product availability",
    "Shipping & returns",
    "Size & fit guidance",
    "New arrivals & collections",
    "Rewards & membership",
    "Discounts & promotions",
    "About the brand",
    "Other",
]

# Map scenario to intent key (internal)
SCENARIO_TO_INTENT = {
    "Check product availability": "availability",
    "Shipping & returns": "shipping_returns",
    "Size & fit guidance": "size_fit",
    "New arrivals & collections": "new_arrivals",
    "Rewards & membership": "rewards",
    "Discounts & promotions": "promotions",
    "About the brand": "about",
    "Other": "other",
    "— Select a scenario —": "none",
}

# Map intent to knowledge files (deterministic fallback loads these files)
INTENT_TO_FILES = {
    "availability": [
        "availability_playbook.md",
        "availability_rules.md",
        "inventory_schema.json",
        "mens_and_womens_catalog.md",
    ],
    "shipping_returns": [
        "shipping_returns.md",
        "free_returns_policy.md",
    ],
    "size_fit": [
        "size_chart.md",
        "vocab.md",
    ],
    "new_arrivals": [
        "new_drop.md",
        "current.md",
    ],
    "rewards": [
        "rewards.md",
    ],
    "promotions": [
        "current.md",
        "promotions_rules.md",
        "price_policy_and_ranges.md",
    ],
    "about": [
        "about.md",
    ],
}

FILE_TO_INTENT: Dict[str, str] = {}
for intent_key, files in INTENT_TO_FILES.items():
    for fn in files:
        FILE_TO_INTENT[fn] = intent_key

def scenario_to_intent(scenario: Optional[str]) -> str:
    if not scenario:
        return "none"
    return SCENARIO_TO_INTENT.get(scenario, "other")


# -------------------------
# Intent detection (rule-based) for auto-switching (Option C) - ENGLISH ONLY
# -------------------------
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "new_arrivals": [
        "new drop", "new arrivals", "new arrival", "new collection", "latest", "this season",
    ],
    "size_fit": [
        "size", "sizing", "fit", "measurement", "measurements", "bust", "waist", "hip",
        "xs", "xl", "cm", "inch",
    ],
    "shipping_returns": [
        "shipping", "delivery", "return", "returns", "exchange", "refund", "ship",
    ],
    "promotions": [
        "discount", "promo", "promotion", "coupon", "code", "sale", "deal",
    ],
    "rewards": [
        "reward", "rewards", "points", "membership", "tier", "vip",
    ],
    "availability": [
        "available", "availability", "in stock", "out of stock", "restock", "sold out", "inventory",
    ],
    "about": [
        "about", "brand", "story", "who are you", "who is", "ceo",
    ],
}

INTENT_TO_SCENARIO = {
    "availability": "Check product availability",
    "shipping_returns": "Shipping & returns",
    "size_fit": "Size & fit guidance",
    "new_arrivals": "New arrivals & collections",
    "rewards": "Rewards & membership",
    "promotions": "Discounts & promotions",
    "about": "About the brand",
}

def detect_intent(user_text: str) -> Optional[str]:
    t = (user_text or "").strip().lower()
    if not t:
        return None
    t = re.sub(r"\s+", " ", t)

    best_intent = None
    best_score = 0
    for intent_key, kws in INTENT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score = score
            best_intent = intent_key
    return best_intent if best_score >= 1 else None


# -------------------------
# Knowledge base loader (LangChain)
# -------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(data_dir: Path) -> Optional[Chroma]:
    if not data_dir.exists():
        return None

    docs = []
    md_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=False,
    )
    docs.extend(md_loader.load())

    json_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.json",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=False,
    )
    docs.extend(json_loader.load())

    for d in docs:
        src = d.metadata.get("source", "")
        name = os.path.basename(src)
        d.metadata["intent"] = FILE_TO_INTENT.get(name, "general")
        d.metadata["filename"] = name

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=MODEL_EMBED, openai_api_key=API_KEY)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="styleloom_kb",
    )

vectorstore = build_vectorstore(DATA_DIR)

def retrieve_context(
    query: str,
    intent_key: Optional[str],
    k: int = 8,
    min_score: float = 0.25,
) -> str:
    if not vectorstore:
        return ""

    filt = None
    if intent_key and intent_key not in ("none", "other"):
        filt = {"intent": intent_key}

    # Prefer relevance-scored search when available
    try:
        hits = vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=filt)
        filtered = [(d, s) for (d, s) in hits if s is not None and s >= min_score]
        if not filtered:
            return ""
        blocks = []
        for i, (d, s) in enumerate(filtered, 1):
            fn = d.metadata.get("filename", "unknown")
            blocks.append(f"[Doc{i} score={s:.2f} file={fn}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)
    except Exception:
        try:
            hits = vectorstore.similarity_search(query, k=k, filter=filt)
        except Exception:
            hits = vectorstore.similarity_search(query, k=k)

        if not hits:
            return ""
        blocks = []
        for i, d in enumerate(hits, 1):
            fn = d.metadata.get("filename", "unknown")
            blocks.append(f"[Doc{i} file={fn}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)


# -------------------------
# Deterministic fallback context loader (scenario files)
# -------------------------
FOLLOWUP_ACK_PAT = re.compile(
    r"^(sure|yes|yeah|yep|ok|okay|go ahead|please do|do it|sounds good|tell me|show me)\b",
    re.IGNORECASE,
)

def is_generic_followup(text: str) -> bool:
    t = (text or "").strip()
    # short confirmations are common in multi-turn; treat as follow-up
    return (len(t) <= 18) and bool(FOLLOWUP_ACK_PAT.search(t))

def load_intent_files_as_context(intent_key: str) -> str:
    files = INTENT_TO_FILES.get(intent_key, [])
    if not files:
        return ""
    blocks = []
    for fn in files:
        fp = DATA_DIR / fn
        if fp.exists():
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                content = ""
            if content:
                blocks.append(f"[FILE: {fn}]\n{content}")
    return "\n\n".join(blocks)


# -------------------------
# LLM helpers
# -------------------------
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()

def answer_grounded(user_text: str, context: str, intent_key: Optional[str] = None) -> str:
    system = f"""You are Style Loom's virtual assistant for a fashion retail study.
You MUST use the BUSINESS CONTEXT below as your source of truth.

Rules:
- If the user asks about a policy (e.g., returns, shipping, discounts, rewards), provide a concise, concrete summary first.
- Include key constraints (e.g., time window, condition requirements, eligibility, processing time) if present in the context.
- Do NOT ask a follow-up question if the context already contains enough information to answer.
- If the context truly does not contain the needed specifics, ask ONE concise follow-up question.

Keep the answer short, direct, and helpful.
Intent: {intent_key or "unknown"}.
"""
    msgs = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"BUSINESS CONTEXT:\n{context}"},
        {"role": "user", "content": user_text},
    ]
    return llm_chat(msgs, temperature=0.2)

def answer_fallback(user_text: str) -> str:
    system = """You are Style Loom's virtual assistant.
If the user asks for exact inventory counts, exact prices, or strict policy exceptions, do not guess.
Ask one concise follow-up question or provide general guidance.
Keep the response brief and natural."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    return llm_chat(msgs, temperature=0.5)

def generate_answer(user_text: str, scenario: Optional[str]) -> Tuple[str, str, bool]:
    """
    Returns (answer, used_intent, used_kb).

    Fixes:
    1) Follow-up continuity: if user says "Sure/Yes/Go ahead", reuse last KB context.
    2) Scenario stability: if vector retrieval is weak, load the scenario files directly.
    """
    intent_key = scenario_to_intent(scenario)

    # Follow-up: reuse prior context
    if is_generic_followup(user_text) and st.session_state.get("last_kb_context", "").strip():
        ctx = st.session_state["last_kb_context"]
        used_intent = st.session_state.get("last_intent_used") or intent_key
        ans = answer_grounded(user_text, ctx, intent_key=used_intent)
        return ans, used_intent, True

    # Try vector retrieval scoped to scenario
    context = ""
    used_kb = False

    if vectorstore:
        context = retrieve_context(user_text, intent_key=intent_key, k=8, min_score=0.25)
        if context.strip():
            used_kb = True

    # Deterministic fallback: load all scenario files (prevents "I don't know" loops)
    if not context.strip() and intent_key not in ("none", "other"):
        context = load_intent_files_as_context(intent_key)
        if context.strip():
            used_kb = True

    # If still nothing, use GPT fallback
    if not context.strip():
        st.session_state["last_kb_context"] = ""
        st.session_state["last_intent_used"] = intent_key
        return answer_fallback(user_text), intent_key, False

    ans = answer_grounded(user_text, context, intent_key=intent_key)

    # Persist for next follow-up
    st.session_state["last_kb_context"] = context
    st.session_state["last_intent_used"] = intent_key

    return ans, intent_key, used_kb


# -------------------------
# Supabase: log session start ONCE
# -------------------------
def log_session_start_once():
    if st.session_state.session_started_logged:
        return

    ts_now = datetime.datetime.utcnow().isoformat() + "Z"
    supabase.table(TBL_SESSIONS).upsert({
        "session_id": st.session_state.session_id,
        "ts_start": ts_now,
        "identity_option": identity_option,
        "brand_type": brand_type,
        "name_present": "present" if show_name else "absent",
        "picture_present": "present" if show_picture else "absent",  # matches SQL
    }).execute()

    st.session_state.session_started_logged = True


# -------------------------
# Greeting (first assistant message)
# -------------------------
if not st.session_state.greeted_once:
    log_session_start_once()

    greet_text = (
        "Hi, I'm Skyler, Style Loom’s virtual assistant. "
        "Style Loom is a mass-market fashion brand founded twenty years ago, "
        "known for its accessibility and broad consumer reach. "
        "I’m here to help with your shopping."
    )
    st.session_state.chat_history.append((chatbot_speaker(), greet_text))
    st.session_state.greeted_once = True


# -------------------------
# UI: scenario dropdown (after greeting)
# -------------------------
st.markdown("**How can I help you today?**")

selected = st.selectbox(
    "Choose a topic:",
    options=SCENARIOS,
    index=SCENARIOS.index(st.session_state.last_user_selected_scenario)
    if st.session_state.last_user_selected_scenario in SCENARIOS else 0,
)

# Track last selection
prev_selected = st.session_state.last_user_selected_scenario
st.session_state.last_user_selected_scenario = selected

# One-time confirmation when user selects a category
if selected != "— Select a scenario —" and selected != prev_selected:
    st.session_state.active_scenario = selected
    confirm_text = f"Sure, I will help you with **{selected}**. Please ask me a question."
    st.session_state.chat_history.append((chatbot_speaker(), confirm_text))
    # Optional: uncomment if you want the confirmation to appear immediately before typing
    # st.rerun()

st.divider()


# -------------------------
# Render chat history (NO icons, text-only for experimental control)
# -------------------------
for spk, msg in st.session_state.chat_history:
    if spk == chatbot_speaker():
        st.markdown(f"**{CHATBOT_NAME}:** {msg}")
    else:
        st.markdown(f"**User:** {msg}")


# -------------------------
# Chat input
# -------------------------
user_text = None
if not st.session_state.ended:
    user_text = st.chat_input("Type your message here...")


# -------------------------
# End button and rating UI
# -------------------------
end_col1, end_col2 = st.columns([1, 2])

with end_col1:
    can_end = (st.session_state.user_turns >= MIN_USER_TURNS) and (not st.session_state.ended)
    if st.button("End chat", disabled=not can_end):
        st.session_state.ended = True

with end_col2:
    if st.session_state.user_turns < MIN_USER_TURNS and (not st.session_state.ended):
        st.caption(f"Please complete at least {MIN_USER_TURNS} user turns before ending the chat.")


# -------------------------
# Save ONLY at the end (transcripts + satisfaction + sessions end)
# -------------------------
if st.session_state.ended and not st.session_state.rating_saved:
    rating = st.slider("Overall satisfaction with the chatbot (1 = very low, 7 = very high)", 1, 7, 4)

    if st.button("Submit rating and save"):
        ts_now = datetime.datetime.utcnow().isoformat() + "Z"

        transcript_lines = []
        transcript_lines.append(f"Session ID: {st.session_state.session_id}")
        transcript_lines.append(f"Identity option: {identity_option}")
        transcript_lines.append(f"Brand type: {brand_type}")
        transcript_lines.append(f"Name present: {'present' if show_name else 'absent'}")
        transcript_lines.append(f"Picture present: {'present' if show_picture else 'absent'}")
        transcript_lines.append("---- Switch log ----")
        transcript_lines.append(json.dumps(st.session_state.switch_log, ensure_ascii=False))
        transcript_lines.append("---- Chat transcript ----")
        for spk, msg in st.session_state.chat_history:
            transcript_lines.append(f"{spk}: {msg}")
        transcript_text = "\n".join(transcript_lines)

        # 1) Save transcript
        supabase.table(TBL_TRANSCRIPTS).insert({
            "session_id": st.session_state.session_id,
            "ts": ts_now,
            "transcript_text": transcript_text,
        }).execute()

        # 2) Save rating
        supabase.table(TBL_SATISFACTION).insert({
            "session_id": st.session_state.session_id,
            "ts": ts_now,
            "rating": int(rating),
        }).execute()

        # 3) Update session end + turns + scenario
        supabase.table(TBL_SESSIONS).upsert({
            "session_id": st.session_state.session_id,
            "ts_end": ts_now,
            "scenario": st.session_state.active_scenario or (selected if selected != "— Select a scenario —" else "Other"),
            "user_turns": st.session_state.user_turns,
            "bot_turns": st.session_state.bot_turns,
        }).execute()

        st.session_state.rating_saved = True
        st.success("Saved. Thank you.")


# -------------------------
# Main interaction
# -------------------------
if user_text and not st.session_state.ended:
    st.session_state.chat_history.append(("User", user_text))
    st.session_state.user_turns += 1

    user_selected = selected if selected != "— Select a scenario —" else None
    active = st.session_state.active_scenario or user_selected or "Other"

    detected_intent = detect_intent(user_text)
    detected_scenario = INTENT_TO_SCENARIO.get(detected_intent) if detected_intent else None

    auto_switched = False
    disclosure = ""
    if detected_scenario and (detected_scenario != active):
        auto_switched = True
        disclosure = f"It looks like your question is about {detected_scenario}, so I will answer using that information."
        st.session_state.switch_log.append({
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "user_selected_scenario": user_selected,
            "from_scenario": active,
            "to_scenario": detected_scenario,
            "detected_intent": detected_intent,
            "user_text": user_text,
        })
        active = detected_scenario

    st.session_state.active_scenario = active

    answer, used_intent, used_kb = generate_answer(user_text, scenario=active)
    if auto_switched and disclosure:
        answer = f"{disclosure}\n\n{answer}"

    st.session_state.chat_history.append((chatbot_speaker(), answer))
    st.session_state.bot_turns += 1

    st.rerun()


