# =========================
# Chatbot App with Name and Picture + Start-up — RAG + Rules + Pending Intents
# Natural UI flow (no explicit step sections)
# FULL VERSION (UPDATED): RAG-grounded factual answers + LLM natural responses
# =========================

# --- Imports ---
import os
import re
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
from openai import OpenAI
from supabase import create_client

# LangChain / Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# Streamlit Page Config (place early)
# =========================
st.set_page_config(page_title="Style Loom — Chatbot Experiment", layout="centered")


# =========================
# Session-state initialization (must be above any session_state usage)
# =========================
defaults = {
    "chat_history": [],
    "session_id": uuid.uuid4().hex[:10],     # 새 세션마다 생성
    "awaiting_feedback": False,
    "ended": False,
    "saved_fpath": None,
    "rating_saved": False,
    "greeted_once": False,
    "scenario_selected_once": False,
    "last_user_selected_scenario": "— Select a scenario —",
    "user_turns": 0,
    "bot_turns": 0,
    "closing_asked": False,
    "flow": {
        "scenario": None, "stage": "start",
        "slots": {
            "product": None, "color": None, "size": None,
            "contact_pref": None, "tier_known": None, "selected_collection": None,
            "return_item": None, "received_date": None, "return_reason": None
        }
    },
    "pending": {"intent": None, "data": {}},
    "session_meta_logged": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 최소 대화 턴 수
MIN_USER_TURNS = 5


# =========================
# OpenAI Client
# =========================
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Please configure it in environment variables or st.secrets.")
    st.stop()
client = OpenAI(api_key=API_KEY)


# =========================
# Supabase Client (single, cached)
# =========================
SUPA_URL = st.secrets.get("SUPABASE_URL")
SUPA_KEY = st.secrets.get("SUPABASE_ANON_KEY")

if not SUPA_URL or not SUPA_KEY:
    st.error("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY in st.secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_supabase():
    return create_client(SUPA_URL, SUPA_KEY)

supabase = get_supabase()


# =========================
# Branding (small, logo-like)
# =========================
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;">
        <div style="font-weight:700;font-size:20px;letter-spacing:0.3px;">Style Loom</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Identity (single-condition file)
# =========================
identity_option = "With name and image"
show_name = True
show_picture = True
CHATBOT_NAME = "Skyler"
CHATBOT_PICTURE = "https://i.imgur.com/4uLz4FZ.png"
brand_type = "Start-up Brand"

def _chatbot_speaker():
    return CHATBOT_NAME if show_name else "Assistant"

if show_picture:
    try:
        st.image(CHATBOT_PICTURE, width=84)
    except Exception:
        pass


# =========================
# Initial greeting (appears first in chat)
# =========================
if not st.session_state.greeted_once:
    greet_text = (
        "Hi, I'm Skyler, Style Loom’s virtual assistant. "
        "Style Loom is a start-up fashion brand founded three years ago, "
        "known for its entrepreneurial spirit and innovative approach. "
        "I’m here to help with your shopping."
    )
    st.session_state.chat_history.append((_chatbot_speaker(), greet_text))
    st.session_state.greeted_once = True


# --- Record session meta to Supabase (run once at start) ---
if not st.session_state.session_meta_logged:
    _payload = {
        "session_id": st.session_state.session_id,
        "ts_start": datetime.datetime.utcnow().isoformat() + "Z",
        "identity_option": identity_option,
        "brand_type": brand_type,
        "name_present": "present" if show_name else "absent",
        "picture_present": "present" if show_picture else "absent",
        "scenario": st.session_state.flow.get("scenario") or None,
        "user_turns": st.session_state.user_turns,
        "bot_turns": st.session_state.bot_turns,
    }
    try:
        supabase.table("sessions").insert(_payload).execute()
        st.session_state.session_meta_logged = True
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            st.session_state.session_meta_logged = True
        else:
            st.warning(f"(non-blocking) Failed to insert session meta: {e}")


# =========================
# Tone / Categories
# =========================
TONE = "informal"
TONE_STYLE = {
    # informal: 친근하지만 군더더기 없는 톤, 이모지는 '최대 1개', 문장 맨 끝에만 사용
    "informal": "Use a friendly, concise tone. Use at most one emoji per reply and place it only at the very end when it truly adds warmth. Do not start with 'Hey there'.",
    # formal: 이모지 금지
    "formal": "Use a formal, respectful tone. No emojis."
}

def tone_instruction() -> str:
    return TONE_STYLE.get(TONE, TONE_STYLE["informal"])

PRODUCT_CATEGORIES = [
    "blouse", "skirt", "pants", "cardigans / sweaters", "dresses",
    "jumpsuits", "jackets", "t-shirts", "sweatshirt / sweatpants",
    "outer", "coat / trenches", "tops / bodysuits", "activewear",
    "shirts", "shorts", "lingerie", "etc."
]


# =========================
# Regex & Extractors
# =========================
YES_PAT = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|please)\b", re.I)
NO_PAT  = re.compile(r"\b(no|nope|nah|not now|later)\b", re.I)

def _is_size_chart_query(t: str) -> bool:
    """Detects 'size chart/guide' style questions anywhere in the text."""
    return bool(re.search(
        r"\b(size\s*(chart|guide)|sizing\s*(chart|guide)?|size\s*info|measurement(s)?)\b",
        t or "", re.I
    ))

def _preprocess_user_text(t: str) -> str:
    """Light normalization: common typos, synonyms, season words, and separators."""
    s = (t or "").strip()

    # 공백/슬래시 정리
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)

    # 계절/수식어 제거 (retrieval에 도움되는 경우가 있어 완전 삭제 대신 약하게 정리)
    # 단, user meaning을 깨는 수준의 삭제는 피함.
    seasonals = [
        r"\bnew arrivals?\b", r"\blatest\b", r"\brecent\b"
    ]
    for pat in seasonals:
        s = re.sub(pat, " ", s, flags=re.I)

    # 자주 나오는 오타 보정
    fixes = {
        r"\boatmilk\b": "oatmeal",
        r"\boat meal\b": "oatmeal",
        r"\bgre(y|ie)ge\b": "greige",
    }
    for pat, repl in fixes.items():
        s = re.sub(pat, repl, s, flags=re.I)

    # 대표 제품명 표준화
    s = re.sub(r"\bcity\s+knit\b", "City Knit", s, flags=re.I)
    s = re.sub(r"\bsoft\s+blouse\b", "Soft Blouse", s, flags=re.I)
    s = re.sub(r"\beveryday\s+jacket\b", "Everyday Jacket", s, flags=re.I)
    s = re.sub(r"\btailored\s+pants?\b", "Tailored Pants", s, flags=re.I)
    s = re.sub(r"\bweekend\s+dress\b", "Weekend Dress", s, flags=re.I)

    return s.strip()

def extract_color(t: str):
    m = re.search(
        r"\b(black|white|ivory|navy|blue|mist\s?blue|greige|beige|red|green|rose\s?beige|pink|cream|sand|olive|charcoal|oatmeal|forest|berry|ink|brown|purple|orange|yellow|khaki|teal|burgundy|maroon|grey|gray)\b",
        t or "", re.I
    )
    return m.group(1).lower() if m else None

def extract_size(t: str):
    text = (t or "").lower()
    word_map = {
        r"\b(extra\s*small|x[\- ]?small|xs|xxs)\b": "XS",
        r"\b(small|s)\b": "S",
        r"\b(medium|med|m)\b": "M",
        r"\b(large|l)\b": "L",
        r"\b(extra\s*large|x[\- ]?large|xl)\b": "XL",
        r"\b(xx[\- ]?large|2xl|xxl)\b": "XXL",
    }
    for pat, label in word_map.items():
        if re.search(pat, text, re.I):
            return label
    m = re.search(r"\b(XXS|XS|S|M|L|XL|XXL|0|2|4|6|8|10|12|14|16|18)\b", t or "", re.I)
    return m.group(1).upper() if m else None

def extract_product(t: str):
    text = _preprocess_user_text(t)
    low = text.lower()

    named = ["City Knit", "Soft Blouse", "Everyday Jacket", "Tailored Pants", "Weekend Dress"]
    for name in named:
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            return name

    if "knit" in low:
        if "city" in low:
            return "City Knit"
        return "sweater"

    if "tee" in low or "t-shirt" in low or "tshirt" in low:
        return "t-shirt"

    if re.search(r"\b(running\s+shoes?|sneakers?|shoes?)\b", low, re.I):
        return "shoes"

    cats = [
        "blouse", "skirt", "pants", "cardigan", "cardigans", "sweater", "sweaters",
        "dress", "dresses", "jumpsuit", "jumpsuits", "jacket", "jackets",
        "t-shirt", "t-shirts", "sweatshirt", "sweatpants", "outer", "coat",
        "trench", "trenches", "top", "tops", "bodysuit", "bodysuits",
        "activewear", "shirt", "shirts", "shorts", "lingerie", "shoes"
    ]
    for c in cats:
        if re.search(rf"\b{re.escape(c)}\b", low, re.I):
            if c in ["cardigans", "sweaters", "jackets", "dresses", "tops", "shirts", "jumpsuits"]:
                return c.rstrip("s")
            return c

    w = re.search(r"\b([\w\-]+(?:\s+[\w\-]+)?)\s+(jacket|skirt|blouse|t-?shirt|dress|pants|sweater|shoes)\b", text, re.I)
    if w:
        noun = w.group(2).lower()
        if noun in ("tshirt", "t-shirt"):
            return "t-shirt"
        if noun == "sweater":
            return "sweater"
        return noun

    return None

def _update_slots_from_text(user_text: str):
    cleaned = _preprocess_user_text(user_text)

    slots = st.session_state.flow["slots"]
    p = extract_product(cleaned)
    c = extract_color(cleaned)
    s = extract_size(cleaned)

    if p:
        slots["product"] = p
    if c:
        slots["color"] = c
    if s:
        slots["size"] = s


# =========================
# Close only on end stage
# =========================
def maybe_add_one_time_closing(reply: str) -> str:
    stage = (st.session_state.flow or {}).get("stage")
    if stage == "end_or_more" and (not st.session_state.closing_asked) and (st.session_state.user_turns >= MIN_USER_TURNS - 1):
        st.session_state.closing_asked = True
        return reply + "\n\nIs there anything else I can help you with?"
    return reply


# =========================
# RAG: Build/Load Vectorstore
# =========================
RAG_DIR = str(Path.cwd() / "rag_docs")

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(rag_dir: str):
    rag_path = Path(rag_dir)
    if not rag_path.exists():
        return None

    persist_dir = str(rag_path / ".chroma")
    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small")

    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        try:
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            st.warning(f"Vectorstore load warning: {e}")

    try:
        loader = DirectoryLoader(
            rag_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True,
        )
        try:
            md_loader = DirectoryLoader(
                rag_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                show_progress=True,
                use_multithreading=True,
            )
            docs = loader.load() + md_loader.load()
        except Exception:
            docs = loader.load()
    except Exception as e:
        st.warning(f"RAG documents could not be loaded: {e}")
        return None

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    try:
        return Chroma.from_documents(
            chunks,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    except Exception as e:
        st.warning(f"Vectorstore build failed: {e}")
        return None

vectorstore = build_or_load_vectorstore(RAG_DIR)

# ---- UPDATED: retrieval query is "doc-friendly" + optional hint ----
def make_query(user_message: str, retrieval_hint: Optional[str] = None) -> str:
    """
    Retrieval query should be document-friendly.
    Keep it close to natural language + salient slot values + optional hint.
    Avoid long meta fields (scenario:, intent:, etc.) that add noise.
    """
    slots = st.session_state.flow.get("slots", {})
    text = _preprocess_user_text(user_message)

    parts = [text]
    if slots.get("product"):
        parts.append(str(slots["product"]))
    if slots.get("color"):
        parts.append(str(slots["color"]))
    if slots.get("size"):
        parts.append(str(slots["size"]))

    if retrieval_hint:
        parts.append(retrieval_hint)

    # mild signal for collection queries
    if re.search(r"\b(new|latest|collection|arrivals?)\b", text, re.I):
        parts.append("collection")

    return " ".join([p for p in parts if p]).strip()

def retrieve_context(query: str, k: int = 6) -> str:
    if not vectorstore:
        return ""
    try:
        hits = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        st.warning(f"Similarity search failed: {e}")
        return ""
    if not hits:
        return ""

    blocks = []
    for i, d in enumerate(hits, 1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[Doc{i} from {os.path.basename(src)}]\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


# =========================
# LLM (RAG) — UPDATED: RAG-first rules + conversation context
# =========================
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+")

def apply_tone_policies(text: str) -> str:
    """
    Tone policy:
    - formal: remove all emojis
    - informal: keep at most one emoji and move it to the very end
    """
    if not isinstance(text, str) or not text:
        return text

    if TONE == "formal":
        return _EMOJI_RE.sub("", text)

    emojis = list(_EMOJI_RE.finditer(text))
    if not emojis:
        return text

    first_emoji = emojis[0].group(0)
    cleaned = _EMOJI_RE.sub("", text).rstrip()

    if cleaned and cleaned[-1] in ".!?":
        return f"{cleaned} {first_emoji}"
    return f"{cleaned} {first_emoji}"

def _recent_messages_for_llm(user_message: str, max_pairs: int = 6) -> List[Dict[str, str]]:
    """
    Include a small recent window of conversation for coherence,
    while avoiding duplication of the current user message.
    """
    hist: List[Tuple[str, str]] = st.session_state.chat_history or []

    # If the last entry is the same user message, exclude it to avoid duplication
    if hist and hist[-1][0] == "User" and (hist[-1][1] or "").strip() == (user_message or "").strip():
        hist = hist[:-1]

    # Take last N messages (approx pairs)
    window = hist[-(max_pairs * 2):] if len(hist) > max_pairs * 2 else hist

    msgs: List[Dict[str, str]] = []
    for spk, msg in window:
        if spk == "User":
            msgs.append({"role": "user", "content": msg})
        else:
            # any bot speaker -> assistant
            msgs.append({"role": "assistant", "content": msg})
    return msgs

def answer_with_rag(user_message: str, retrieval_hint: Optional[str] = None) -> str:
    _update_slots_from_text(user_message)

    query = make_query(user_message, retrieval_hint=retrieval_hint)
    context = retrieve_context(query, k=6)
    has_context = bool((context or "").strip())

    style_instruction = tone_instruction()

    flow = st.session_state.flow or {}
    known_slots = flow.get("slots", {})
    current_scenario = flow.get("scenario")
    current_stage = flow.get("stage")

    # Identity string for experimental condition
    bot_identity = f"a customer service chatbot named {CHATBOT_NAME}" if show_name else "a customer service chatbot"

    # RAG-first rules: prevents “막힘” and forces summarization when context exists
    system_rules = (
        "You are a helpful customer service chatbot for Style Loom.\n\n"
        "CRITICAL RULES (follow in order):\n"
        "1) If the user asks about products, colors, collections, sizing, shipping, returns, rewards, promotions, pricing, or brand offerings, "
        "you MUST first use the BUSINESS CONTEXT provided.\n"
        "2) If the BUSINESS CONTEXT contains relevant information, answer directly by summarizing it in your own words (do not quote long passages).\n"
        "3) If only partial information is available, provide what is known and then ask ONE concise follow-up question.\n"
        "4) Ask ONE follow-up question ONLY when the BUSINESS CONTEXT truly lacks the needed information.\n"
        "5) Do NOT invent policy details, numbers, SKUs, prices, or inventory counts.\n"
        "6) Keep the answer short, natural, and helpful.\n"
    )

    meta_block = (
        f"Meta:\n"
        f"- Current scenario: {current_scenario}\n"
        f"- Current stage: {current_stage}\n"
        f"- Known slots: {known_slots}\n"
        f"- Product categories: {', '.join(PRODUCT_CATEGORIES)}\n"
    )

    context_block = context if has_context else (
        "[No document snippets were retrieved for this query. "
        "Provide only general guidance and ask one concise follow-up question if needed.]"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_rules},
        {"role": "system", "content": meta_block},
        {"role": "system", "content": f"Style:\n{style_instruction}"},
        {"role": "system", "content": f"BUSINESS CONTEXT (retrieved):\n{context_block}"},
    ]
    messages += _recent_messages_for_llm(user_message, max_pairs=6)
    messages += [{"role": "user", "content": user_message}]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            top_p=1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"LLM call failed: {e}")
        return "Sorry, I had trouble generating a response. Could you rephrase your question?"

def llm_fallback(user_message: str) -> str:
    # Generic fallback uses no extra retrieval hint
    return answer_with_rag(user_message)


# =========================
# Pending yes/no logic (global)
# =========================
def set_pending(intent: str, data: dict | None = None):
    st.session_state.pending = {"intent": intent, "data": (data or {})}

def consume_pending():
    p = st.session_state.pending
    st.session_state.pending = {"intent": None, "data": {}}
    return p

def ask_yesno(intent: str, message: str, data: dict | None = None) -> str:
    set_pending(intent, data or {})
    return message

def handle_pending_yes(user_text: str) -> str | None:
    pend = st.session_state.pending
    intent = pend.get("intent")
    if not intent:
        return None

    # YES branch
    if YES_PAT.search(user_text):
        if intent == "rewards_more":
            consume_pending()
            # Use RAG to answer details (not hardcoded). Provide retrieval hint.
            return answer_with_rag(
                "Please explain how the rewards/membership program works: earning, tiers, redemption, expiration, and shipping perks, based on policy.",
                retrieval_hint="rewards membership points tiers redemption expiration free express shipping"
            )

        if intent == "colors_sizes_more":
            slots = st.session_state.flow["slots"]
            product = slots.get("product") or "item"
            color = slots.get("color")
            size  = slots.get("size")
            consume_pending()
            q = f"Please describe available colors and sizes for {product}"
            if color:
                q += f" in {color}"
            if size:
                q += f" size {size}"
            q += ", and how to check availability."
            return answer_with_rag(q, retrieval_hint="availability colors sizes inventory how to check")

        if intent == "confirm_switch":
            pend_consumed = consume_pending()
            target = pend_consumed["data"].get("target")
            if target:
                st.session_state.flow = {
                    "scenario": target, "stage": "start",
                    "slots": { **st.session_state.flow["slots"] }
                }
                return f"Great—switching to **{target}**. How can I help within this topic?"
            return "Okay—switching contexts."

        consume_pending()
        return "Got it."

    # NO branch
    if NO_PAT.search(user_text):
        if intent == "confirm_switch":
            consume_pending()
            return "No problem—let’s continue with the current topic."
        consume_pending()
        return "All set! If you want the details later, just ask."

    return None


# =========================
# Auto-pending inference (scenario-aware)
# =========================
def infer_pending_from_bot_reply(reply_text: str) -> None:
    sc = (st.session_state.flow.get("scenario") or "").strip()
    text = (reply_text or "").strip().lower()
    if not text:
        return

    def _match_any(patterns):
        return any(re.search(p, text, re.I) for p in patterns)

    if sc == "Rewards & membership":
        if _match_any([
            r"\bwould you like\b",
            r"\bdo you want\b",
        ]):
            set_pending("rewards_more")
            return

    if sc in ("Check product availability", "Size & fit guidance", "New arrivals & collections"):
        if _match_any([
            r"\bwould you like\b",
            r"\bdo you want\b",
        ]):
            set_pending("colors_sizes_more")
            return
    return


# =========================
# Shipping intent detector
# =========================
try:
    _is_shipping_query
except NameError:
    def _is_shipping_query(t: str) -> bool:
        """Detect shipping/delivery questions (exclude 'free return(s)' and 'return shipping')."""
        text = (t or "")
        if re.search(r"\bfree\s+return(s)?(\s+shipping)?\b", text, re.I):
            return False
        if re.search(r"\breturn\s+shipping\b", text, re.I):
            return False
        return bool(re.search(
            r"\b(ship|shipping|deliver(y|ed|ing)?|eta|track(ing)?|when\s+will\s+it\s+(arrive|be\s+delivered)|how\s+long.*(deliver|shipping|arrive))\b",
            text, re.I
        ))


# =========================
# Helpers for pattern matching & avoiding repetition
# =========================
def _any(patterns, text):
    return any(re.search(p, text or "", re.I) for p in patterns)

def maybe_dedupe_reply(reply: str) -> str:
    hist = st.session_state.chat_history or []
    last_bot = next((m for (spk, m) in reversed(hist) if spk == _chatbot_speaker()), None)
    if last_bot and last_bot.strip() == reply.strip():
        return reply + "\n\nWould you like to narrow by a category or color?"
    return reply


# =========================
# Rule-based scenario router (UPDATED: facts answered via RAG)
# =========================
def route_by_scenario(current_scenario: str, user_text: str) -> str | None:
    flow = st.session_state.flow
    slots = flow["slots"]
    stage = flow.get("stage") or "start"

    _update_slots_from_text(user_text)

    # ---- About the brand ----
    if current_scenario == "About the brand":
        flow["stage"] = "end_or_more"
        return answer_with_rag(
            user_text,
            retrieval_hint="brand story about style loom start-up founded mission positioning values"
        )

    # ---- Rewards & membership ----
    if current_scenario == "Rewards & membership":
        if stage in (None, "start"):
            flow["stage"] = "rewards_intro"
            # Provide short helpful start + offer details
            intro = answer_with_rag(
                "Give a brief overview of the rewards/membership program and ask if the user wants earning or redemption details.",
                retrieval_hint="rewards membership overview points tiers redeem expiration"
            )
            return ask_yesno("rewards_more", intro)

        # Any follow-up in this scenario -> use RAG
        flow["stage"] = "end_or_more"
        return answer_with_rag(
            user_text,
            retrieval_hint="rewards membership points tiers redeem expiration free express shipping"
        )

    # ---- Discounts & promotions ----
    if current_scenario == "Discounts & promotions":
        flow["stage"] = "end_or_more"
        return answer_with_rag(
            user_text,
            retrieval_hint="promotions discounts coupon codes exclusions stacking order of operations"
        )

    # ---- New arrivals & collections ----
    if current_scenario == "New arrivals & collections":
        text = _preprocess_user_text(user_text)

        if stage in (None, "start"):
            flow["stage"] = "new_intro"
            return "Looking for a category or a color from the new collection?"

        # Use RAG to answer lineup/colors; keep natural and ask 1 follow-up if needed
        flow["stage"] = "end_or_more"
        return maybe_dedupe_reply(
            answer_with_rag(
                text,
                retrieval_hint="new arrivals collection lineup items colors categories"
            )
        )

    # ---- Check product availability ----
    if current_scenario == "Check product availability":
        if stage == "start":
            flow["stage"] = "collect"
            stage = "collect"

        if stage == "collect":
            if not slots.get("product"):
                return "Sure—what product are you looking for (e.g., jacket, dress, t-shirt)?"
            if not slots.get("color"):
                return f"Great—what color of {slots['product']}?"
            if not slots.get("size"):
                return f"What size for the {slots['product']} in {slots['color']}?"

            # Now we have enough; use RAG to respond naturally (avoid inventing exact counts)
            flow["stage"] = "end_or_more"
            q = (
                f"The user wants to check availability for {slots['product']} in {slots['color']} size {slots['size']}. "
                f"Based on inventory/availability policy, explain how to confirm stock and suggest what to do if low stock."
            )
            return answer_with_rag(q, retrieval_hint="availability in stock low stock restock alert variants")

        if stage == "end_or_more":
            return "Happy to help. Anything else you’d like to check?"
        return None

    # ---- Size & fit guidance ----
    if current_scenario == "Size & fit guidance":
        if _is_size_chart_query(user_text):
            flow["stage"] = "end_or_more"
            # Pull size guide info from RAG docs rather than hardcoded chart
            return answer_with_rag(
                "Please provide the size guide/size chart information and where to find it on the product page.",
                retrieval_hint="size guide size chart measurements bust waist hip inseam fit notes"
            )

        if stage in (None, "start"):
            flow["stage"] = "fit_collect"
            return "Sure—tell me your current size and how it fits."

        if stage == "fit_collect":
            # Let LLM handle nuanced sizing advice grounded in docs, using the user's message + slots
            flow["stage"] = "end_or_more"
            return answer_with_rag(
                user_text,
                retrieval_hint="size fit guidance runs small runs large true to size recommendations"
            )

        if stage == "end_or_more":
            return "Happy to help. Anything else I can assist you with?"
        return None

    # ---- Shipping & returns ----
    if current_scenario == "Shipping & returns":
        # shipping question detection prioritized
        if _is_shipping_query(user_text):
            flow["stage"] = "end_or_more"
            return answer_with_rag(
                user_text,
                retrieval_hint="shipping delivery processing time standard express international tracking eta"
            )

        # Return flow (keep structure, but generate policy/instructions via RAG)
        if stage in (None, "start"):
            flow["stage"] = "returns_collect_item"
            return "Of course! I can help with a return. What item would you like to return?"

        if stage == "returns_collect_item":
            if user_text.strip():
                flow["slots"]["return_item"] = user_text.strip()
            flow["stage"] = "returns_collect_date"
            return "Got it. When did you receive the item? (Please provide a date like 2025-09-10)"

        if stage == "returns_collect_date":
            m = re.search(r"\b(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\b", user_text)
            flow["slots"]["received_date"] = m.group(1) if m else "unknown"
            flow["stage"] = "returns_condition_check"
            # Ask condition check; policy details should come from RAG when needed
            return "Thanks. Can you confirm the item is unworn and in original condition? (yes/no)"

        if stage == "returns_condition_check":
            if YES_PAT.search(user_text):
                flow["stage"] = "returns_reason"
                return "Understood. Could you tell me the reason for the return? (e.g., too small, defective, changed mind)"
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return answer_with_rag(
                    "Explain return eligibility requirements and alternatives (exchange/repair) based on policy.",
                    retrieval_hint="returns eligibility unworn original condition exchange repair options"
                )
            return "Please reply yes or no: is the item unworn and in original condition?"

        if stage == "returns_reason":
            flow["slots"]["return_reason"] = user_text.strip()
            flow["stage"] = "returns_instructions"
            item = flow["slots"].get("return_item", "the item")
            received = flow["slots"].get("received_date", "unknown")
            reason = flow["slots"].get("return_reason", "not specified")

            # Generate instructions via RAG (avoid hardcoding windows/fees)
            q = (
                f"Create return instructions for {item}. Received date: {received}. Reason: {reason}. "
                f"Include the correct return window, label process, and refund timing based on policy. "
                f"End by asking if the user wants the return label emailed now (yes/no)."
            )
            return answer_with_rag(q, retrieval_hint="returns steps label prepaid refund timing return window")

        if stage == "returns_instructions":
            if YES_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return "Great—your return label request is noted. You’ll receive it shortly. Anything else I can help with?"
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return "No problem. If you need the label later, just ask. Anything else I can help with?"
            return "Would you like me to email the return label now? (yes/no)"

        if stage == "end_or_more":
            return "Happy to help. Anything else I can assist you with?"
        return None

    # ---- Other ----
    if current_scenario in ("Other", None, ""):
        # Use RAG + natural LLM for broad questions
        flow["stage"] = "end_or_more"
        return answer_with_rag(user_text, retrieval_hint="general help catalog policies")

    return None


# =========================
# Global intent detection (cross-scenario)
# =========================
GLOBAL_INTENTS = [
    (r"\b(new\s+arrivals?|latest\s+(drop|collection|release)s?|new\s+collection|this\s+(winter|fall|autumn|spring|summer)|"
     r"(winter|fall|autumn)\s+(arrivals?|collection))\b",
     "new_arrivals_intent", "New arrivals & collections", 10, True),

    (r"\b(size\s*(chart|guide)|sizing\s*(chart|guide)?|size\s*info|size\s*measurement(s)?)\b",
     "size_chart_intent", "Size & fit guidance", 9, True),

    (r"\b(free\s+return(s)?(\s+shipping)?|return\s+shipping\s+covered)\b",
     "free_returns_intent", "Shipping & returns", 10, True),

    (r"\b(ship|shipping|deliver(y|ed|ing)?|eta|track(ing)?|when\s+will\s+it\s+(arrive|be\s+delivered)|how\s+long.*(deliver|shipping|arrive))\b",
     "shipping_intent", "Shipping & returns", 9, True),

    (r"\b(return\s+policy|refund\s+policy|return\s+window|return|refund|send back|exchange)\b",
     "returns_intent", "Shipping & returns", 8, True),

    (r"\b(availability|in stock|stock|have .* size|colors?|sizes?(?!\s*(chart|guide)))\b",
     "availability_intent", "Check product availability", 7, True),

    (r"\b(reward|point|redeem|earn|membership|tier)\b",
     "rewards_intent", "Rewards & membership", 6, True),

    (r"\b(deals?|discounts?|promotions?|sale|promo code|coupon)\b",
     "promotions_intent", "Discounts & promotions", 8, True),

    (r"\b(price|price\s*range|how much|cost)\b",
     "price_intent", "Check product availability", 7, True),

    (r"\b(mens|men’s|for\s+a\s+man|male)\b",
     "mens_catalog_intent", "Other", 7, True),
]

def detect_global_intent(user_text: str):
    text = (user_text or "").lower()
    best = None
    for pat, key, target, prio, can_inline in GLOBAL_INTENTS:
        if re.search(pat, text, re.I):
            if (best is None) or (prio > best["priority"]):
                best = {"key": key, "target": target, "priority": prio, "can_inline": can_inline}
    return best


# =========================
# Inline answer functions (UPDATED: use RAG instead of hardcoding)
# =========================
def inline_answer_shipping(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="shipping delivery processing standard express international tracking eta")

def inline_answer_return_policy(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="returns policy return window eligibility refund timing return shipping covered")

def inline_answer_availability(user_text: str) -> str:
    _update_slots_from_text(user_text)
    slots = st.session_state.flow["slots"]
    p = slots.get("product") or "item"
    c = slots.get("color")
    s = slots.get("size")
    q = f"User asks about availability for {p}"
    if c:
        q += f" in {c}"
    if s:
        q += f" size {s}"
    q += ". Explain how to check stock and what details you need."
    return answer_with_rag(q, retrieval_hint="availability in stock colors sizes inventory how to check")

def inline_answer_fit(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="size fit guidance true to size runs small runs large")

def inline_answer_rewards(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="rewards membership points tiers redeem expiration")

def inline_answer_size_chart(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="size chart size guide measurements bust waist hip")

def inline_answer_new_arrivals(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="new arrivals collection lineup items colors categories")

def inline_answer_promotions(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="promotions discounts coupon codes exclusions stacking")

def inline_answer_price(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="pricing price range typical ranges by category")

def inline_answer_mens_catalog(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="mens catalog men and women categories shirts pants jackets knitwear")

def inline_answer_free_returns(user_text: str) -> str:
    return answer_with_rag(user_text, retrieval_hint="free returns return shipping covered exceptions")


# =========================
# Inline handler mapping
# =========================
INLINE_HANDLERS = {
    "availability_intent": inline_answer_availability,
    "rewards_intent":      inline_answer_rewards,
    "size_chart_intent":   inline_answer_size_chart,
    "shipping_intent":     inline_answer_shipping,
    "new_arrivals_intent": inline_answer_new_arrivals,
    "promotions_intent":   inline_answer_promotions,
    "price_intent":        inline_answer_price,
    "mens_catalog_intent": inline_answer_mens_catalog,
    "returns_intent":      inline_answer_return_policy,
    "free_returns_intent": inline_answer_free_returns,
}


# =========================
# Orchestrator
# =========================
def handle_message(user_text: str) -> str:
    # 1) Handle pending yes/no first
    pending_reply = handle_pending_yes(user_text)
    if pending_reply:
        return maybe_add_one_time_closing(apply_tone_policies(pending_reply))

    # 2) Global intent detection and inline
    detected = detect_global_intent(user_text)
    if detected:
        current = st.session_state.flow.get("scenario")
        target  = detected["target"]

        if detected["can_inline"]:
            inline_fun = INLINE_HANDLERS.get(detected["key"])
            if inline_fun:
                # same scenario -> answer directly
                if current == target:
                    reply = inline_fun(user_text)
                    return maybe_add_one_time_closing(apply_tone_policies(reply))

        # scenario switch suggestion when different
        if current != target:
            if detected["can_inline"]:
                inline_fun = INLINE_HANDLERS.get(detected["key"])
                if inline_fun:
                    reply = inline_fun(user_text)
                    set_pending("confirm_switch", {"target": target})
                    return maybe_add_one_time_closing(apply_tone_policies(reply))
            msg = f"It sounds like **{target}** might be more helpful. Switch to that topic?"
            set_pending("confirm_switch", {"target": target})
            return maybe_add_one_time_closing(apply_tone_policies(msg))

    # 3) Scenario-specific routing
    current_scenario = st.session_state.flow.get("scenario")
    rule_reply = route_by_scenario(current_scenario, user_text)
    if rule_reply is not None:
        infer_pending_from_bot_reply(rule_reply)
        return maybe_add_one_time_closing(apply_tone_policies(rule_reply))

    # 4) LLM fallback
    bot_reply = llm_fallback(user_text)
    infer_pending_from_bot_reply(bot_reply)
    return maybe_add_one_time_closing(apply_tone_policies(bot_reply))


# =========================
# UI — 전체(인사/채팅 → 시나리오 → 입력/진행/종료/만족도) + 즉시 표시
# =========================
chat_area = st.container()
st.markdown("---")
scenario_area = st.container()
st.markdown("---")
control_area = st.container()

# -------------------------
# (중간) 시나리오 드롭다운 — 선택 처리
# -------------------------
with scenario_area:
    st.markdown("**How can I help you with?**")
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

    scenario = st.selectbox(
        "Select a scenario",
        SCENARIOS,
        index=0,
        key="scenario_select",
        label_visibility="collapsed",
    )

    other_goal_input = ""
    if scenario == "Other":
        other_goal_input = st.text_input(
            "If 'Other', briefly describe your goal (optional)"
        )

    if (
        scenario != "— Select a scenario —"
        and st.session_state.last_user_selected_scenario != scenario
    ):
        st.session_state.scenario_selected_once = True
        st.session_state.last_user_selected_scenario = scenario
        st.session_state.flow = {
            "scenario": scenario,
            "stage": "start",
            "slots": {
                "product": None, "color": None, "size": None,
                "contact_pref": None, "tier_known": None, "selected_collection": None,
                "return_item": None, "received_date": None, "return_reason": None
            }
        }
        st.session_state.chat_history.append(
            (_chatbot_speaker(), f"Sure, I will help you with **{scenario}**. Please ask me a question.")
        )

# -------------------------
# (하단) 입력/진행/종료 또는 만족도 단계
# -------------------------
with control_area:
    scenario_selected = (st.session_state.flow.get("scenario") is not None)

    if not st.session_state.awaiting_feedback and not st.session_state.ended:
        if scenario_selected:
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("Your message:")
                submitted = st.form_submit_button("Send")

            if submitted and user_input.strip():
                st.session_state.user_turns += 1
                st.session_state.chat_history.append(("User", user_input.strip()))
                bot_reply = handle_message(user_input.strip())
                st.session_state.chat_history.append((_chatbot_speaker(), bot_reply))
                st.session_state.bot_turns += 1
        else:
            st.info("Please choose a topic above to start chatting.")

        if st.session_state.user_turns < MIN_USER_TURNS:
            remaining = MIN_USER_TURNS - st.session_state.user_turns
            st.info(
                f"You’ve sent {st.session_state.user_turns}/{MIN_USER_TURNS} messages (minimum). "
                f"{remaining} more to go."
            )
        st.progress(min(st.session_state.user_turns / MIN_USER_TURNS, 1.0))

        st.markdown("---")
        can_end = (st.session_state.user_turns >= MIN_USER_TURNS)
        help_text = None if can_end else f"Please send at least {MIN_USER_TURNS - st.session_state.user_turns} more message(s) before ending."
        if st.button("End Session", disabled=not can_end, help=help_text):
            st.session_state.awaiting_feedback = True
            st.rerun()

    else:
        if st.session_state.awaiting_feedback and not st.session_state.ended:
            st.subheader("Before you go…")
            st.write("**Overall, how satisfied are you with this chatbot service today?**")
            st.caption("1 = Very dissatisfied, 7 = Very satisfied")
            rating = st.slider("Your overall satisfaction", min_value=1, max_value=7, value=5, step=1)

            prolific_id = st.text_input(
                "Please provide your Prolific ID (write N/A if none) — only to check the submission completion.",
                value=""
            )

            if st.button("Submit Rating"):
                ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                scenostr = st.session_state.flow.get("scenario") or "— Select a scenario —"

                transcript_lines = []
                transcript_lines.append("===== Session Transcript =====")
                transcript_lines.append(f"timestamp       : {ts}")
                transcript_lines.append(f"session_id      : {st.session_state.session_id}")
                transcript_lines.append(f"identity_option : {identity_option}")
                transcript_lines.append(f"name_present    : {'present' if show_name else 'absent'}")
                transcript_lines.append(f"picture_present : {'present' if show_picture else 'absent'}")
                transcript_lines.append(f"scenario        : {scenostr}")
                transcript_lines.append(f"user_turns      : {st.session_state.user_turns}")
                transcript_lines.append(f"bot_turns       : {st.session_state.bot_turns}")
                transcript_lines.append(f"prolific_id     : {prolific_id if prolific_id.strip() else 'N/A'}")
                transcript_lines.append("--------------------------------")
                for spk, msg in st.session_state.chat_history:
                    transcript_lines.append(f"{spk}: {msg}")
                transcript_lines.append("--------------------------------")
                transcript_lines.append(f"Satisfaction (1-7): {rating}")
                transcript_text = "\n".join(transcript_lines)

                try:
                    supabase.table("transcripts").insert({
                        "session_id": st.session_state.session_id,
                        "ts": datetime.datetime.utcnow().isoformat() + "Z",
                        "transcript_text": transcript_text,
                    }).execute()

                    supabase.table("sessions").upsert(
                        {
                            "session_id": st.session_state.session_id,
                            "ts_start": datetime.datetime.utcnow().isoformat() + "Z",
                            "ts_end": datetime.datetime.utcnow().isoformat() + "Z",
                            "identity_option": identity_option,
                            "brand_type": brand_type,
                            "name_present": "present" if show_name else "absent",
                            "picture_present": "present" if show_picture else "absent",
                            "scenario": scenostr,
                            "user_turns": st.session_state.user_turns,
                            "bot_turns": st.session_state.bot_turns,
                        },
                        on_conflict="session_id"
                    ).execute()

                except Exception as e:
                    st.error(f"Failed to save to Supabase: {e}")
                else:
                    st.session_state.rating_saved = True
                    st.session_state.ended = True
                    st.session_state.awaiting_feedback = False
                    st.success("Thanks! Your feedback has been recorded. The session is now closed.")

# -------------------------
# (상단) 채팅 렌더링 — 인사 → 선택 반영 → 방금 입력/응답 순서로 즉시 표시
# -------------------------
with chat_area:
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(
                f"""
                <div style='text-align:right; margin:6px 0;'>
                    <span style='background-color:#DCF8C6; padding:8px 12px; border-radius:12px; display:inline-block;'>
                        <b>You:</b> {message}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='text-align:left; margin:6px 0;'>
                    <span style='background-color:#F1F0F0; padding:8px 12px; border-radius:12px; display:inline-block;'>
                        <b>{speaker}:</b> {message}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

