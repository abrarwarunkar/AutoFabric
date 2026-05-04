"""
AutoFabric — Streamlit Dashboard
Enterprise AI Agent Orchestration Platform frontend.
Single-file, production-grade UI with dark mode, animations, and rich UX.
"""
import time
import os
import streamlit as st
import requests
import json

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoFabric | AI Agent Orchestration",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend URL ────────────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #091018 100%);
    color: #e2e8f0;
  }

  /* Glassmorphism cards */
  .glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    transition: border-color 0.3s ease, transform 0.2s ease;
  }
  .glass-card:hover {
    border-color: rgba(99, 179, 237, 0.3);
    transform: translateY(-1px);
  }

  /* Brand header */
  .brand-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    padding: 0;
  }

  .brand-tagline {
    color: #64748b;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 500;
  }

  /* Metric cards */
  .metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 179, 237, 0.1);
    border: 1px solid rgba(99, 179, 237, 0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #63b3ed;
    margin: 3px;
  }

  /* Status badges */
  .badge-verified {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #10b981;
  }

  .badge-unverified {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #ef4444;
  }

  /* Warning banners */
  .banner-pii {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.08));
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #fbbf24;
    font-size: 0.88rem;
    font-weight: 500;
  }

  .banner-injection {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.08));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #fca5a5;
    font-size: 0.88rem;
    font-weight: 500;
  }

  /* Answer box */
  .answer-box {
    background: rgba(99, 179, 237, 0.04);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 12px;
    padding: 22px 26px;
    font-size: 0.96rem;
    line-height: 1.75;
    color: #e2e8f0;
    margin: 12px 0;
    white-space: pre-wrap;
  }

  /* Source chips */
  .source-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #a78bfa;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Trace step */
  .trace-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.83rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
  }
  .trace-step:last-child { border-bottom: none; }

  /* Chunk card */
  .chunk-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
    font-size: 0.84rem;
    color: #cbd5e1;
    line-height: 1.6;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.45) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
  }

  /* Section headers */
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin: 20px 0 8px 0;
  }

  /* Divider */
  .custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 179, 237, 0.3), transparent);
    margin: 20px 0;
  }

  /* Override Streamlit defaults */
  .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stTextArea textarea:focus {
    border-color: rgba(102, 126, 234, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def call_query(query: str, top_k: int) -> dict:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": query, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to AutoFabric backend. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def call_ingest(file_bytes: bytes, filename: str) -> dict:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ingest",
            files={"file": (filename, file_bytes)},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to AutoFabric backend."}
    except Exception as e:
        return {"error": str(e)}


def call_metrics() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/metrics", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def call_health() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-header">🧬 AutoFabric</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-tagline">Enterprise AI Agent Orchestration</p>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Health status
    health = call_health()
    if health:
        status_color = "🟢" if health.get("status") == "ok" else "🔴"
        redis_icon = "🟢" if health.get("redis") else "🔴"
        faiss_icon = "🟢" if health.get("faiss_loaded") else "🔴"
        st.markdown(f"""
        <div class="glass-card" style="padding:14px;">
          <div style="font-size:0.78rem;color:#64748b;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">System Status</div>
          <div style="display:flex;flex-direction:column;gap:6px;font-size:0.82rem;color:#94a3b8;">
            <div>{status_color} API: {'Online' if health.get('status') == 'ok' else 'Offline'}</div>
            <div>{redis_icon} Redis Cache</div>
            <div>{faiss_icon} FAISS Index ({health.get('faiss_chunks', 0)} chunks)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="padding:14px;border-color:rgba(239,68,68,0.3);">
          <div style="color:#ef4444;font-size:0.85rem;">🔴 Backend Offline</div>
          <div style="color:#64748b;font-size:0.78rem;margin-top:4px;">Start the FastAPI server first</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">⚙️ Configuration</div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=10, value=5, step=1)

    st.markdown('<div class="section-label">📄 Ingest Documents</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload .txt or .pdf",
        type=["txt", "pdf"],
        help="Documents are chunked, embedded, and added to the FAISS knowledge base",
    )
    if uploaded_file:
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner("Ingesting document..."):
                result = call_ingest(uploaded_file.read(), uploaded_file.name)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"✅ Ingested! +{result['chunks_added']} chunks ({result['total_chunks']} total)")

    # Platform Metrics
    st.markdown('<div class="section-label">📊 Platform Metrics</div>', unsafe_allow_html=True)
    metrics = call_metrics()
    if metrics:
        total_q = metrics.get("total_queries", 0)
        cache_rate = f"{metrics.get('cache_hit_rate', 0)*100:.1f}%"
        pii_rate = f"{metrics.get('pii_detection_rate', 0)*100:.1f}%"
        avg_lat = f"{metrics.get('avg_latency_ms', 0):.0f}ms"

        col1, col2 = st.columns(2)
        col1.metric("Queries", total_q)
        col2.metric("Avg Latency", avg_lat)
        col1.metric("Cache Hit", cache_rate)
        col2.metric("PII Rate", pii_rate)
    else:
        st.caption("No metrics yet — run a query first")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;text-align:center;">
      Powered by LangGraph · Groq · FAISS · MCP
    </div>
    """, unsafe_allow_html=True)


# ── Main Area ─────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div style="margin-bottom: 24px;">
  <h1 style="font-family:'Inter',sans-serif;font-size:2.4rem;font-weight:700;
             background:linear-gradient(135deg,#667eea,#764ba2,#06b6d4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin:0;padding:0;">
    AutoFabric
  </h1>
  <p style="color:#64748b;font-size:0.95rem;margin:6px 0 0 0;">
    Enterprise AI Agent Orchestration Platform · Hybrid RAG · LangGraph · MCP
  </p>
</div>
""", unsafe_allow_html=True)

# Query Input
st.markdown('<div class="section-label">💬 Ask Your Question</div>', unsafe_allow_html=True)

query_input = st.text_area(
    "Enter your query",
    value=st.session_state.get("query_text", ""),
    height=100,
    placeholder="Ask anything... AutoFabric will retrieve, summarize, and validate an answer from your knowledge base.",
    label_visibility="collapsed",
    key="query_textarea",
)

col1, col2, col3 = st.columns([2, 1, 4])
with col1:
    submit = st.button("🚀 Run Query", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.rerun()

# ── Query Execution ───────────────────────────────────────────────────────────
if submit and query_input.strip():
    with st.spinner("🧬 AutoFabric agents processing your query..."):
        t0 = time.time()
        result = call_query(query_input.strip(), top_k)
        wall_time = (time.time() - t0) * 1000
    st.session_state["last_result"] = result

# ── Results Display ───────────────────────────────────────────────────────────
result = st.session_state.get("last_result")

if result:
    if "error" in result:
        st.markdown(f"""
        <div class="banner-injection">
          ❌ <strong>Error:</strong> {result['error']}
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Warning Banners ────────────────────────────────────────────────────
        if result.get("injection_detected"):
            st.markdown("""
            <div class="banner-injection">
              🛑 <strong>Prompt Injection Detected</strong> — This query was blocked by the AutoFabric security layer.
              No LLM calls were made.
            </div>
            """, unsafe_allow_html=True)

        if result.get("pii_detected"):
            st.markdown("""
            <div class="banner-pii">
              🔒 <strong>PII Detected &amp; Redacted</strong> — Personal information was found in your query
              and redacted before processing. Your data was not sent to any LLM.
            </div>
            """, unsafe_allow_html=True)

        # ── Validation Badge + Mini Metrics ───────────────────────────────────
        validation = result.get("validation", {})
        is_grounded = validation.get("is_grounded", False)
        confidence = validation.get("confidence", 0.0)
        latency = result.get("latency_ms", 0)
        cache_hit = result.get("cache_hit", False)

        badge_col, m1, m2, m3 = st.columns([2, 1, 1, 1])
        with badge_col:
            if is_grounded:
                st.markdown(f"""
                <div class="badge-verified">
                  ✅ VERIFIED &nbsp;·&nbsp; Confidence: {confidence:.0%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="badge-unverified">
                  ⚠️ UNVERIFIED &nbsp;·&nbsp; Confidence: {confidence:.0%}
                </div>
                """, unsafe_allow_html=True)
        with m1:
            st.metric("Latency", f"{latency:.0f}ms")
        with m2:
            st.metric("Cache", "HIT ⚡" if cache_hit else "MISS")
        with m3:
            st.metric("Sources", len(result.get("sources", [])))

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Answer ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">🤖 Answer</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="answer-box">{result.get("answer", "No answer generated.")}</div>
        """, unsafe_allow_html=True)

        # ── Sources ────────────────────────────────────────────────────────────
        sources = result.get("sources", [])
        if sources:
            st.markdown('<div class="section-label">📚 Sources</div>', unsafe_allow_html=True)
            chips = "".join(
                f'<span class="source-chip">📄 {s["source"]} · {s["score"]:.3f}</span>'
                for s in sources
            )
            st.markdown(f'<div style="margin:8px 0;">{chips}</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Expandable: Retrieved Chunks ───────────────────────────────────────
        with st.expander(f"🔍 Retrieved Chunks ({len(sources)} documents)", expanded=False):
            if sources:
                for i, src in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="chunk-card">
                      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                        <span style="color:#a78bfa;font-weight:600;font-family:'JetBrains Mono',monospace;">
                          [{i}] {src['source']}
                        </span>
                        <span style="color:#64748b;font-size:0.78rem;">
                          relevance: {src['score']:.4f}
                        </span>
                      </div>
                      <div>{src.get('content_preview','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No chunks retrieved.")

        # ── Expandable: Agent Trace ────────────────────────────────────────────
        agent_trace = result.get("agent_trace", [])
        with st.expander(f"🧠 Agent Trace ({len(agent_trace)} steps)", expanded=False):
            if agent_trace:
                icons = {"Supervisor": "🎯", "SearchAgent": "🔍", "SummarizerAgent": "✍️", "ValidatorAgent": "✅"}
                for step in agent_trace:
                    icon = next((v for k, v in icons.items() if k in step), "▸")
                    st.markdown(f"""
                    <div class="trace-step">
                      <span style="color:#667eea;">{icon}</span>
                      <span>{step}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No trace recorded.")

        # ── Expandable: Validation Detail ──────────────────────────────────────
        flags = validation.get("flags", [])
        with st.expander("🛡️ Validation Detail", expanded=False):
            col_a, col_b = st.columns(2)
            col_a.metric("Grounded", "Yes ✅" if is_grounded else "No ❌")
            col_b.metric("Confidence", f"{confidence:.1%}")
            if flags:
                st.markdown("**Flagged Claims:**")
                for flag in flags:
                    st.markdown(f"- `{flag}`")

elif not submit:
    # Landing state
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:48px 24px;">
      <div style="font-size:3.5rem;margin-bottom:16px;">🧬</div>
      <div style="font-size:1.15rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">
        Ready to Orchestrate
      </div>
      <div style="font-size:0.88rem;color:#475569;max-width:460px;margin:0 auto;line-height:1.7;">
        Upload documents via the sidebar, then ask any question.<br/>
        AutoFabric will route your query through hybrid RAG retrieval,
        LLaMA-powered summarization, and hallucination validation.
      </div>
      <div style="margin-top:24px;display:flex;justify-content:center;flex-wrap:wrap;gap:8px;">
        <span class="metric-pill">🔍 Hybrid Search</span>
        <span class="metric-pill">🤖 LangGraph Agents</span>
        <span class="metric-pill">🛡️ PII Redaction</span>
        <span class="metric-pill">✅ Hallucination Check</span>
        <span class="metric-pill">⚡ Redis Cache</span>
        <span class="metric-pill">📡 LangSmith Tracing</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
