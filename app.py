"""Can Search Speak gRPC? - live demo: REST vs gRPC on OpenSearch for bulk ingest, vector search and log RAG.

Run:
    export OS_PASSWORD='<admin password>'
    .venv/bin/streamlit run app.py
"""

import json
import statistics
import time

import altair as alt
import grpc
import pandas as pd
import streamlit as st
from google.protobuf import text_format
from opensearch.protobufs.schemas import common_pb2 as pb

from log_intel import assistant, config, dataset, indexing, load, ml, search
from log_intel.clients import OpenSearchClients

st.set_page_config(page_title="Can Search Speak gRPC?", page_icon="🔎", layout="wide")

COLORS = alt.Scale(domain=["REST", "gRPC"], range=["#E8833A", "#2E86DE"])
N_LOGS = 10_000
EXAMPLES = [
    "Why were customers unable to check out?",
    "What happened to the order database?",
    "Is someone attacking our login?",
    "Which service ran out of memory?",
    "Are emails delayed?",
    "Is any disk running out of space?",
]


# ---------------------------------------------------------------- shared resources

@st.cache_resource
def get_clients():
    return OpenSearchClients()


@st.cache_resource
def get_model_id():
    return ml.ensure_embedding_model(get_clients())


@st.cache_resource
def get_dataset():
    return dataset.load_dataset(N_LOGS)


def state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def ratio_text(rest, grpc_val, lower_is_better=True, unit=""):
    if not rest or not grpc_val:
        return ""
    r = rest / grpc_val if lower_is_better else grpc_val / rest
    if r >= 1.05:
        return f"gRPC {r:.2f}x better{unit}"
    if r <= 0.95:
        return f"REST {1 / r:.2f}x better{unit}"
    return "about the same"


def fmt_bytes(n):
    return f"{n / 1024:,.1f} KB" if n >= 10_240 else f"{n:,} B"


def short_ratio(rest, grpc_val, lower_is_better=True):
    """Compact verdict for metric tiles, e.g. 'gRPC 2.6x', 'REST 1.1x' or '≈ same'."""
    text = ratio_text(rest, grpc_val, lower_is_better)
    return "≈ same" if text == "about the same" else text.replace(" better", "")


def pair_metric(col, label, rest, grpc_val, fmt, lower_is_better=True):
    """One tile: gRPC value big, REST value + verdict underneath."""
    col.metric(f"{label} (gRPC)", fmt(grpc_val),
               f"REST {fmt(rest)} · {short_ratio(rest, grpc_val, lower_is_better)}", delta_color="off")


def _copy_pill(pill_key, target_key):
    """Pill callback: copy the clicked example into another key, then clear the pill so it can be clicked again."""
    if st.session_state.get(pill_key):
        st.session_state[target_key] = st.session_state[pill_key]
    st.session_state[pill_key] = None


# ---------------------------------------------------------------- sidebar: live status

def sidebar():
    st.sidebar.title("🔎 Can Search Speak gRPC?")
    st.sidebar.caption("OpenSearch 3.7 · REST :9200 (HTTPS/JSON) · gRPC :9400 (HTTP/2 + Protobuf)")
    try:
        clients = get_clients()
    except RuntimeError as e:
        st.sidebar.error(str(e))
        st.stop()

    info = clients.call("GET", "/")
    st.sidebar.markdown(f"**REST** {'✅' if 'version' in info else '❌'}  "
                        f"{info.get('version', {}).get('number', '')}")
    st.sidebar.markdown(f"**gRPC** {'✅' if clients.grpc_ready() else '❌'}  {config.GRPC_ADDR}")
    try:
        model_id = get_model_id()
        st.sidebar.markdown(f"**Embeddings** ✅ `{config.EMBED_MODEL}` via ML Commons")
    except Exception as e:  # noqa: BLE001 - show any setup failure in the UI
        st.sidebar.markdown("**Embeddings** ❌")
        st.sidebar.caption(str(e))
        model_id = None
    llm_ok, _ = assistant.ollama_ok()
    st.sidebar.markdown(f"**LLM** {'✅' if llm_ok else '❌'} `{config.LLM_MODEL}` (Ollama)")

    st.sidebar.divider()
    docs = get_dataset()
    if docs is None:
        st.sidebar.warning(f"{N_LOGS:,} logs not prepared yet")
        if model_id and st.sidebar.button("Generate + embed logs (~1 min)", type="primary"):
            bar = st.sidebar.progress(0.0, "Embedding logs…")
            dataset.build_dataset(clients, model_id, N_LOGS,
                                  progress=lambda d, t: bar.progress(d / t, f"Embedding logs… {d:,}/{t:,}"))
            get_dataset.clear()
            st.rerun()
    else:
        st.sidebar.markdown(f"**Dataset** ✅ {len(docs):,} logs × {len(docs[0]['embedding'])}-dim vectors")
    for idx in (config.INDEX_REST, config.INDEX_GRPC):
        st.sidebar.caption(f"`{idx}`: {indexing.doc_count(clients, idx):,} docs")

    if llm_ok and st.sidebar.button("Warm up LLM"):
        assistant.warm_up()
        st.sidebar.success("LLM loaded")
    return clients, model_id, docs


# ---------------------------------------------------------------- tab 1: two languages

def tab_languages(clients, model_id, docs):
    st.subheader("Same question, two languages")
    st.markdown(
        "REST speaks **JSON over HTTP/1.1**. gRPC speaks **Protocol Buffers over HTTP/2** using typed messages "
        "generated from `.proto` files (`opensearch-protobufs`). Below is one real k-NN search, encoded both ways.")
    if docs is None or model_id is None:
        st.info("Prepare the dataset first (sidebar).")
        return

    question = st.text_input("Query", "payment provider timeouts", key="lang_q")
    vector = ml.embed_query(clients, model_id, question)
    rest_body = json.dumps({"size": 5, "_source": {"excludes": ["embedding"]},
                            "query": {"knn": {"embedding": {"vector": vector, "k": 5}}}})
    req = pb.SearchRequest(index=[config.SEARCH_INDEX], search_request_body=pb.SearchRequestBody(
        size=5, x_source=pb.SourceConfig(filter=pb.SourceFilter(excludes=["embedding"])),
        query=pb.QueryContainer(knn=pb.KnnQuery(field="embedding", vector=vector, k=5))))
    wire = req.SerializeToString()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### REST · `POST /{config.SEARCH_INDEX}/_search`")
        st.metric("Request body", fmt_bytes(len(rest_body)))
        st.code(rest_body[:420] + " …", language="json", wrap_lines=True)
        st.caption("Each float is text, e.g. `-0.0123456,` ≈ 10 bytes.")
    with c2:
        st.markdown("#### gRPC · `SearchService/Search`")
        st.metric("Request message", fmt_bytes(len(wire)),
                  ratio_text(len(rest_body), len(wire)).replace("better", "smaller"))
        shown = pb.SearchRequest()
        shown.CopyFrom(req)
        del shown.search_request_body.query.knn.vector[6:]
        st.code(text_format.MessageToString(shown) + "  # … 762 more floats", language="protobuf")
        st.caption("`repeated float vector` is packed: exactly 4 bytes per float.")
        st.code(" ".join(f"{b:02x}" for b in wire[:48]) + " …", language="text")
        st.caption("First bytes on the wire (binary protobuf).")

    cost = state("encode_cost", {})
    if question not in cost:
        cost[question] = load.encode_cost(vector)
    enc = cost[question]
    e = st.columns(3)
    e[0].metric("Build + serialize request: REST (JSON)", f"{enc['REST']:.0f} µs")
    e[1].metric("Build + serialize request: gRPC (protobuf)", f"{enc['gRPC']:.0f} µs")
    e[2].metric("Client CPU to encode", short_ratio(enc["REST"], enc["gRPC"]))
    st.caption("Pure client-side encoding of this request, no network and no TLS (median of 2,000 runs). "
               "Turning 768 floats into decimal text is the expensive part of JSON.")

    st.markdown("#### What's available over gRPC in OpenSearch 3.7 (measured on this cluster)")
    st.dataframe(pd.DataFrame([
        ["Bulk (index / create / update / delete)", "✅", "✅", "Documents are still JSON bytes inside protobuf"],
        ["Search, incl. k-NN, filters, collapse, inner_hits", "✅", "✅", "Typed query DSL"],
        ["Create index / mappings / settings", "✅", "❌", "Admin APIs stay on REST"],
        ["ML predict / agents (streaming)", "✅ (stream needs extra setup)", "❌",
         "MLService.ExecuteAgentStream exists in newer protobufs, not in the 3.7 server"],
        ["Binary vectors in Bulk (extra_field_values)", "-", "⚠️", "Accepted by 3.7 but vectors are not indexed"],
        ["Tooling", "curl, Dashboards, every client", "grpcurl, generated stubs", "Ecosystem trade-off"],
    ], columns=["Capability", "REST", "gRPC", "Notes"]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 0: scoreboard

def _load_vectors(clients, model_id):
    if "load_vectors" not in st.session_state:
        st.session_state["load_vectors"] = [ml.embed_query(clients, model_id, q) for q in EXAMPLES]
    return st.session_state["load_vectors"]


def tab_scoreboard(clients, model_id, docs):
    st.subheader("Why gRPC? Measured live on this cluster")
    st.markdown("One click runs the workloads where gRPC's design pays off: **concurrent vector search**, "
                "**vector request encoding**, and **high-frequency log shipping**. Same cluster, same data, fast "
                "clients on both sides; every number below is measured now, not quoted.")
    if docs is None or model_id is None or indexing.doc_count(clients, config.SEARCH_INDEX) == 0:
        st.info("Prepare the dataset (sidebar) and run the batch load in ② once so the logs are indexed.")
        return

    if st.button("⚡ Measure the scoreboard (~30 s)", type="primary"):
        board = {}
        steps = st.progress(0.0, "Encoding a 768-dim k-NN request both ways…")
        vectors = _load_vectors(clients, model_id)
        board["encode"] = load.encode_cost(vectors[0])
        sizes = search.compare(clients, vectors[0], 10)
        board["req"] = {p: sizes[p][0]["req_bytes"] for p in sizes}

        steps.progress(0.2, "Concurrent k-NN search: 32 in flight…")
        sweep = load.run_sweep(clients, vectors, levels=(32,), requests=1000)
        board["load"] = {r["protocol"]: r for r in sweep}
        state("load_runs", []).append(sweep)

        steps.progress(0.6, "Log shippers: 16 senders × 10 docs per request…")
        plain = [{k: v for k, v in d.items() if k != "embedding"} for d in docs]
        ship = indexing.compare_shippers(clients, plain, batch=10, parallel=16)
        board["ship"] = {p: statistics.mean(ship[p]) for p in ship}
        state("shipper_runs", []).append({"batch": 10, "parallel": 16, **board["ship"],
                                          "REST runs": [round(x) for x in ship["REST"]],
                                          "gRPC runs": [round(x) for x in ship["gRPC"]]})
        steps.empty()
        st.session_state["scoreboard"] = board

    board = st.session_state.get("scoreboard")
    if not board:
        return
    lr, lg = board["load"]["REST"], board["load"]["gRPC"]
    tiles = [
        ("Search throughput, 32 concurrent", lr["qps"], lg["qps"], lambda v: f"{v:,.0f}/s", False, "more"),
        ("Search p99 latency, 32 concurrent", lr["p99"], lg["p99"], lambda v: f"{v:.1f} ms", True, "lower"),
        ("Client CPU per search", lr["cpu_ms_per_req"], lg["cpu_ms_per_req"], lambda v: f"{v:.2f} ms", True,
         "less"),
        ("Encode a 768-dim k-NN request", board["encode"]["REST"], board["encode"]["gRPC"],
         lambda v: f"{v:.0f} µs", True, "faster"),
        ("k-NN request size", board["req"]["REST"], board["req"]["gRPC"], fmt_bytes, True, "smaller"),
        ("Log shippers (16 × 10 docs)", board["ship"]["REST"], board["ship"]["gRPC"],
         lambda v: f"{v:,.0f} docs/s", False, "more"),
    ]
    for row in (tiles[:3], tiles[3:]):
        cols = st.columns(3)
        for col, (label, r, g, fmt, lower, word) in zip(cols, row):
            ratio = (r / g) if lower else (g / r)
            col.metric(label, f"{ratio:.1f}× {word}" if ratio >= 1.05 else "≈ same",
                       f"REST {fmt(r)} → gRPC {fmt(g)}", delta_color="normal" if ratio >= 1.05 else "off",
                       delta_arrow="off", border=True)
    st.caption("Fairness: urllib3 keep-alive pool for REST vs one gRPC channel, warm-up first, alternating order, "
               "client encode/decode included. REST runs over TLS here; the gRPC port is plaintext. "
               "Where gRPC does *not* win (large vector batch loads) is shown in ② and ⑥.")


# ---------------------------------------------------------------- tab 2: bulk ingest race

def tab_ingest(clients, docs):
    st.subheader("Bulk ingest: two real-world shapes")
    if docs is None:
        st.info("Prepare the dataset first (sidebar).")
        return
    _ingest_shippers(clients, docs)
    st.divider()
    _ingest_race(clients, docs)


def _ingest_shippers(clients, docs):
    st.markdown("### A · Log shippers: many small batches in parallel (plain logs, no vectors)")
    st.markdown("Agents like Fluent Bit or Vector send small batches every few seconds from many hosts. "
                "Documents are small, so the **per-request** cost (HTTP/1.1 + TLS connection per in-flight "
                "request vs. one multiplexed HTTP/2 channel) matters more than the per-document cost.")
    c = st.columns([1, 1, 2])
    batch = c[0].select_slider("Docs per request", [10, 25, 50, 100], value=10)
    parallel = c[1].select_slider("Parallel senders", [4, 8, 16, 32], value=16)
    c[2].caption(f"{len(docs):,} logs without the embedding field, each protocol run twice (alternating who goes "
                 "first), into scratch indexes that are deleted afterwards.")
    runs = state("shipper_runs", [])
    if st.button("📦 Run log-shipper scenario", type="primary"):
        plain = [{k: v for k, v in d.items() if k != "embedding"} for d in docs]
        with st.spinner("Shipping logs over REST and gRPC…"):
            result = indexing.compare_shippers(clients, plain, batch, parallel)
        runs.append({"batch": batch, "parallel": parallel,
                     "REST": statistics.mean(result["REST"]), "gRPC": statistics.mean(result["gRPC"]),
                     "REST runs": [round(x) for x in result["REST"]], "gRPC runs": [round(x) for x in result["gRPC"]]})
    if not runs:
        return
    last = runs[-1]
    m = st.columns(3)
    pair_metric(m[0], "Throughput", last["REST"], last["gRPC"], lambda v: f"{v:,.0f} docs/s",
                lower_is_better=False)
    m[1].metric("Requests per second (gRPC)", f"{last['gRPC'] / last['batch']:,.0f}",
                f"REST {last['REST'] / last['batch']:,.0f}", delta_color="off")
    m[2].metric("Individual runs", f"{last['gRPC runs']}", f"REST {last['REST runs']}", delta_color="off")
    if len(runs) > 1:
        st.markdown("##### All log-shipper runs this session")
        st.dataframe(pd.DataFrame([{
            "Docs per request": x["batch"], "Parallel senders": x["parallel"],
            "REST docs/s (avg)": f"{x['REST']:,.0f}", "gRPC docs/s (avg)": f"{x['gRPC']:,.0f}",
            "gRPC vs REST": f"{x['gRPC'] / x['REST']:.2f}×",
            "REST runs": " / ".join(f"{v:,}" for v in x["REST runs"]),
            "gRPC runs": " / ".join(f"{v:,}" for v in x["gRPC runs"]),
        } for x in runs]), hide_index=True, width="stretch")
    st.caption("Measured on this cluster: gRPC ~1.3-1.5x with 10 docs per request and 16 senders; the gain shrinks "
               "as batches grow (~1.2x at 50 docs). With 768-dim vectors in each document it's a tie again: the "
               "server's per-document work dominates.")


def _ingest_race(clients, docs):
    st.markdown(f"### B · Where it's a tie: one big batch load, {N_LOGS:,} logs with 768-dim vectors")
    c1, c2, c3 = st.columns([1, 1, 2])
    n = c1.select_slider("Logs", [1000, 2500, 5000, 10_000], value=10_000)
    batch = c2.select_slider("Batch size", [100, 250, 500, 1000], value=500)
    c3.caption("Each protocol writes to its own fresh index. Batches alternate REST/gRPC first, "
               "timing includes request encoding and response decoding.")
    runs = state("ingest_runs", [])

    if st.button("🏁 Start the race", type="primary"):
        cols = st.columns(2)
        bars = {"REST": cols[0].progress(0.0, "REST"), "gRPC": cols[1].progress(0.0, "gRPC")}
        chart_slot = st.empty()
        events = []
        for ev in indexing.run_ingest(clients, docs[:n], batch):
            events.append(ev)
            p = ev["protocol"]
            elapsed = sum(e["seconds"] for e in events if e["protocol"] == p)
            bars[p].progress(ev["docs_done"] / n, f"{p}: {ev['docs_done']:,}/{n:,} docs · {elapsed:.2f}s")
            if len(events) % 4 == 0 or ev["docs_done"] == n:
                chart_slot.altair_chart(_batch_chart(events), width="stretch")
        runs.append({"logs": n, "batch": batch, "events": events})

    if not runs:
        return
    last = runs[-1]
    ev = pd.DataFrame(last["events"])
    agg = ev.groupby("protocol").agg(seconds=("seconds", "sum"), req=("req_bytes", "sum"),
                                     resp=("resp_bytes", "sum"), took=("server_took_ms", "sum"),
                                     errors=("errors", "sum"))
    r, g = agg.loc["REST"], agg.loc["gRPC"]
    st.markdown(f"#### Result: {last['logs']:,} logs, batch {last['batch']}")
    m = st.columns(5)
    m[0].metric("REST total", f"{r.seconds:.2f}s", f"{last['logs'] / r.seconds:,.0f} docs/s", delta_color="off")
    m[1].metric("gRPC total", f"{g.seconds:.2f}s", f"{last['logs'] / g.seconds:,.0f} docs/s", delta_color="off")
    m[2].metric("Speed", short_ratio(r.seconds, g.seconds))
    pair_metric(m[3], "Request", r.req, g.req, lambda v: f"{v / 1e6:.1f} MB")
    pair_metric(m[4], "Response", r.resp, g.resp, lambda v: f"{v / 1e3:.0f} KB")
    st.altair_chart(_batch_chart(last["events"]), width="stretch")
    st.caption(f"Server-side `took` total: REST {r.took:,.0f} ms · gRPC {g.took:,.0f} ms - the server does the same "
               f"indexing work; differences are transport and encoding. Errors: REST {int(r.errors)}, "
               f"gRPC {int(g.errors)}.")
    st.info("**Why this is a tie:** in Bulk, each document travels as JSON bytes inside the protobuf "
            "message (`bytes object`), so the 768 floats per log are JSON text in *both* protocols and the server "
            "does the same parsing and indexing. More data doesn't change that (50k docs: 0.98x). "
            "gRPC still wins on the response (typed items instead of verbose JSON).")
    if len(runs) > 1:
        st.markdown("##### All runs this session")
        st.dataframe(pd.DataFrame([_ingest_summary(x) for x in runs]), hide_index=True, width="stretch")


def _batch_chart(events):
    df = pd.DataFrame(events)
    df["ms"] = df["seconds"] * 1000
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("batch:Q", title="Batch #"), y=alt.Y("ms:Q", title="Batch latency (ms)"),
        color=alt.Color("protocol:N", scale=COLORS), tooltip=["protocol", "batch", alt.Tooltip("ms:Q", format=".1f")],
    ).properties(height=260)


def _ingest_summary(run):
    ev = pd.DataFrame(run["events"]).groupby("protocol")["seconds"].sum()
    return {"logs": run["logs"], "batch": run["batch"],
            "REST docs/s": round(run["logs"] / ev["REST"]), "gRPC docs/s": round(run["logs"] / ev["gRPC"]),
            "verdict": ratio_text(ev["REST"], ev["gRPC"])}


# ---------------------------------------------------------------- tab 3: vector search

def tab_search(clients, model_id, docs):
    st.subheader("Semantic log search: REST vs gRPC, same index, same vector")
    if docs is None or model_id is None:
        st.info("Prepare the dataset first (sidebar).")
        return
    if indexing.doc_count(clients, config.SEARCH_INDEX) == 0:
        st.info("Run the ingest race first so the logs are indexed.")
        return

    st.pills("Try", EXAMPLES, key="search_example", on_change=_copy_pill, args=("search_example", "search_q"))
    state("search_q", "payment provider timeouts")
    query = st.text_input("Search the logs in plain English", key="search_q")
    c = st.columns([1, 1, 1, 1, 1])
    services = c[0].multiselect("Service", sorted({d["service"] for d in docs}))
    levels = c[1].multiselect("Level", ["ERROR", "WARN", "INFO", "DEBUG"])
    k = c[2].slider("Results (k)", 5, 50, 10)
    group = c[3].toggle("Group by log pattern", True, help="collapse on `pattern` - one hit per message template")
    repeats = c[4].select_slider("Repeat", [1, 20, 100, 300], value=100,
                                 help="Repeat the search to get a latency distribution")
    filters = {"service": services, "level": levels}

    if not st.button("🔍 Search with both protocols", type="primary"):
        return
    t = time.perf_counter()
    vector = ml.embed_query(clients, model_id, query)
    embed_ms = (time.perf_counter() - t) * 1000
    search.compare(clients, vector, k, filters, group, repeats=3)  # warm-up
    with st.spinner(f"Running {repeats}× per protocol…"):
        runs = search.compare(clients, vector, k, filters, group, repeats=repeats)
    rest_runs, grpc_runs = runs["REST"], runs["gRPC"]
    lat = {p: [x["seconds"] * 1000 for x in runs[p]] for p in runs}
    same = [h["_id"] for h in rest_runs[0]["hits"]] == [h["_id"] for h in grpc_runs[0]["hits"]]

    m = st.columns(5)
    m[0].metric("Query embedding", f"{embed_ms:.0f} ms", "ML Commons, same for both", delta_color="off")
    pair_metric(m[1], "p50 latency", statistics.median(lat["REST"]), statistics.median(lat["gRPC"]),
                lambda v: f"{v:.2f} ms")
    pair_metric(m[2], "Request", rest_runs[0]["req_bytes"], grpc_runs[0]["req_bytes"], fmt_bytes)
    pair_metric(m[3], "Response", rest_runs[0]["resp_bytes"], grpc_runs[0]["resp_bytes"], fmt_bytes)
    m[4].metric("Same results?", "✅ identical" if same else "⚠️ differ")

    df = pd.DataFrame([{"protocol": p, "ms": v} for p in lat for v in lat[p]])
    st.caption("One search at a time, the two protocols differ by a fraction of a millisecond, so a single run "
               "can go either way; that's why this shows the distribution. See ④ Under load for where it adds up.")
    st.altair_chart(alt.Chart(df).mark_boxplot(extent="min-max", size=40).encode(
        y=alt.Y("protocol:N", title=None), x=alt.X("ms:Q", title=f"Client latency over {repeats} runs (ms)"),
        color=alt.Color("protocol:N", scale=COLORS, legend=None)).properties(height=160),
        width="stretch")
    state("search_runs", []).append({
        "query": query, "k": k, "group": group, "repeats": repeats,
        "REST p50": statistics.median(lat["REST"]), "gRPC p50": statistics.median(lat["gRPC"]),
        "REST p95": _p95(lat["REST"]), "gRPC p95": _p95(lat["gRPC"]),
        "REST req": rest_runs[0]["req_bytes"], "gRPC req": grpc_runs[0]["req_bytes"],
        "REST resp": rest_runs[0]["resp_bytes"], "gRPC resp": grpc_runs[0]["resp_bytes"]})

    cols = st.columns(2)
    for col, name, run in ((cols[0], "REST", rest_runs[0]), (cols[1], "gRPC", grpc_runs[0])):
        col.markdown(f"##### {name} results")
        col.dataframe(_hits_frame(run["hits"], group), hide_index=True, width="stretch", height=380)


def _p95(values):
    return statistics.quantiles(values, n=20)[-1] if len(values) > 1 else values[0]


def _hits_frame(hits, group):
    rows = []
    for h in hits:
        row = {"score": round(h["score"], 3), "level": h["level"], "service": h["service"], "message": h["message"]}
        if group:
            row = {"score": row["score"], "lines": h["count"], **{k: row[k] for k in ("level", "service", "message")},
                   "first": h["first"][5:19].replace("T", " "), "last": h["last"][5:19].replace("T", " ")}
        else:
            row["time"] = h["@timestamp"][5:19].replace("T", " ")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tab 4: under load

def tab_load(clients, model_id, docs):
    st.subheader("Under load: many searches in flight at once")
    st.markdown(
        "A real app server doesn't send one search at a time. Here N worker threads share **one client per "
        "protocol**: REST uses a keep-alive pool (HTTP/1.1 needs a separate TLS connection for each in-flight "
        "request), gRPC uses **one HTTP/2 channel** that multiplexes all of them.")
    if docs is None or model_id is None or indexing.doc_count(clients, config.SEARCH_INDEX) == 0:
        st.info("Prepare the dataset and run the ingest race first.")
        return
    c = st.columns([2, 1, 2])
    levels = c[0].multiselect("Concurrency levels", [1, 4, 8, 16, 32, 64, 128], default=[1, 8, 32, 64])
    n = c[1].select_slider("Searches per run", [300, 1000, 2000], value=1000)
    c[2].caption("Each level runs both protocols (alternating who goes first) after a warm-up. "
                 "CPU = client process CPU time per search.")
    runs = state("load_runs", [])

    if st.button("🚦 Run load test", type="primary", disabled=not levels):
        vectors = _load_vectors(clients, model_id)
        bar = st.progress(0.0, "Starting…")

        def progress(done, total, last):
            bar.progress(done / total, f"{last['protocol']} @ {last['concurrency']} concurrent: "
                                       f"{last['qps']:,.0f} searches/s, p99 {last['p99']:.1f} ms")
        runs.append(load.run_sweep(clients, vectors, sorted(levels), n, progress=progress))

    if not runs:
        return
    df = pd.DataFrame(runs[-1])
    top = df["concurrency"].max()
    r, g = (df[(df.protocol == p) & (df.concurrency == top)].iloc[0] for p in ("REST", "gRPC"))
    st.markdown(f"#### At {top} concurrent searches")
    m = st.columns(3)
    pair_metric(m[0], "Throughput", r.qps, g.qps, lambda v: f"{v:,.0f}/s", lower_is_better=False)
    pair_metric(m[1], "p99 latency", r.p99, g.p99, lambda v: f"{v:.1f} ms")
    pair_metric(m[2], "Client CPU / search", r.cpu_ms_per_req, g.cpu_ms_per_req, lambda v: f"{v:.2f} ms")

    charts = st.columns(3)
    for col, field, title in ((charts[0], "qps", "Throughput (searches/s) ↑"),
                              (charts[1], "p99", "p99 latency (ms) ↓"),
                              (charts[2], "cpu_ms_per_req", "Client CPU per search (ms) ↓")):
        col.altair_chart(alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("concurrency:O", title="Concurrent searches"), y=alt.Y(f"{field}:Q", title=title),
            color=alt.Color("protocol:N", scale=COLORS),
            tooltip=["protocol", "concurrency", alt.Tooltip(f"{field}:Q", format=",.2f")],
        ).properties(height=260, title=title), width="stretch")
    with st.expander("All numbers"):
        st.dataframe(df.round(2), hide_index=True, width="stretch")
    st.info("**Why gRPC pulls ahead here:** a single request differs by a fraction of a millisecond, but under "
            "load the per-request client cost (encoding 768 floats as text, HTTP/1.1 + TLS per connection) "
            "adds up, while gRPC multiplexes everything over one HTTP/2 connection with a native client stack. "
            "Caveat: REST runs over TLS here, the gRPC port is plaintext.")


# ---------------------------------------------------------------- tab 5: ask the logs

def tab_assistant(clients, model_id, docs):
    st.subheader("Ask the logs: gRPC retrieval + streaming LLM answer")
    st.caption(f"Question → embedding (ML Commons) → grouped k-NN over **gRPC** → top log patterns → "
               f"`{config.LLM_MODEL}` streams the answer.")
    if docs is None or model_id is None or indexing.doc_count(clients, config.SEARCH_INDEX) == 0:
        st.info("Prepare the dataset and run the ingest race first.")
        return
    chat = state("chat", [])
    for turn in chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                st.caption(turn["meta"])

    st.pills("Examples", EXAMPLES, key="chat_example", on_change=_copy_pill, args=("chat_example", "pending_q"))
    question = st.chat_input("Ask about incidents, errors, attacks…") or st.session_state.pop("pending_q", None)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        r = assistant.retrieve(clients, model_id, question)
        groups = r["hits"]
        with st.expander(f"🔎 {len(groups)} log patterns retrieved via gRPC in {r['seconds'] * 1000:.0f} ms "
                         f"({fmt_bytes(r['resp_bytes'])})"):
            st.dataframe(_hits_frame(groups, True), hide_index=True, width="stretch")
        timing = {}
        answer = st.write_stream(assistant.stream_answer(question, groups, history=chat, timing=timing))
        meta = (f"embed {r['embed_seconds'] * 1000:.0f} ms · gRPC k-NN {r['seconds'] * 1000:.0f} ms · "
                f"first token {timing.get('ttft', 0):.2f}s · total {timing.get('total', 0):.1f}s")
        st.caption(meta)
    chat.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer, "meta": meta}])


# ---------------------------------------------------------------- tab 5: trade-offs

def tab_tradeoffs():
    st.subheader("Trade-offs, measured in this session")
    ingest, searches, loads = state("ingest_runs", []), state("search_runs", []), state("load_runs", [])
    if not ingest and not searches and not loads:
        st.info("Run the ingest race, a few searches and the load test; their numbers show up here.")
    rows = []
    shippers = state("shipper_runs", [])
    if shippers:
        s = shippers[-1]
        rows.append([f"Log shippers ({s['parallel']} senders × {s['batch']} docs)", f"{s['REST']:,.0f} docs/s",
                     f"{s['gRPC']:,.0f} docs/s", ratio_text(s["REST"], s["gRPC"], lower_is_better=False)])
    if loads:
        df = pd.DataFrame(loads[-1])
        top = df["concurrency"].max()
        r, g = (df[(df.protocol == p) & (df.concurrency == top)].iloc[0] for p in ("REST", "gRPC"))
        rows += [
            [f"Search throughput @ {top} concurrent", f"{r.qps:,.0f} q/s", f"{g.qps:,.0f} q/s",
             ratio_text(r.qps, g.qps, lower_is_better=False)],
            [f"Search p99 @ {top} concurrent", f"{r.p99:.1f} ms", f"{g.p99:.1f} ms", ratio_text(r.p99, g.p99)],
            ["Client CPU per search", f"{r.cpu_ms_per_req:.2f} ms", f"{g.cpu_ms_per_req:.2f} ms",
             ratio_text(r.cpu_ms_per_req, g.cpu_ms_per_req)],
        ]
    if ingest:
        ev = pd.DataFrame(ingest[-1]["events"]).groupby("protocol").sum(numeric_only=True)
        n = ingest[-1]["logs"]
        rows += [
            ["Bulk ingest throughput", f"{n / ev.loc['REST', 'seconds']:,.0f} docs/s",
             f"{n / ev.loc['gRPC', 'seconds']:,.0f} docs/s",
             ratio_text(ev.loc["REST", "seconds"], ev.loc["gRPC", "seconds"])],
            ["Bulk request bytes", f"{ev.loc['REST', 'req_bytes'] / 1e6:.1f} MB",
             f"{ev.loc['gRPC', 'req_bytes'] / 1e6:.1f} MB",
             ratio_text(ev.loc["REST", "req_bytes"], ev.loc["gRPC", "req_bytes"])],
            ["Bulk response bytes", f"{ev.loc['REST', 'resp_bytes'] / 1e3:.0f} KB",
             f"{ev.loc['gRPC', 'resp_bytes'] / 1e3:.0f} KB",
             ratio_text(ev.loc["REST", "resp_bytes"], ev.loc["gRPC", "resp_bytes"])],
        ]
    if searches:
        s = pd.DataFrame(searches)
        rows += [
            ["k-NN latency p50 (median of runs)", f"{s['REST p50'].median():.2f} ms", f"{s['gRPC p50'].median():.2f} ms",
             ratio_text(s["REST p50"].median(), s["gRPC p50"].median())],
            ["k-NN latency p95", f"{s['REST p95'].median():.2f} ms", f"{s['gRPC p95'].median():.2f} ms",
             ratio_text(s["REST p95"].median(), s["gRPC p95"].median())],
            ["k-NN request bytes", fmt_bytes(int(s["REST req"].median())), fmt_bytes(int(s["gRPC req"].median())),
             ratio_text(s["REST req"].median(), s["gRPC req"].median())],
            ["k-NN response bytes", fmt_bytes(int(s["REST resp"].median())), fmt_bytes(int(s["gRPC resp"].median())),
             ratio_text(s["REST resp"].median(), s["gRPC resp"].median())],
        ]
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "REST", "gRPC", "Verdict"]), hide_index=True,
                     width="stretch")

    c1, c2 = st.columns(2)
    c1.markdown("""
#### Where gRPC helps
- **Under concurrent load:** HTTP/2 multiplexes every request over one connection; higher throughput, much lower tail latency
- **Many small bulk requests** (log shippers, 10 docs/request): ~1.3-1.5x more docs/s
- **Client CPU:** protobuf encoding of vectors is ~10x cheaper than JSON text; the gRPC client stack is native code
- **Vector queries:** packed 4-byte floats instead of JSON text → ~3x smaller requests
- **Typed contracts:** generated clients, compile-time field checks
""")
    c2.markdown("""
#### Where it doesn't (yet)
- **Single, sequential requests:** the gap is a fraction of a millisecond, so run-to-run noise can flip it
- **Bulk with vectors / large batches** is server-bound and docs are JSON bytes inside protobuf → a tie
- **Coverage:** admin APIs, ML agents stay on REST in 3.7
- **Tooling & security:** curl/Dashboards vs grpcurl; this gRPC port is plaintext while REST uses TLS
""")
    st.caption("Fairness: fast keep-alive clients for both (urllib3 pool vs one gRPC channel), warm-up before "
               "measuring, alternating order, same index and same vector, client-side encode/decode included.")


# ---------------------------------------------------------------- main

def main():
    clients, model_id, docs = sidebar()
    tabs = st.tabs(["★ Why gRPC", "① Two languages", "② Bulk ingest", "③ Vector search", "④ Under load",
                    "⑤ Ask the logs", "⑥ Trade-offs"])
    with tabs[0]:
        tab_scoreboard(clients, model_id, docs)
    with tabs[1]:
        tab_languages(clients, model_id, docs)
    with tabs[2]:
        tab_ingest(clients, docs)
    with tabs[3]:
        tab_search(clients, model_id, docs)
    with tabs[4]:
        tab_load(clients, model_id, docs)
    with tabs[5]:
        tab_assistant(clients, model_id, docs)
    with tabs[6]:
        tab_tradeoffs()


try:
    main()
except grpc.RpcError as e:
    st.error(f"gRPC error: {e.code().name}: {e.details()}")
