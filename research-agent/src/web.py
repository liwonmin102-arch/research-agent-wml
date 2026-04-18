"""FastAPI web UI for the multi-agent research pipeline with SSE progress streaming."""

import asyncio
import html
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from src.main import render_markdown
from src.models import DraftReport, ResearchSession
from src.orchestrator import run_research

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@dataclass
class Job:
    id: str
    topic: str
    status: str = "pending"  # pending | running | complete | error
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    session: ResearchSession | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)


JOBS: dict[str, Job] = {}
# Idempotency: client IP -> currently running job_id (cleared when job terminates).
ACTIVE_JOBS_BY_IP: dict[str, str] = {}


# --- Progress line formatting -------------------------------------------


def _stage_line(stage: str, data: dict) -> str:
    """Human-readable progress line for a given stage callback."""
    if stage == "planner_start":
        return f"Planner: starting — topic: {data.get('topic', '')!r}"
    if stage == "planner_done":
        return (
            f"Planner: done — effort={data['effort_level']}, "
            f"target={data['target_source_count']} sources, "
            f"{len(data['queries'])} queries planned"
        )
    if stage == "search_start":
        return f"Searcher: running {data['query_count']} queries…"
    if stage == "search_done":
        return f"Searcher: collected {data['source_count']} unique sources"
    if stage == "analyst_start":
        return f"Analyst: analyzing {data['source_count']} sources in parallel…"
    if stage == "analyst_done":
        return f"Analyst: {data['analyzed_count']} sources analyzed"
    if stage == "critic_start":
        return f"Critic: starting (round {data['round']})"
    if stage == "critic_done":
        return (
            f"Critic round {data['round']} — confidence {data['overall_confidence']}/100, "
            f"{data['contradictions']} contradictions, {data['gaps']} gaps"
        )
    if stage == "reflection_start":
        q = data.get("follow_up_queries", [])
        return f"Reflection: running {len(q)} follow-up queries to resolve gaps"
    if stage == "synthesizer_start":
        return "Synthesizer: drafting the final report…"
    if stage == "synthesizer_done":
        return f"Synthesizer: done — '{data['title']}'"
    return f"{stage}: {data}"


def _stage_class(stage: str) -> str:
    if stage.endswith("_start"):
        return "step"
    if stage.endswith("_done"):
        return "done"
    if stage == "reflection_start":
        return "warn"
    return ""


# --- Report HTML rendering (sent in the 'complete' event) ---------------


def _render_report_html(session: ResearchSession) -> str:
    e = html.escape
    r: DraftReport | None = session.draft_report
    assert r is not None

    parts: list[str] = ['<section class="report-section">']
    parts.append(f"<h1>{e(r.title)}</h1>")
    meta_bits: list[str] = [
        f"Sources: {len(session.source_analyses)}",
        f"Findings: {len(r.key_findings)}",
        f"References: {len(r.references)}",
    ]
    if session.critic_report is not None:
        meta_bits.append(f"Confidence: {session.critic_report.overall_confidence}/100")
    parts.append(f'<p class="report-meta"><em>{e(" · ".join(meta_bits))}</em></p>')

    parts.append("<h2>Executive Summary</h2>")
    parts.append(f"<p>{e(r.executive_summary)}</p>")

    parts.append("<h2>Introduction</h2>")
    for para in r.introduction.split("\n\n"):
        if para.strip():
            parts.append(f"<p>{e(para)}</p>")

    parts.append("<h2>Key Findings</h2>")
    for f in r.key_findings:
        parts.append('<article class="source-card">')
        parts.append(f"<h3>{e(f.theme)}</h3>")
        for para in f.content.split("\n\n"):
            if para.strip():
                parts.append(f"<p>{e(para)}</p>")
        if f.supporting_sources:
            links = " · ".join(
                f'<a href="{e(url)}" target="_blank" rel="noopener">{e(url)}</a>'
                for url in f.supporting_sources
            )
            parts.append(f'<p class="url"><small>Sources: {links}</small></p>')
        parts.append("</article>")

    parts.append("<h2>Contradictions, Debates, and Uncertainties</h2>")
    parts.append('<div class="contradiction-callout">')
    for para in r.contradictions_section.split("\n\n"):
        if para.strip():
            parts.append(f"<p>{e(para)}</p>")
    parts.append("</div>")

    for heading, body in (
        ("Implications", r.implications),
        ("Limitations", r.limitations),
        ("Conclusion", r.conclusion),
    ):
        parts.append(f"<h2>{heading}</h2>")
        for para in body.split("\n\n"):
            if para.strip():
                parts.append(f"<p>{e(para)}</p>")

    parts.append("<h2>References</h2><ol>")
    for ref in r.references:
        author = f" — {e(ref.author)}" if ref.author else ""
        date = f" ({e(ref.publication_date)})" if ref.publication_date else ""
        parts.append(
            f'<li><a href="{e(ref.url)}" target="_blank" rel="noopener">'
            f"{e(ref.title)}</a>{author}{date}</li>"
        )
    parts.append("</ol>")

    parts.append("<h2>Study Guide</h2>")
    parts.append("<h3>Key Takeaways</h3><ul>")
    for t in r.study_guide.key_takeaways:
        parts.append(f"<li>{e(t)}</li>")
    parts.append("</ul>")
    parts.append("<h3>Critical Thinking Questions</h3><ol>")
    for q in r.study_guide.critical_thinking_questions:
        parts.append(f"<li>{e(q)}</li>")
    parts.append("</ol>")
    parts.append("<h3>Further Reading</h3><ul>")
    for ref in r.study_guide.further_reading:
        parts.append(
            f'<li><a href="{e(ref.url)}" target="_blank" rel="noopener">'
            f"{e(ref.title)}</a></li>"
        )
    parts.append("</ul>")

    parts.append("</section>")
    return "".join(parts)


# --- Background worker --------------------------------------------------


def _run_pipeline_thread(
    job_id: str, enable_reflection: bool, loop: asyncio.AbstractEventLoop
) -> None:
    """Runs in a worker thread. Pushes events onto the job's asyncio.Queue via run_coroutine_threadsafe."""
    job = JOBS[job_id]
    job.status = "running"
    print(f"[thread] job={job_id} topic={job.topic!r} reflection={enable_reflection}", flush=True)

    def _put(name: str, data: dict) -> None:
        asyncio.run_coroutine_threadsafe(job.events.put((name, data)), loop)

    def on_progress(stage: str, data: dict) -> None:
        _put(
            "progress",
            {"line": _stage_line(stage, data), "class": _stage_class(stage)},
        )
        _put(stage, data)
        print(f"[{job_id[:8]}] {stage}", flush=True)

    try:
        session = run_research(
            topic=job.topic,
            enable_reflection=enable_reflection,
            progress_callback=on_progress,
        )
        job.session = session
        job.status = "complete"
        _put("complete", {"html": _render_report_html(session)})
        print(f"[thread] job={job_id} complete", flush=True)
    except Exception as e:
        import traceback

        job.error = str(e)
        job.status = "error"
        _put("error", {"message": str(e)})
        print(f"[thread] job={job_id} ERROR: {e}", flush=True)
        traceback.print_exc()
    finally:
        # Release the IP's active-job slot so the user can start a new topic.
        for ip, active_id in list(ACTIVE_JOBS_BY_IP.items()):
            if active_id == job_id:
                del ACTIVE_JOBS_BY_IP[ip]


# --- Routes -------------------------------------------------------------


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/research")
async def create_research(
    request: Request,
    topic: str = Form(...),
    reflection: str = Form("on"),
):
    topic = topic.strip()
    if len(topic) < 3 or len(topic) > 500:
        raise HTTPException(status_code=400, detail="Topic must be 3–500 characters.")

    client_ip = request.client.host if request.client else "unknown"

    # Idempotency guard: if this IP already has an active job, redirect to it instead of starting a new one.
    existing_id = ACTIVE_JOBS_BY_IP.get(client_ip)
    if existing_id:
        existing = JOBS.get(existing_id)
        if existing and existing.status in ("pending", "running"):
            print(
                f"[POST /research] IP={client_ip} already has active job {existing_id}; redirecting.",
                flush=True,
            )
            return RedirectResponse(url=f"/research/{existing_id}", status_code=303)

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, topic=topic)
    JOBS[job_id] = job
    ACTIVE_JOBS_BY_IP[client_ip] = job_id

    loop = asyncio.get_running_loop()
    enable_reflection = reflection != "off"
    threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, enable_reflection, loop),
        daemon=True,
    ).start()

    print(f"[POST /research] job={job_id} ip={client_ip} topic={topic!r}", flush=True)
    return RedirectResponse(url=f"/research/{job_id}", status_code=303)


@app.get("/research/{job_id}")
async def research_page(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return templates.TemplateResponse(
        request,
        "research.html",
        {"job_id": job_id, "topic": job.topic},
    )


@app.get("/research/stream/{job_id}")
async def stream_progress(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    async def event_generator():
        # Fast path: if the job is already terminal (e.g. user reconnected after completion),
        # send the final event and close immediately. Do NOT loop.
        if job.status == "complete":
            assert job.session is not None
            yield {
                "event": "complete",
                "data": json.dumps({"html": _render_report_html(job.session)}),
            }
            yield {"event": "close", "data": "{}"}
            return
        if job.status == "error":
            yield {
                "event": "error",
                "data": json.dumps({"message": job.error or "Unknown error"}),
            }
            yield {"event": "close", "data": "{}"}
            return

        # Live path: drain the queue until we see a terminal event, then close.
        while True:
            if await request.is_disconnected():
                return
            try:
                event_name, data = await asyncio.wait_for(job.events.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Nothing to send; if the job finished during the wait, deliver and close.
                if job.status in ("complete", "error"):
                    if job.status == "complete" and job.session is not None:
                        yield {
                            "event": "complete",
                            "data": json.dumps(
                                {"html": _render_report_html(job.session)}
                            ),
                        }
                    else:
                        yield {
                            "event": "error",
                            "data": json.dumps({"message": job.error or "Unknown error"}),
                        }
                    yield {"event": "close", "data": "{}"}
                    return
                continue

            yield {"event": event_name, "data": json.dumps(data)}

            if event_name in ("complete", "error"):
                yield {"event": "close", "data": "{}"}
                return

    return EventSourceResponse(event_generator())


@app.get("/research/result/{job_id}")
async def get_result(job_id: str, format: str = "html"):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job.status != "complete" or job.session is None:
        raise HTTPException(
            status_code=409, detail=f"Job not complete (status: {job.status})"
        )

    if format == "json":
        return JSONResponse(job.session.model_dump(mode="json"))
    return HTMLResponse(_render_report_html(job.session))


@app.get("/research/download/{job_id}")
async def download_report(job_id: str, format: str = "markdown"):
    job = JOBS.get(job_id)
    if job is None or job.session is None:
        raise HTTPException(status_code=404, detail="Report not ready")

    session: ResearchSession = job.session
    slug = "".join(c if c.isalnum() else "-" for c in job.topic.lower()).strip("-")[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if format == "json":
        body = json.dumps(session.model_dump(mode="json"), indent=2, default=str)
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{slug}-{timestamp}.json"'
            },
        )

    body = render_markdown(session)
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{slug}-{timestamp}.md"'
        },
    )


def run() -> None:
    """Console-script entry point: `research-agent-web`."""
    import uvicorn

    uvicorn.run("src.web:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
