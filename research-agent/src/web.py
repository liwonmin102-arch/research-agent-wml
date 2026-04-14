"""FastAPI web UI for the research agent with SSE progress streaming."""

import asyncio
import html
import threading
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request

load_dotenv()
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from src.agent import ResearchAgent
from src.models import ResearchReport
from src.report import render_html, render_json, render_markdown

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# task_id -> {status, progress, report, error, topic}
tasks: dict[str, dict] = {}


def _classify_line(msg: str) -> str:
    low = msg.lower()
    if low.startswith("step "):
        return "step"
    if "complete!" in low or low.startswith("research complete"):
        return "done"
    if "error" in low or "failed" in low:
        return "error"
    if low.startswith("found ") and "contradiction" in low:
        return "warn"
    return ""


def _run_research(task_id: str, topic: str) -> None:
    print(f"[thread] started task_id={task_id} topic={topic!r}", flush=True)

    def on_progress(msg: str) -> None:
        tasks[task_id]["progress"].append(msg)
        print(f"[progress {task_id[:8]}] {msg}", flush=True)

    try:
        agent = ResearchAgent(on_progress=on_progress)
        report = agent.research(topic)
        tasks[task_id]["report"] = report
        tasks[task_id]["status"] = "complete"
        print(f"[thread] task_id={task_id} complete", flush=True)
    except Exception as e:
        import traceback

        tasks[task_id]["error"] = str(e)
        tasks[task_id]["status"] = "error"
        print(f"[thread] task_id={task_id} ERROR: {e}", flush=True)
        traceback.print_exc()


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/research")
async def create_research(topic: str = Form(...)):
    topic = topic.strip()
    if len(topic) < 3 or len(topic) > 500:
        raise HTTPException(status_code=400, detail="Topic must be 3–500 characters.")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "progress": [],
        "report": None,
        "error": None,
        "topic": topic,
    }
    print(f"[POST /research] created task_id={task_id} topic={topic!r}", flush=True)
    t = threading.Thread(target=_run_research, args=(task_id, topic), daemon=True)
    t.start()
    print(f"[POST /research] thread alive={t.is_alive()}", flush=True)
    return RedirectResponse(url=f"/research/{task_id}", status_code=303)


@app.get("/research/{task_id}")
async def research_page(request: Request, task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return templates.TemplateResponse(
        request,
        "research.html",
        {"task_id": task_id, "topic": task["topic"]},
    )


@app.get("/api/progress/{task_id}")
async def progress_stream(request: Request, task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    print(f"[SSE] connected task_id={task_id} status={task['status']}", flush=True)

    async def event_generator():
        last_index = 0
        while True:
            if await request.is_disconnected():
                print(f"[SSE {task_id[:8]}] client disconnected", flush=True)
                break

            progress = task["progress"]
            while last_index < len(progress):
                msg = progress[last_index]
                last_index += 1
                cls = _classify_line(msg)
                cls_attr = f" {cls}" if cls else ""
                print(f"[SSE {task_id[:8]}] yield progress #{last_index}: {msg[:60]}", flush=True)
                yield {
                    "event": "progress",
                    "data": f'<span class="line{cls_attr}">{html.escape(msg)}</span>',
                }

            if task["status"] == "complete":
                report: ResearchReport = task["report"]
                download_bar = (
                    '<div class="download-bar">'
                    f'<a href="/api/report/{task_id}/download?format=markdown" role="button">Download Markdown</a>'
                    f'<a href="/api/report/{task_id}/download?format=json" role="button" class="secondary">Download JSON</a>'
                    "</div>"
                )
                print(f"[SSE {task_id[:8]}] yield complete", flush=True)
                yield {
                    "event": "complete",
                    "data": render_html(report) + download_bar,
                }
                break
            if task["status"] == "error":
                print(f"[SSE {task_id[:8]}] yield error: {task['error']}", flush=True)
                yield {
                    "event": "error",
                    "data": html.escape(task["error"] or "Unknown error"),
                }
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/report/{task_id}/download")
async def download_report(task_id: str, format: str = "markdown"):
    task = tasks.get(task_id)
    if task is None or task["report"] is None:
        raise HTTPException(status_code=404, detail="Report not ready")

    report: ResearchReport = task["report"]
    slug = "".join(c if c.isalnum() else "-" for c in task["topic"].lower()).strip("-")[:50]

    if format == "json":
        body = render_json(report)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{slug}.json"'},
        )

    body = render_markdown(report)
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )


def run() -> None:
    """Console-script entry point: `research-agent-web`."""
    import uvicorn

    uvicorn.run("src.web:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
