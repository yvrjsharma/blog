---
title: "Read, redact, share: three apps on OpenAI Privacy Filter"
thumbnail: /blog/assets/openai-privacy-filter-web-apps/thumbnail.png
authors:
- user: ysharma
- user: freddyaboulton
---

# Read, redact, share: three apps on OpenAI Privacy Filter

OpenAI released Privacy Filter on the Hub this week: an open-source PII detector that labels text across eight categories in a single forward pass over a 128k context. [Model card](https://huggingface.co/openai/privacy-filter). I spent a few hours building with it and landed on three apps that each reveal a different slice of what it can do.

- [**Document Privacy Explorer**](https://huggingface.co/spaces/ysharma/OPF-Document-PII-Explorer): drop in a PDF or DOCX, read the document back with every PII span highlighted in place.
- [**Image Anonymizer**](https://huggingface.co/spaces/ysharma/OPF-Image-Anonymizer): drop in a image, get it back with black bars over names, emails, and account numbers, editable on a canvas.
- [**SmartRedact Paste**](https://huggingface.co/spaces/ysharma/OPF-SmartRedact-Paste): paste sensitive text, share a public URL that serves the redacted version, keep a private reveal link for yourself.

All three are built on **`gradio.Server`** ([intro post](https://huggingface.co/blog/introducing-gradio-server)), which lets you pair custom HTML/JS frontends with Gradio's queueing, ZeroGPU allocation, and `gradio_client` SDK. In all these apps, **`gradio.Server`** plays the same backend role, and that consistency is exactly what makes it really powerful.

## The model

Privacy Filter is a 1.5B-parameter model with 50M active parameters. PII categories are `private_person`, `private_address`, `private_email`, `private_phone`, `private_url`, `private_date`, `account_number`, `secret`. Context is 128,000 tokens. License is Apache 2.0. Achieves state-of-the-art performance on the PII-Masking-300k benchmark. Full numbers and methodology are in the [official release blog](https://openai.com/index/introducing-openai-privacy-filter/).

## 1. Document Privacy Explorer

<video alt="Using gradio.Server and OpenAI Privacy Filter model to build an app that can redact PII from any given document" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/
openai-privacy-filter-web-apps/doc-pii-explorer.mp4" type="video/mp4">
</video>

**User problem.** You want to read a PII-heavy document (a contract, a resume, an exported chat log) with every detected span highlighted by category, a filter in the sidebar, and a summary dashboard up top. The reading experience should feel like a normal document, not a form.

**What Privacy Filter does here.** The whole file goes through in a single 128k-context forward pass, so there's no chunking, no stitching, and span offsets line up directly with the rendered text. BIOES decoding keeps span boundaries clean through long ambiguous runs.

**What `gr.Server` does here.** You could wire this up in Blocks with `gr.HighlightedText` and a sidebar, and it would work. The reading experience I wanted (serif body, category filters that toggle CSS classes client-side instead of re-running the model, a summary dashboard that doesn't force a page re-render) was easier to hand-author than to compose. `gr.Server` lets me serve the reader view as a single HTML file and hit one POST for the spans:

```python
import gradio as gr
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

server = gr.Server()

@server.get("/", response_class=HTMLResponse)
async def homepage():
    return FRONTEND_HTML                           # reader view; see app.py

@server.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    text = extract_text(file)                      # PyMuPDF / python-docx
    source_text, spans = run_privacy_filter(text)  # single 128k pass
    return JSONResponse({
        "text":  source_text,
        "spans": spans,                            # [{start, end, label}, ...]
        "stats": compute_stats(source_text, spans),
    })

@server.api(name="analyze_text")
def analyze_text_api(text: str) -> str:
    source_text, spans = run_privacy_filter(text)
    return json.dumps({"text": source_text, "spans": spans})
```

`@server.api(name="analyze_text")` is the same logic without the file upload, callable from any `gradio_client`. Try it at [ysharma/OPF-Document-PII-Explorer](https://huggingface.co/spaces/ysharma/OPF-Document-PII-Explorer).

## 2. Image Anonymizer

<video alt="Using gradio.Server and OpenAI Privacy Filter model to build an app that can redact PII from any given image" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/
openai-privacy-filter-web-apps/image-pii-redact.mp4" type="video/mp4">
</video>

**User problem.** You want to share an image or any screenshot (a Slack thread, a receipt, a Stripe dashboard) with black bars over the PII. You want to toggle bars on and off, drag them to reposition, or draw one by hand for anything the model missed, then export the result.

**What Privacy Filter does here.** Tesseract runs OCR and returns per-word bounding boxes. The backend reconstructs the full text with a char-offset to box map, then runs Privacy Filter once over the whole text. Detected character spans are looked up against the word map and joined into pixel rectangles per line.

**What `gr.Server` does here.** `gr.ImageEditor` supports layered annotation and is a reasonable starting point for image redaction. The workflow I wanted (per-bar category metadata, toggle all bars in a category at once, client-side PNG export at natural resolution with no server round-trip) was cleaner to build on a custom `<canvas>` frontend. `gr.Server` hands back pixel rectangles over a plain POST and lets the canvas own everything else:

```python
@server.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    full_text, char_to_box = ocr_image(img)        # per-word boxes + char map
    spans = run_privacy_filter(full_text)
    boxes = spans_to_pixel_boxes(spans, char_to_box)
    return JSONResponse({
        "image_data_url": pil_to_base64(img),
        "width":  img.width,
        "height": img.height,
        "boxes":  boxes,                           # [{x, y, w, h, label, text}, ...]
    })
```

Toggles, drags, new-bar drawing, and PNG export all happen in the browser. Edits never round-trip to the server. Try it at [ysharma/OPF-Image-Anonymizer](https://huggingface.co/spaces/ysharma/OPF-Image-Anonymizer).

## 3. SmartRedact Paste

<video alt="Using gradio.Server and OpenAI Privacy Filter model to build an app that can redact PII from any pasted text and generate a live link to share the redacted text" autoplay loop muted playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/
openai-privacy-filter-web-apps/smartredact-paste.mp4" type="video/mp4">
</video>

**User problem.** You want a pastebin that redacts before sharing. You paste a log line, an email, a support ticket. You get two URLs back. The public one serves the redacted version with `<PRIVATE_PERSON>`, `<PRIVATE_EMAIL>`, `<ACCOUNT_NUMBER>` placeholders, following the redaction convention from the [official blog examples](https://openai.com/index/introducing-openai-privacy-filter/#:~:text=coherent%20masking%20boundaries.-,Example%20input%20text,-Subject%3A%20Q2%20Planning). The private one is gated by a token you keep and shows the original with spans highlighted.

**What Privacy Filter does here.** Swap each detected span with a `<CATEGORY>` placeholder on the stored paste. That's the entire redaction step. Multilingual text (Spanish, French, Chinese, Hindi, and others in the model-card examples) routes through the same call with no change.

**What `gr.Server` does here.** This is the app that genuinely couldn't exist in Blocks. It needs two distinct GET routes for the same paste ID, one public and one token-gated, and the URL shape matters because the reveal URL is the thing you keep. `gr.Blocks` doesn't expose custom routes, so there's no way to build `/view/{pid}?token=...` on top of it. `gr.Server` does, because it's a FastAPI app underneath:

```python
@server.post("/api/paste")
async def create_paste(req: Request):
    body = await req.json()
    source_text, spans = run_privacy_filter(body["text"])
    redacted = redact(source_text, spans)          # <CATEGORY> placeholders
    pid, reveal_token = secrets.token_urlsafe(6), secrets.token_urlsafe(22)
    PASTES[pid] = Paste(pid, reveal_token, source_text, redacted, spans,
                        expires_at=_ttl(body.get("ttl")))  # see app.py
    return JSONResponse({
        "view_path":   f"/view/{pid}",
        "reveal_path": f"/view/{pid}?token={reveal_token}",
    })

@server.get("/view/{pid}", response_class=HTMLResponse)
async def view_paste(pid: str, token: str | None = None):
    p = _store_get(pid)                            # see app.py for store
    if p is None:
        return HTMLResponse(_not_found(), status_code=404)
    revealed = bool(token) and secrets.compare_digest(token, p.reveal_token)
    return HTMLResponse(_render_view(p, revealed))
```

A daemon thread evicts expired pastes every 30 seconds. An `@server.api(name="analyze_paste")` endpoint makes the whole flow callable from `gradio_client`. The whole service, including storage, is about 200 lines of application code because everything lives in one process. Try it at [ysharma/OPF-SmartRedact-Paste](https://huggingface.co/spaces/ysharma/OPF-SmartRedact-Paste).

## What Server provides

| App | What Server provides |
| --- | --- |
| Document Privacy Explorer | `GET /` serving the custom reader view; a single `POST /api/analyze` for spans |
| Screenshot Anonymizer | `POST /api/detect` that returns pixel rectangles to a client-side `<canvas>` |
| SmartRedact Paste | Two GET routes for the same paste ID, one token-gated, plus a background TTL sweeper |

Each app also keeps a Gradio-side `@server.api` endpoint, so the model is callable through `gradio_client` without any extra plumbing.

## Try them

- [Document Privacy Explorer](https://huggingface.co/spaces/ysharma/OPF-Document-PII-Explorer)
- [Image Anonymizer](https://huggingface.co/spaces/ysharma/OPF-Image-Anonymizer)
- [SmartRedact Paste](https://huggingface.co/spaces/ysharma/OPF-SmartRedact-Paste)

Drop in a resume, a screenshot of a Slack thread, a log line with a token in it. The fun part is seeing what Privacy Filter catches (and occasionally misses) on text you actually care about.

## Recommended reading

- OpenAI's release post: [Introducing OpenAI Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/)
- Model card: [openai/privacy-filter on Hugging Face](https://huggingface.co/openai/privacy-filter)
- [Redaction examples and taxonomy on Model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)