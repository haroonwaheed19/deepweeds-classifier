"""REST API for the weed classifier.

    uvicorn service.app:app --port 7860

Env: MODEL_PATH, CARD_PATH (defaults: artifacts/model.onnx, artifacts/model_card.json).
"""
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from weedcls.inference.predictor import Predictor

MAX_BYTES = 5 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 32, 6000
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
MAX_BATCH = 16

log = logging.getLogger("weedapi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = os.environ.get("MODEL_PATH", "artifacts/model.onnx")
    card = os.environ.get("CARD_PATH", "artifacts/model_card.json")
    try:
        state["predictor"] = Predictor(model, card)
        log.info("model loaded: %s (%s)", model, state["predictor"].card.get("model_version"))
    except Exception:
        log.exception("could not load model from %s / %s", model, card)
        state["predictor"] = None
    yield


app = FastAPI(title="DeepWeeds classifier", version="1.0", lifespan=lifespan)


def _predictor() -> Predictor:
    p = state.get("predictor")
    if p is None:
        raise HTTPException(503, "model is not loaded")
    return p


async def _read_image(f: UploadFile) -> Image.Image:
    if f.content_type not in ALLOWED:
        raise HTTPException(415, f"unsupported content type '{f.content_type}', use JPEG, PNG or WebP")
    data = await f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, "file is not a readable image") from e
    w, h = img.size
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        raise HTTPException(400, f"image side must be between {MIN_SIDE} and {MAX_SIDE}px, got {w}x{h}")
    return img


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    request.state.rid = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("[%s] unhandled error on %s", rid, request.url.path)
        return JSONResponse({"error": "internal error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    log.info("[%s] %s %s -> %d (%.0f ms)", rid, request.method, request.url.path,
             response.status_code, (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail, "request_id": getattr(request.state, "rid", None)},
                        status_code=exc.status_code)


@app.get("/health")
def health():
    return {"status": "ok" if state.get("predictor") else "model_not_loaded"}


@app.get("/model-info")
def model_info():
    return _predictor().card


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    p = _predictor()
    result = p.predict([await _read_image(file)])[0]
    result["model_version"] = p.card.get("model_version")
    return result


@app.post("/predict/batch")
async def predict_batch(files: Annotated[list[UploadFile], File()]):
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"at most {MAX_BATCH} images per request")
    p = _predictor()
    images = [await _read_image(f) for f in files]
    return {"model_version": p.card.get("model_version"), "results": p.predict(images)}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
