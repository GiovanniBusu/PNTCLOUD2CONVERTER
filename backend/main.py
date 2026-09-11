"""FastAPI backend: upload a point cloud, convert it to a 3DGS PLY, poll
progress, download the result. Serves the static frontend too.

Run with `python run.py` from the repo root (see README.md).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from splatconv.config import SplatParams  # noqa: E402
from splatconv.errors import SplatConvError  # noqa: E402
from splatconv.io.readers import SUPPORTED_EXTENSIONS, detect_format  # noqa: E402

from backend.jobs import UPLOAD_DIR, job_manager  # noqa: E402

app = FastAPI(title="E57/RCP -> Gaussian Splat Converter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB soft ceiling for a local tool


class ConvertParams(BaseModel):
    k_neighbors: int = 10
    scale_factor: float = Field(1.5, gt=0)
    anisotropy_ratio: float = Field(0.3, gt=0, le=1)
    opacity_dense: float = Field(0.97, gt=0, lt=1)
    opacity_sparse: float = Field(0.5, gt=0, lt=1)
    center_mode: str = "centroid"
    voxel_size: float | None = None
    include_f_rest: bool = False

    def to_splat_params(self) -> SplatParams:
        return SplatParams(
            k_neighbors=self.k_neighbors,
            scale_factor=self.scale_factor,
            anisotropy_ratio=self.anisotropy_ratio,
            opacity_dense=self.opacity_dense,
            opacity_sparse=self.opacity_sparse,
            center_mode=self.center_mode,
            voxel_size=self.voxel_size,
            include_f_rest=self.include_f_rest,
        )


@app.exception_handler(SplatConvError)
async def splatconv_error_handler(request, exc: SplatConvError):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=422, content={"detail": exc.message})


@app.get("/api/formats")
def list_formats():
    return {"supported": sorted(SUPPORTED_EXTENSIONS), "rejected_with_guidance": [".rcp", ".rcs"]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "upload.bin"
    ext = os.path.splitext(filename)[1].lower()

    # Fail fast with a clear message for .rcp/unsupported files, before
    # spending time/disk on the upload.
    try:
        detect_format(filename)
    except SplatConvError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc

    job_id_dir_name = os.urandom(6).hex()
    dest_dir = os.path.join(UPLOAD_DIR, job_id_dir_name)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    size = 0
    with open(dest_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail="Fichier trop volumineux.")
            out.write(chunk)

    if size == 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Le fichier envoyé est vide.")

    job = job_manager.create_job(filename=filename, input_path=dest_path)
    return {"job_id": job.id, "filename": filename, "size_bytes": size, "format": ext}


@app.post("/api/convert/{job_id}")
def start_convert(job_id: str, params: ConvertParams):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    if job.status == "processing":
        raise HTTPException(status_code=409, detail="Conversion déjà en cours pour ce job.")

    try:
        splat_params = params.to_splat_params()
        splat_params.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_manager.start_conversion(job_id, splat_params)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    return job.to_public_dict()


@app.get("/api/download/{job_id}")
def download_ply(job_id: str):
    job = job_manager.get(job_id)
    if job is None or job.status != "done" or not job.output_path:
        raise HTTPException(status_code=404, detail="Résultat non disponible.")
    filename = os.path.splitext(job.filename)[0] + ".splat.ply"
    return FileResponse(job.output_path, media_type="application/octet-stream", filename=filename)


@app.get("/api/download/{job_id}/offset")
def download_offset(job_id: str):
    job = job_manager.get(job_id)
    if job is None or job.status != "done" or not job.offset_sidecar_path:
        raise HTTPException(status_code=404, detail="Résultat non disponible.")
    filename = os.path.splitext(job.filename)[0] + ".offset.json"
    return FileResponse(job.offset_sidecar_path, media_type="application/json", filename=filename)


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
