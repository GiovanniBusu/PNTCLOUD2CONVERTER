"""In-memory job registry running conversions on a background thread pool.

Deliberately simple: this is a local single-user tool (`python run.py`), not
a multi-tenant service, so an in-process dict + ThreadPoolExecutor is enough
-- no external queue or database needed.
"""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

from splatconv.config import SplatParams
from splatconv.errors import SplatConvError
from splatconv.pipeline import ConversionResult, convert_point_cloud

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class Job:
    id: str
    filename: str
    input_path: str
    status: str = "uploaded"  # uploaded -> processing -> done | error
    progress: float = 0.0
    label: str = ""
    output_path: Optional[str] = None
    offset_sidecar_path: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None

    def to_public_dict(self) -> dict:
        d = {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 4),
            "label": self.label,
            "error": self.error,
            "result": self.result,
        }
        return d


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_job(self, filename: str, input_path: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, filename=filename, input_path=input_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def start_conversion(self, job_id: str, params: SplatParams) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.status = "processing"
        job.progress = 0.0
        job.label = "En attente..."
        job.error = None
        self._executor.submit(self._run, job_id, params)

    def _run(self, job_id: str, params: SplatParams) -> None:
        job = self.get(job_id)
        if job is None:
            return

        def on_progress(label: str, fraction: float) -> None:
            job.label = label
            job.progress = fraction

        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.splat.ply")
        try:
            result: ConversionResult = convert_point_cloud(
                job.input_path, output_path, params, progress_cb=on_progress
            )
            job.output_path = result.output_path
            job.offset_sidecar_path = result.offset_sidecar_path
            job.result = {
                "n_points_source": result.n_points_source,
                "n_splats": result.n_splats,
                "file_size_bytes": result.file_size_bytes,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "warnings": result.warnings,
            }
            job.status = "done"
            job.progress = 1.0
            job.label = "Terminé"
        except SplatConvError as exc:
            job.status = "error"
            job.error = exc.message
        except MemoryError:
            job.status = "error"
            job.error = (
                "Mémoire insuffisante pour traiter ce nuage. "
                "Essayez d'activer le sous-échantillonnage voxel."
            )
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = f"Erreur inattendue : {exc}"


job_manager = JobManager()
