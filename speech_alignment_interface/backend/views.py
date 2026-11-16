import json
import os

import pika
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .models import Job

QUEUE = "align_jobs"
IN_DIR = os.environ.get("IN_DIR", "/data/in")


def _publish(job_id):
    conn = pika.BlockingConnection(pika.URLParameters(os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/?heartbeat=600")))
    ch = conn.channel()
    ch.queue_declare(QUEUE, durable=True)
    ch.basic_publish(
        "", QUEUE, json.dumps({"job_id": str(job_id)}),
        pika.BasicProperties(delivery_mode=2),  # persistent
    )
    conn.close()



@csrf_exempt
@require_POST
def upload(request):
    """POST multipart: audio (file), transcript (optional text), language (optional)."""
    audio = request.FILES.get("audio")
    transcript = request.POST.get("transcript", "").strip()

    if not audio:
        return JsonResponse({"error": "Please provide an audio file."}, status=400)

    job = Job.objects.create(
        transcript=transcript,
        language=request.POST.get("language", ""),
        audio_path=""
    )

    os.makedirs(IN_DIR, exist_ok=True)
    job.audio_path = os.path.join(
        IN_DIR, f"{job.id}_{os.path.basename(audio.name)}"
    )

    with open(job.audio_path, "wb") as f:
        for chunk in audio.chunks():
            f.write(chunk)

    job.save(update_fields=["audio_path"])
    _publish(job.id)

    return JsonResponse({"job_id": str(job.id), "status": job.status}, status=202)



@require_GET
def status(request, job_id):
    job = get_object_or_404(Job, pk=job_id)
    return JsonResponse({"job_id": str(job.id), "status": job.status, "error": job.error})


@require_GET
def result(request, job_id):
    """GET ?format=json (default) or ?format=textgrid"""
    job = get_object_or_404(Job, pk=job_id)
    if job.status != Job.Status.DONE:
        return JsonResponse({"status": job.status, "error": job.error}, status=409)
    if request.GET.get("format") == "textgrid":
        resp = HttpResponse(job.textgrid, content_type="text/plain; charset=utf-8")
        resp["Content-Disposition"] = f'attachment; filename="{job.id}.TextGrid"'
        return resp
    return JsonResponse({"job_id": str(job.id), "duration_sec": job.duration_sec, "words": job.words})
