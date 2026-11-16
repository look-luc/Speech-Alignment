
import os
import json

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "speech_alignment_interface.settings",
)

import django
django.setup()

import pika
from django.utils import timezone

from speech_alignment_interface.backend.models import Job
from src.speech_alignment.aligner import Aligner, load_audio, TARGET_SR
from src.speech_alignment.to_textgrid import words_to_textgrid
from src.speech_alignment.asr import ASR

QUEUE = "align_jobs"
RABBITMQ_URL = os.environ.get(
    "RABBITMQ_URL",
    "amqp://guest:guest@localhost:5672/?heartbeat=600",
)

OUT_DIR = os.path.join("data", "out")


def process_job(job_id, aligner, asr):
    job = Job.objects.get(pk=job_id)

    job.status = Job.Status.RUNNING
    job.started_at = timezone.now()
    job.error = ""
    job.save(update_fields=["status", "started_at", "error"])

    audio = load_audio(job.audio_path)
    duration = len(audio) / TARGET_SR

    transcript = job.transcript
    language = job.language or None

    if not transcript:
        print(f"[*] No transcript provided for job {job.id}. Running ASR...")

        transcript, detected_language = asr.transcribe(
            job.audio_path,
            language=language,
        )

        language = detected_language or language

        print(f"[*] ASR transcript: {transcript}")
        print(f"[*] Detected language: {language}")

        job.transcript = transcript
        job.language = language
        job.save(update_fields=["transcript", "language"])

    words = aligner.align(
        audio,
        transcript,
        language=language,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    textgrid_path = os.path.join(OUT_DIR, f"{job.id}.TextGrid")

    words_to_textgrid(words, duration, textgrid_path)

    with open(textgrid_path, "r", encoding="utf-8") as f:
        textgrid_content = f.read()

    job.duration_sec = duration
    job.words = words
    job.textgrid = textgrid_content
    job.status = Job.Status.DONE
    job.finished_at = timezone.now()
    job.save()

    print(f"[x] Job {job.id} completed successfully")


def callback(ch, method, properties, body, aligner, asr):
    try:
        message = json.loads(body.decode())
        job_id = message["job_id"]

        print(f"[x] Processing job {job_id}")
        process_job(job_id, aligner, asr)

    except Exception as exc:
        print(f"[!] Job failed: {exc}")

        try:
            if "job_id" in locals():
                job = Job.objects.get(pk=job_id)
                job.status = Job.Status.FAILED
                job.error = str(exc)
                job.finished_at = timezone.now()
                job.save()
        except Exception as db_exc:
            print(f"[!] Could not update failed job: {db_exc}")

    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    print("[*] Loading speech alignment model...")
    aligner = Aligner()
    print("[*] Alignment model loaded")

    print("[*] Loading ASR model...")
    asr = ASR()
    print("[*] ASR model loaded")

    connection = pika.BlockingConnection(
        pika.URLParameters(RABBITMQ_URL)
    )

    channel = connection.channel()
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=QUEUE,
        on_message_callback=lambda ch, method, properties, body:
            callback(ch, method, properties, body, aligner, asr),
    )

    print(f"[*] Waiting for jobs in '{QUEUE}'")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[*] Stopping worker")
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
