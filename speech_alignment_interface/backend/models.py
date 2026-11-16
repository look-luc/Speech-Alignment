import uuid

from django.db import models


class Job(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        DONE = "done"
        FAILED = "failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.QUEUED)
    language = models.TextField(blank=True, default="")
    transcript = models.TextField()
    audio_path = models.TextField()
    duration_sec = models.FloatField(null=True, blank=True)
    words = models.JSONField(null=True, blank=True)       # [{"word","start","end"}]
    textgrid = models.TextField(blank=True, default="")
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)   # started/finished give
    finished_at = models.DateTimeField(null=True, blank=True)  # processing time per audio minute
