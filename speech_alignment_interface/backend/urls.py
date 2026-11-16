from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload, name="upload"),
    path("jobs/<uuid:job_id>/status/", views.status, name="job_status"),
    path("jobs/<uuid:job_id>/result/", views.result, name="job_result"),
]
