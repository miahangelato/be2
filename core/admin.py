from django.contrib import admin
from .models import Fingerprint, Participant, Fingerprint

# Register your models here.
admin.site.register(Participant)
admin.site.register(Fingerprint)