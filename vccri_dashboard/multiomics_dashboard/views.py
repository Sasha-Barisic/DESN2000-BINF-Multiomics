from django.shortcuts import render
from . import app


def home(request):
    return render(request, "home.html")
