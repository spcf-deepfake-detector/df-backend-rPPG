from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    #django admin
    path('admin/', admin.site.urls),
    #main application
    path('api/', include('app.urls')),
    #deepfake detection api
    path('api/detection/', include('detection.urls'))
]