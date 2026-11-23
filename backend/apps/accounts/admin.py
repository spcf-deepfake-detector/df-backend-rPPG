from django.contrib import admin
from django.contrib import admin import UserAdmin
from .models import User


# Register your models here.
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    lisyt_display = ('username', 
                     'email', 
                     'first_name', 
                     'last_name', 
                     'is_staff')

    list_filter = ('role', 
                   'is_active',
                   'is_staff',
    )

    search_fields = ('username',
                     'email')

    ordering = ('-date_joined',)