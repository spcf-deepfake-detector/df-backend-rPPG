from django.db import models


class DetectionRecord(models.Model):

    # Uploaded file types
    IMAGE = 'image'
    VIDEO = 'video'

    FILE_TYPE_CHOICES = [
        (IMAGE, 'Image'),
        (VIDEO, 'Video'),
    ]

    # Main detection results
    REAL = 'real'
    FAKE = 'fake'

    MAIN_PREDICTION_CHOICES = [
        (REAL, 'Real'),
        (FAKE, 'Fake'),
    ]

    # Fake content classifications
    FACE_SWAPPED = 'face_swapped'
    AI_GENERATED = 'ai_generated'

    FAKE_TYPE_CHOICES = [
        (FACE_SWAPPED, 'Face Swapped'),
        (AI_GENERATED, 'AI Generated'),
    ]

    # Processing status
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (PROCESSING, 'Processing'),
        (COMPLETED, 'Completed'),
        (FAILED, 'Failed'),
    ]

    # Uploaded file information
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(
        max_length=10,
        choices=FILE_TYPE_CHOICES
    )
    uploaded_file = models.FileField(upload_to='uploads/%Y/%m/')

    # Processing status
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=PENDING
    )

    # Individual model scores
    cnn_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True
    )
    vit_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True
    )
    rppg_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True
    )

    # Final detection result
    main_prediction = models.CharField(
        max_length=10,
        choices=MAIN_PREDICTION_CHOICES,
        null=True,
        blank=True
    )

    # Only applicable when the main prediction is FAKE
    fake_type = models.CharField(
        max_length=20,
        choices=FAKE_TYPE_CHOICES,
        null=True,
        blank=True
    )

    # Final confidence percentage
    confidence = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True
    )

    # Error information
    error_message = models.TextField(
        null=True,
        blank=True
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Detection Record'
        verbose_name_plural = 'Detection Records'

    def __str__(self):
        if self.main_prediction == self.FAKE and self.fake_type:
            return f"{self.file_name} - Fake ({self.get_fake_type_display()})"

        if self.main_prediction:
            return f"{self.file_name} - {self.get_main_prediction_display()}"

        return f"{self.file_name} - Pending"