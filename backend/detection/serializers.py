from rest_framework import serializers
from .models import DetectionRecord


class DetectionRecordSerializer(serializers.ModelSerializer):

    uploaded_file = serializers.FileField(read_only=True)

    class Meta:
        model = DetectionRecord

        fields = [
            'id',

            # Uploaded file information
            'file_name',
            'file_type',
            'uploaded_file',

            # Processing status
            'status',

            # Individual model scores
            'cnn_score',
            'vit_score',
            'rppg_score',

            # Final detection result
            'main_prediction',
            'fake_type',
            'confidence',

            # Error information
            'error_message',

            # Timestamps
            'created_at',
            'updated_at',
        ]

        read_only_fields = [
            'id',
            'status',
            'cnn_score',
            'vit_score',
            'rppg_score',
            'main_prediction',
            'fake_type',
            'confidence',
            'error_message',
            'created_at',
            'updated_at',
        ]


class DetectionUploadSerializer(serializers.ModelSerializer):

    class Meta:
        model = DetectionRecord

        fields = [
            'id',
            'file_name',
            'file_type',
            'uploaded_file',
        ]

        read_only_fields = [
            'id',
            'file_name',
            'file_type',
        ]

    def validate_uploaded_file(self, value):

        allowed_extensions = {
            'jpg',
            'jpeg',
            'png',
            'mp4',
            'avi',
            'mov',
        }

        extension = value.name.rsplit('.', 1)[-1].lower()

        if extension not in allowed_extensions:
            raise serializers.ValidationError(
                'Unsupported file format. '
                'Please upload JPG, JPEG, PNG, MP4, AVI, or MOV files.'
            )

        return value

    def create(self, validated_data):

        uploaded_file = validated_data['uploaded_file']

        extension = uploaded_file.name.rsplit('.', 1)[-1].lower()

        image_extensions = {'jpg', 'jpeg', 'png'}

        if extension in image_extensions:
            file_type = DetectionRecord.IMAGE
        else:
            file_type = DetectionRecord.VIDEO

        validated_data['file_name'] = uploaded_file.name
        validated_data['file_type'] = file_type

        return DetectionRecord.objects.create(**validated_data)