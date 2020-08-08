from rest_framework import serializers

from . models import coinPicture

class coinSerializer(serializers.ModelSerializer):
    class Meta:
        model = coinPicture
        fields = '__all__'



