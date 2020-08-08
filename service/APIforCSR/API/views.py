from django.shortcuts import render
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image, ImageOps
from numpy import asarray
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from . models import coinPicture
from . serializers import coinSerializer
from . coin_classifier import predictor

class coinPictureList(APIView):

    def get(self, request):
        coin = coinPicture.objects.all()
        serializer = coinSerializer(coin, many = True)
        return Response(serializer.data)

    def post(self,request):
        
        
        picture = request.data['picture']
        
        image = Image.open(picture)
        size = (240, 320)
        fit_and_resized_image = ImageOps.fit(image, size, Image.ANTIALIAS)
        data = asarray(fit_and_resized_image)
        data.resize(1,240,320,3)
        print(data.shape)
        output = predictor(data)
        serializer = coinSerializer(data = request.data)
        ##if serializer.is_valid():
           ##serializer.save()
        return Response(np.argmax(output[0]))
        ##else:
           ##return Response(serializer.errors,status=status.HTTP_400_BAD_REQUEST)

