from django.db import models


class coinPicture(models.Model):
    picture = models.ImageField(upload_to='pictures/')