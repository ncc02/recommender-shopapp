from django.db import models

class User(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    lasttime = models.DateTimeField(null=True)

    def __str__(self):
        return self.id


class Item(models.Model):
    id = models.CharField(max_length=255, primary_key=True)

    def __str__(self):
        return self.id

from django.db import models

class UserItem(models.Model):
    id_user = models.ForeignKey(User, on_delete=models.CASCADE)
    id_item = models.ForeignKey(Item, on_delete=models.CASCADE)
    rating = models.FloatField()

    def __str__(self):
        return f"{self.id_user} - {self.id_item} - Rating: {self.rating}"
