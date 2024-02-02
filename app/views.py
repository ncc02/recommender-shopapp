from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
import random
from rest_framework.views import APIView
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta

User = []
Item = {}
IndexItem = {}
UserItem = {}
UserItemBuy = {}


def CollectUserData():
    
    # Thay đổi các thông số kết nối dựa trên thông tin của bạn
    mongodb_uri = "mongodb+srv://shopapp:shopapp@cluster0.y1hhe4z.mongodb.net/test"
    database_name = "prod"
    collection_name = "UserData"

    # Kết nối đến MongoDB
    client = MongoClient(mongodb_uri)

    # Chọn cơ sở dữ liệu và collection
    db = client[database_name]
    collection = db[collection_name]

    # Lấy tất cả các documents từ collection
    documents = collection.find()

    # In ra các documents
    for document in documents:
        for key, val in document.items():
            if (key == "_id"):
                User.append(val)
            elif (key == "recentCare"):
                for v in val: 
                    Item[v] = key
                    UserItem[(User[-1], v)]= 1
            elif (key == "recentAdd"):
                for v in val: 
                    Item[v] = key
                    UserItem[(User[-1], v)]= 2
            elif (key == "recentBuy"):
                for v in val: 
                    Item[v] = key
                    UserItem[(User[-1], v)]= 5
                    UserItemBuy[(str(User[-1]), v)] = True
        
    # Đóng kết nối
    client.close()

def CollectItems():
    Product = []
    # Thay đổi các thông số kết nối dựa trên thông tin của bạn
    mongodb_uri = "mongodb+srv://shopapp:shopapp@cluster0.y1hhe4z.mongodb.net/test"
    database_name = "prod"
    collection_name = "Product"

    # Kết nối đến MongoDB
    client = MongoClient(mongodb_uri)

    # Chọn cơ sở dữ liệu và collection
    db = client[database_name]
    collection = db[collection_name]

    # Lấy tất cả các documents từ collection
    documents = collection.find()

    # In ra các documents
    for document in documents:
        for key, val in document.items():
            if (key == "_id"):
                Product.append(str(val))
               
    # Đóng kết nối
    client.close()
    return Product


def MatrixInit():
    # print('User: ', *User)
    # print('Item: ', *Item)
    for index, item in enumerate(Item):
        IndexItem[index] = item

    for user in User:
        for item in Item:
            UserItem[(user, item)] = UserItem.get((user, item), np.nan)
            # print('user', user, '+ item', item, '->', UserItem[(user, item)])

    interaction_weights_list = [
        [UserItem.get((user, item), np.nan) for item in Item] for user in User
    ]

    return np.array(interaction_weights_list)

class RecommenderPagination(PageNumberPagination):
    page_size = 4
    page_size_query_param = 'page_size'
    max_page_size = 100


def matrix_factorization_upgrade(a, K, beta, lamda, epos):

  users, items = a.shape

  W = np.random.rand(users, K)
  H = np.random.rand(items, K)

  #upgrade
  _u = np.nanmean(a)

  bias_users = []
  bias_items = []
  for u in range(users):
    bias_users.append(np.nanmean(a[u, :]) - _u)

  for i in range(items):
    bias_items.append(np.nanmean(a[:, i]) - _u)
  #######

  a[np.isnan(a)] = 0
#   print(a)
  # Training
  for step in range(epos):
      for u in range(users):
          for i in range(items):
              if a[u, i] > 0:

                  aui = _u + bias_users[u] + bias_items[i] + np.dot(W[u, :], H[i, :])

                  error = a[u, i] - aui
                  _u = _u + beta * error
                  bias_users[u] = bias_users[u] + beta * (error - lamda * bias_users[u])
                  bias_items[i] = bias_items[i] + beta * (error - lamda * bias_items[i])

                  for k in range(K):

                      W[u, k] += beta * (error * H[i, k] - lamda * W[u, k])
                      H[i, k] += beta * (error * W[u, k] - lamda * H[i, k])
    #   print('epos',step,'loss:',error)
  matrix = np.dot(W, H.T)
  for u in range(users):
    for i in range(items):
      matrix[u][i] += _u + bias_users[u] + bias_items[i]
  return matrix

from app.models import User as Kh
from app.models import Item as Sp
from app.models import UserItem as KhSp

from django.utils import timezone
class Recommender(APIView):

    def post(self, request, format=None):
        # Lấy dữ liệu từ request body
        
        user_id = request.data.get('user_id', None)
        page_size = request.data.get('page_size', None) 
        collect_data = request.data.get('collect_data', "0")
        try:
            KH = Kh.objects.get(id=user_id)
            time_prev = KH.lasttime
            time_threshold = time_prev + timedelta(hours=12)
        except:
            collect_data = "1"

        
        if collect_data == "0" and timezone.now() <= time_threshold:
            # return Response({'now':str(timezone.now()), 'prev':str(time_threshold) })
            if user_id != None:
                user_items = KhSp.objects.filter(id_user=user_id).order_by('-rating')
            else:
                user_items = KhSp.objects.order_by('?').first()

            recommended_items = list(set(item.id_item.id for item in user_items))
            
            # Phân trang dữ liệu
            paginator = RecommenderPagination()
            if page_size:
                paginator.page_size = page_size
            result_page = paginator.paginate_queryset(recommended_items, request)
                    
            return paginator.get_paginated_response(result_page)

        else:
            # return Response({'now':str(datetime.now()), 'prev':str(time_threshold) })
            CollectUserData()
            Product = CollectItems()
            
            if user_id is not None:
                # Sử dụng user_id ở đây
                a = MatrixInit()
                K = 2  #latent_factors
                beta = 0.01 #leaning_rate
                lamda = 0.02 #regularization
                epos = 69
                items = []
                rating = {}
                matrix = matrix_factorization_upgrade(a, K, beta, lamda, epos)
                for index, user in enumerate(User):
                    if str(user_id) == str(user):
                        items = matrix[index, :].copy()
                        vec = [(item, index) for index, item in enumerate(items)]
                        vec.sort(key=lambda x: x[0], reverse=True)

                        recommended_items = [IndexItem[index] for _, index in vec if (str(user_id), IndexItem[index]) not in UserItemBuy]
                        for point, index in vec:
                            if (str(user_id), IndexItem[index]) not in UserItemBuy:
                                rating[IndexItem[index]] = point
                        for _, index in vec:
                            if (str(user_id), IndexItem[index]) in UserItemBuy:
                                rating[IndexItem[index]] = point
                                recommended_items.append(IndexItem[index])

                        product = []
                        for p in Product:
                            Found = False
                            for i in recommended_items:
                                if (str(i) == p):
                                    Found = True
                                    break
                            if not Found:
                                product.append(p)
                        
                        recommended_items += product

                        current_time = timezone.now()
                        Kh.objects.filter(id=user_id).delete()
                        user_object, created = Kh.objects.get_or_create(id=user_id)
                        user_object.lasttime = current_time
                        user_object.save()
                        
                        cnt = 0
                        for item_id in recommended_items:

                            item_object, created = Sp.objects.get_or_create(id=item_id)
                            item_object.save()
                          
                            try:
                                point = rating[item_id]
                            except:
                                point = 0
                    
                            
                            useritem_object = KhSp.objects.create(id_user=user_object, id_item=item_object, rating=point)
                            useritem_object.save()
                            

                        # Phân trang dữ liệu
                        paginator = RecommenderPagination()
                        if page_size:
                            paginator.page_size = page_size
                        result_page = paginator.paginate_queryset(recommended_items, request)
                        
                        return paginator.get_paginated_response(result_page)
                        # return Response({"cnt":str(cnt)})
                # Rest of your code
                return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)        
            else:
                # Xử lý trường hợp không có user_id
                # Trả về một mảng gồm các key trong Item (phân trang như cũ)
                
                keys = list(Item.keys())
                # print(keys)
                random.shuffle(keys)
                print('len product', len(Product))
                product = []
                for p in Product:
                    Found = False
                    for i in keys:
                        if (str(i) == p):
                            Found = True
                            break
                    if not Found:
                        product.append(p)
                
                keys += product
                # print('random', keys)
                # Phân trang dữ liệu
                paginator = RecommenderPagination()
                if page_size:
                    paginator.page_size = page_size
                result_page = paginator.paginate_queryset(keys, request)
                
                return paginator.get_paginated_response(result_page)


    