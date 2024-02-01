from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
import random
from rest_framework.views import APIView
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta

def CollectData():
    User = []
    Item = {}
    UserItem = {}
    UserItemBuy = {}

    # Thay đổi các thông số kết nối dựa trên thông tin của bạn
    mongodb_uri = "mongodb+srv://shopapp:shopapp@cluster0.y1hhe4z.mongodb.net/test"
    database_name = "test"
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
    
    collection_name = "Product"
    # Chọn cơ sở dữ liệu và collection
    db = client[database_name]
    collection = db[collection_name]

    # Lấy tất cả các documents từ collection
    documents = collection.find()

    User.append("Guest")
    # In ra các documents
    for document in documents:
        for key, val in document.items():
            if (key == "_id"):
                Item[str(val)] = key
                UserItem[(User[-1], str(val))]= 0

    # Đóng kết nối
    client.close()
    return (User, Item, UserItem, UserItemBuy)

def MatrixInit(User, Item, UserItem):
    # print('User: ', *User) ############################3333333
    # print('Item: ', *Item) ###################################
    IndexItem = {}
    
    for index, item in enumerate(Item):
        IndexItem[index] = item

    for user in User:
        for item in Item:
            UserItem[(user, item)] = UserItem.get((user, item), np.nan)
            # print('user', user, '+ item', item, '->', UserItem[(user, item)]) ###############################

    interaction_weights_list = [
        [UserItem.get((user, item), np.nan) for item in Item] for user in User
    ]

    return np.array(interaction_weights_list), IndexItem

class RecommenderPagination(PageNumberPagination):
    page_size = 3
    page_size_query_param = 'page_size'
    max_page_size = 100

def baseline(a, _u, user, item):
  
  bias_user = np.nanmean(a[user, :]) - _u
  bias_item = np.nanmean(a[:, item]) - _u

  return _u + bias_user + bias_item

class Recommender(APIView):        
        
    def post(self, request, format=None):
        # Lấy dữ liệu từ request body
        user_id = request.data.get('user_id', None)
        page_size = request.data.get('page_size', None)

        User, Item, UserItem, UserItemBuy = CollectData()
        
        if user_id is not None:
            # Sử dụng user_id ở đây

            a, IndexItem = MatrixInit(User, Item, UserItem)
            _u = np.nanmean(a)

            matrix = []
            for u in range(User):
                v = []
                for i in range(Item):
                    v.append(baseline(a, _u, u, i))
                matrix.append(v)  

            # print(matrix) ###########################

            for index, user in enumerate(User):
                if str(user_id) == str(user):
                    items = matrix[index, :].copy()
                    vec = [(item, index) for index, item in enumerate(items)]
                    vec.sort(key=lambda x: x[0], reverse=True)

                    recommended_items = [IndexItem[index] for _, index in vec if (str(user_id), IndexItem[index]) not in UserItemBuy]
                    for _, index in vec:
                        if (str(user_id), IndexItem[index]) in UserItemBuy:
                            recommended_items.append(IndexItem[index])

                    # Phân trang dữ liệu
                    paginator = RecommenderPagination()
                    if page_size:
                        paginator.page_size = page_size
                    result_page = paginator.paginate_queryset(recommended_items, request)
                    
                    return paginator.get_paginated_response(result_page)
                
            # Rest of your code
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)        
        else:
            # Xử lý trường hợp không có user_id
            # Trả về một mảng gồm các key trong Item (phân trang như cũ)
            
            keys = list(Item.keys())
            # print(keys)
            random.shuffle(keys)
            # print('len keys', len(keys)) #################################
        
            paginator = RecommenderPagination()
            if page_size:
                paginator.page_size = page_size
            result_page = paginator.paginate_queryset(keys, request)
            
            return paginator.get_paginated_response(result_page)


    