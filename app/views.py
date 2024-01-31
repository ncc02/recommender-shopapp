from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from rest_framework.views import APIView
import numpy as np
from pymongo import MongoClient

User = []
Item = {}
IndexItem = {}
UserItem = {}
UserItemBuy = {}

def CollectUserData():
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
        
    # Đóng kết nối
    client.close()


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
    page_size = 3
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


class Recommender(APIView):

    def post(self, request, format=None):
        # Lấy dữ liệu từ request body
        user_id = request.data.get('user_id', None)

        if user_id is not None:
            # Sử dụng user_id ở đây
            CollectUserData()
            a = MatrixInit()
            K = 2  #latent_factors
            beta = 0.01 #leaning_rate
            lamda = 0.02 #regularization
            epos = 1000 
            items = []
            matrix = matrix_factorization_upgrade(a, K, beta, lamda, epos)
            for index, user in enumerate(User):
                if str(user_id) == str(user):
                    items = matrix[index, :].copy()
                    vec = [(item, index) for index, item in enumerate(items)]
                    vec.sort(key=lambda x: x[0], reverse=True)

                    recommended_items = [IndexItem[index] for _, index in vec if (str(user_id), IndexItem[index]) not in UserItemBuy]
                    
                    # Phân trang dữ liệu
                    paginator = RecommenderPagination()
                    result_page = paginator.paginate_queryset(recommended_items, request)
                    
                    return paginator.get_paginated_response(result_page)
                
            # Rest of your code
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)        
        else:
            # Xử lý trường hợp không có user_id
            return Response({'error': 'Missing user_id'}, status=status.HTTP_400_BAD_REQUEST)


    