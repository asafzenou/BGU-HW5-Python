# # from scipy.cluster.hierarchy import dendrogram, linkage
# # from matplotlib import pyplot as plt
# #
# #
# # X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
# # Z = linkage(X, 'ward','euclidean')
# # fig = plt.figure(figsize=(25, 10))
# # dn = dendrogram(Z)
# # plt.show()
# import numpy as np
#
# # X = np.clip([-4, 4, 5], -3, 5)
# # print(X)
#
# # arr = np.array([[1, 2, 3], [4, 5, 6]])
# # cumulative_sum = np.cumsum(arr, axis=0)
# # print(cumulative_sum)
#
#
#
# # import numpy as np
# #
# # image = np.array([[1, 2, 3],
# #                   [4, 5, 6],
# #                   [7, 8, 9]])
# #
# # kernal_row = 3
# # kernal_col = 2
# #
# # padded_image = np.pad(image, (( kernal_row-1), ( kernal_col-1)), mode='constant')
# #
# # print(padded_image)
# #
# #
# # import numpy as np
# #
# # arr = np.array([[1, 2, 3],
# #                 [4, 5, 6]])
# #
# # padded_arr = np.pad(arr, ((0, 2), (0, 2)), mode='constant', constant_values=0)
# #
# # print(padded_arr)
#
# # print(8%3)
# #
# # print(3 * (8//3 + 1))
# #
# # x = np.array([[1,3,4], [3,4,5]])
# # print(x.shape)
# #
# # print(np.power(3,2))
#
# def root(num):
#     min = 0
#     max = num
#     r = (max + min) / 2
#     while r*r != num:
#         if r*r > num:
#             max = r
#         else:
#             min = r
#         r = (max + min) / 2
#     return r
#
#
# print(root(81))


import numpy as np

col = 30
x =  np.arange(-col/2,col/2)
print(x)