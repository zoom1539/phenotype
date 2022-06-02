from scipy.optimize import linear_sum_assignment
import numpy as np

# cost = np.array([[1,2.1],
#                  [2,3],
#                  [0,1]])
# row, col = linear_sum_assignment(cost)
# print(row)
# print(col)
# print(cost[row, col])

# kc = [[1,2,3], [2,3,4]]
# l = []
# l.append({2:kc})
# print(l[0][2])

# s = set([1,2])
# # s.add(1)
# # s.add(2)
# print(not 1 in s)

# l = [1,2]
# for i in range(len(l)):
#     print(i)

# class A:
#     def __init__(self):
#         self.a = [1,2,3]
#     def run(self, bs):
#         for b in bs:
#             print(b)  
#     def run(self, a, n):
#         print(a)
    
# aa = A()
# aa.run([1,2])

a = np.array([[1,2],
             [3,4]])

rows = [0,1]
cols = [1,0]
print(a[rows, cols])