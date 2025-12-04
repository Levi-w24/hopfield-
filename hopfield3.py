import numpy as np
class hopfieldNetwork:
    def __init__(self,size):
        self.size = size
        self.weight = np.zeros((size,size))

    def train(self,pattern):
        print("开始训练")
        for p in pattern:
            p = p.reshape(-1,1)
            self.weight += np.dot(p,p.T)
        np.fill_diagonal(self.weight,0)
        self.weight = self.weight / self.size
        print("记忆完成")


    def predict(self,input_pattern,max_iter=100):
        current_pattern = input_pattern.copy().astype(float)
        n =len(current_pattern)
        for q in range(max_iter):
            print(f"第{q}次修复")
            new_n = np.random.permutation(n)
            change = False
            for i in new_n:
                new_pattern = np.dot(self.weight[i],current_pattern)
                new_val = np.sign(new_pattern) if new_pattern != 0 else current_pattern[i]
                if new_val != current_pattern[i]:
                    current_pattern[i] = new_val
                    change = True
            if not change:
                print(f"第{q+1}次修复，修复已完成")
                return current_pattern.astype(int)
        print("已到最大修复次数，修复可能未完成")
        return current_pattern.astype(int)
def grid_to_vector(grid):
    return np.array(grid).flatten()
def vector_to_grid(vector):
    return vector.reshape(3,3)
one_grid =  [
    [-1, 1, -1],
    [1, 1, 1],
    [-1, -1, -1]
]

two_grid =[
    [-1, -1, -1],
    [1, 1, 1],
    [-1, 1, -1]
]

one_lvc= grid_to_vector(one_grid)

two_lvc= grid_to_vector(two_grid)

hopfield = hopfieldNetwork(9)

hopfield.train([one_lvc, two_lvc])

print("1. 网络权值矩阵W：")
# 格式化输出权值矩阵（保留2位小数，便于查看）
print(np.round(hopfield.weight,2))
print("\n")


test1_grid =  [
    [-1, 1, -1],
    [1, -1, 1],  # 噪声：中间位从1→0（对应-1）
    [-1, -1, -1]
]

test2_grid =  [
    [-1, -1, -1],
    [1, 1, -1],  # 噪声：第三位从1→0（对应-1）
    [-1, 1, -1]
]

test1_lvc= grid_to_vector(test1_grid)

test2_lvc= grid_to_vector(test2_grid)

output1 = hopfield.predict(test1_lvc)

output2 = hopfield.predict(test2_lvc)

output_grid1 = vector_to_grid(output1)

output_grid2 = vector_to_grid(output2)

print("2. 测试数据1的稳定状态：")
print(output_grid1)
print("对应字符：上\n")

print("3. 测试数据2的稳定状态：")
print(output_grid2)
print("对应字符：下")
