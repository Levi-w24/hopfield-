import numpy as np


class HopfieldNetwork:
    def __init__(self, size):
        """
        size: 神经元的数量（也就是输入向量的长度）
        """
        self.size = size
        # 初始化权重矩阵为全0，大小是 size x size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        训练函数：就是让网络“记住”这些 patterns
        patterns: 一个包含多个向量的列表
        """
        print("开始记忆模式...")
        for p in patterns:
            # 将向量 p 转换成矩阵形式 (列向量 x 行向量)
            p = p.reshape(-1, 1)
            # 赫布规则：权重 += 向量 * 向量的转置
            self.weights += np.dot(p, p.T)

        # 关键点：Hopfield网络中，神经元不连接自己，所以对角线设为0
        np.fill_diagonal(self.weights, 0)

        # 归一化（可选，为了数值稳定）
        self.weights /= self.size
        print("记忆完成！")

    def predict(self, input_pattern, max_iter=100):
        """
        预测/修复函数：输入一个噪点向量，输出修复后的向量
        """
        # 复制一份输入，避免修改原数据
        current_pattern = input_pattern.copy()

        for i in range(max_iter):
            print(f"第 {i + 1} 次迭代修复...")

            # 核心公式：新状态 = 符号函数(权重矩阵 x 当前状态)
            # np.dot 是矩阵乘法
            # np.sign 是符号函数：正数变1，负数变-1，0变0
            new_pattern = np.sign(np.dot(self.weights, current_pattern))

            # 处理0的情况：如果计算结果是0，保持原来的状态，或者设为1
            new_pattern[new_pattern == 0] = 1

            # 判断：如果状态不再发生变化，说明已经“收敛”了，修复完成
            if np.array_equal(new_pattern, current_pattern):
                print("网络状态已稳定！")
                return new_pattern

            current_pattern = new_pattern

        print("达到最大迭代次数，可能未完全收敛。")
        return current_pattern
# 定义将3x3网格转换为9元素向量的函数
def grid_to_vector(grid):
    # 将3x3网格按行展开为9元素的一维数组
    return np.array(grid).flatten()

# 定义将9元素向量转换为3x3网格的函数
def vector_to_grid(vector):
    # 将9元素的一维数组重塑为3x3网格
    return vector.reshape(3, 3)
# 定义字母T的3×3点阵（1=黑色，-1=白色）
T_grid = [
    [1, 1, 1],    # 第一行：全黑
    [-1, 1, -1],  # 第二行：中间黑
    [-1, 1, -1]   # 第三行：中间黑
]
# 定义字母L的3×3点阵
L_grid = [
    [1, -1, -1],  # 第一行：第一列黑
    [1, -1, -1],  # 第二行：第一列黑
    [1, 1, 1]     # 第三行：全黑
]
L_lvc= grid_to_vector(L_grid)
T_lvc= grid_to_vector(T_grid)
hopfield = HopfieldNetwork(9)
hopfield.train([T_lvc,L_lvc])
print("1. 网络权值矩阵W：")
# 格式化输出权值矩阵（保留2位小数，便于查看）
print(np.round(hopfield.weights, 2))
print("\n")
# 定义测试数据1的3×3点阵
test1_grid = [
    [-1, 1, 1],   # 第一行：后两列黑
    [-1, 1, -1],  # 第二行：中间黑
    [-1, -1, -1]  # 第三行：全白
]
# 定义测试数据2的3×3点
test2_grid = [
    [-1, -1, -1], # 第一行：全白
    [1, -1, -1],  # 第二行：第一列黑
    [-1, 1, -1]   # 第三行：中间黑
]
test1_lvc= grid_to_vector(test1_grid)
test2_lvc= grid_to_vector(test2_grid)
output1 = hopfield.predict(test1_lvc)
output2 = hopfield.predict(test2_lvc)
output_grid1 = vector_to_grid(output1)
output_grid2 = vector_to_grid(output2)

print("2. 测试数据1的稳定状态（3×3点阵）：")
print(output_grid1)
print("对应字符：T\n")

print("3. 测试数据2的稳定状态（3×3点阵）：")
print(output_grid2)
print("对应字符：L")