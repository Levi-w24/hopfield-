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
        current_pattern = input_pattern.copy().astype(float)
        n = len(current_pattern)

        for q in range(max_iter):

            previous_pattern = current_pattern.copy()
            change = False

            new_n = np.random.permutation(n)
            for i in new_n:
                new_pattern = np.dot(self.weights[i], current_pattern)

                new_val = np.sign(new_pattern) if new_pattern != 0 else current_pattern[i]

            # 仅当值变化时更新，并标记
                if new_val != current_pattern[i]:
                    current_pattern[i] = new_val
                    change = True
            # 收敛判断：本轮无任何神经元更新 → 状态稳定
            if not change:
                print(f" 迭代{q + 1}次后收敛，修复完成！")
                return current_pattern.astype(int)

        print(" 达到最大迭代次数，可能未完全收敛")
        return current_pattern.astype(int)


# 定义将4x4网格转换为16元素向量的函数
def grid_to_vector(grid):
    # 将4x4网格按行展开为16元素的一维数组
    return np.array(grid).flatten()

# 定义将16元素向量转换为4x4网格的函数
def vector_to_grid(vector):
    # 将16元素的一维数组重塑为4x4网格
    return vector.reshape(4, 4)
# 定义字母1的4×4点阵（1=黑色，-1=白色）
one_grid =  [
    [-1, -1, 1, -1],  # 行0：第3列（索引2）为黑色
    [-1, 1, 1, -1],  # 行1：第2列（索引1）为黑色
    [-1, -1, 1, -1],  # 行2：第3列（索引2）为黑色
    [-1, -1, 1, -1]   # 行3：第3列（索引2）为黑色
]
# 定义字母2的4×4点阵
two_grid = [
    [1, 1, -1, -1],   # 行0：前2列为黑色
    [-1, -1, 1, -1],  # 行1：第3列（索引2）为黑色
    [-1, 1, -1, -1],  # 行2：第2列（索引1）为黑色
    [1, 1, 1, 1]      # 行3：全列为黑色
]
one_lvc= grid_to_vector(one_grid)

two_lvc= grid_to_vector(two_grid)

hopfield = HopfieldNetwork(16)

hopfield.train([one_lvc, two_lvc])

print("1. 网络权值矩阵W：")
# 格式化输出权值矩阵（保留2位小数，便于查看）
print(np.round(hopfield.weights, 2))
print("\n")

# 定义测试数据1的4×4点阵
test1_grid = [
    [1, -1, 1, -1],   # 行0：第1列误激活（噪声）
    [-1, -1, -1, -1], # 行1：第2列误失活（噪声）
    [-1, -1, 1, -1],
    [-1, -1, 1, -1]
]
# 测试数据2（数字2的噪声版）
test2_grid = [
    [1, -1, 1, -1],   # 行0：第2列误激活、第2列误失活（噪声）
    [-1, -1, 1, 1],   # 行1：第4列误激活（噪声）
    [-1, 1, -1, -1],
    [1, -1, 1, 1]     # 行3：第2、4列误失活/激活（噪声）
]

test1_lvc= grid_to_vector(test1_grid)

test2_lvc= grid_to_vector(test2_grid)

output1 = hopfield.predict(test1_lvc)

output2 = hopfield.predict(test2_lvc)

output_grid1 = vector_to_grid(output1)

output_grid2 = vector_to_grid(output2)

print("2. 测试数据1的稳定状态（4×4点阵）：")
print(output_grid1)
print("对应字符：1\n")

print("3. 测试数据2的稳定状态（4×4点阵）：")
print(output_grid2)
print("对应字符：2")