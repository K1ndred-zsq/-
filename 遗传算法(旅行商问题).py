import random
from matplotlib import pyplot as plt
import numpy as np
import math
plt.rcParams['font.sans-serif'] = ['SimHei']


# 城市类
class City:

    def __init__(self, size:int):
        # 城市数量
        self.size = size
        # 位置矩阵
        self.position = np.zeros((size, 2))
        x_list = random.sample(range(0, 100), size)
        y_list = random.sample(range(0, 100), size)
        self.position[:, 0] = x_list[:]
        self.position[:, 1] = y_list[:]
        # 距离矩阵D
        self.D = np.zeros((size, size), dtype='float64')
        for i in range(size):
            for j in range(size):
                self.D[i, j] = math.sqrt(pow((self.position[i, 0] - self.position[j, 0]), 2) + pow((self.position[i, 1] - self.position[j, 1]), 2))

    # 可视化方法
    def show(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.position[:, 0], self.position[:, 1], s=30, color='b')
        ax.legend()
        x_list = self.position[:, 0].copy().tolist()
        y_list = self.position[:, 1].copy().tolist()
        for a, b in zip(x_list, y_list):
            plt.text(a, b, (a,b), ha='center', va='bottom', fontsize=10)
        plt.show()



# 个体类
class Individuality:

    def __init__(self, city: City):
        # 染色体
        self.chromosome = []
        # 适应度
        self.fitness = 0.0
        self.size = city.size
        self.city = city

    # 设置起点城市完善个体
    def initByStart(self, num):
        self.chromosome = [i for i in range(self.size)]
        random.shuffle(self.chromosome)
        self.chromosome.remove(num)
        self.chromosome.insert(0, num)
        self.setFitness()

    # 根据染色体完善个体
    def initByChromosome(self, chromosome):
        self.chromosome = chromosome
        self.setFitness()

    # 计算适应度
    def setFitness(self):
        for i in range(self.size - 1):
            self.fitness += self.city.D[self.chromosome[i], self.chromosome[i + 1]]
        self.fitness += self.city.D[self.chromosome[-1], self.chromosome[0]]
    
    # 可视化方法
    def show(self, title:str):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.city.position[:, 0], self.city.position[:, 1], s=30, color='b')
        ax.legend()
        plt.plot(self.city.position[self.chromosome, 0], self.city.position[self.chromosome, 1], 'r')
        plt.plot([self.city.position[self.chromosome[-1], 0], self.city.position[self.chromosome[0], 0]], [self.city.position[self.chromosome[-1], 1], self.city.position[self.chromosome[0], 1]], 'r')
        plt.title(title+f"/适应度:{self.fitness}")
        x_list = self.city.position[:, 0].copy().tolist()
        y_list = self.city.position[:, 1].copy().tolist()
        for a, b in zip(x_list, y_list):
            plt.text(a, b, (a,b), ha='center', va='bottom', fontsize=10)
        plt.show()


class Train:

    def __init__(self, city:City, lens, groups):
        # 导入城市布局
        self.city = city
        # 种群列表
        self.gen = []
        # 组大小
        self.lens = lens
        # 组数
        self.groups = groups
        # 种群大小
        self.size = lens * groups
    
    # 初始化种族
    def init_gen(self):
        self.gen = []
        for i in range(self.groups):
            for j in range(self.lens):
                ind = Individuality(self.city)
                ind.initByStart(i)
                self.gen.append(ind)

    # 交叉
    def cross(self):
        new_chros = []
        # 按顺序两两交叉,若总个数为奇数则忽略最后一名，所以建议总数为偶数
        for i in range(self.size // 2):
            chro1 = self.gen[i * 2].chromosome.copy()
            chro2 = self.gen[i * 2 + 1].chromosome.copy()
            # 随机交换片段
            index1 = random.randint(0, self.city.size - 2)
            index2 = random.randint(index1 + 1, self.city.size - 1)
            vk1 = {v: k for k, v in enumerate(chro1)}
            vk2 = {v: k for k, v in enumerate(chro2)}
            for j in range(index1, index2 + 1):
                v1 = chro1[j]
                v2 = chro2[j]
                nk1 = vk1[v2]
                nk2 = vk2[v1]
                chro1[j], chro1[nk1] = chro1[nk1], chro1[j]
                chro2[j], chro2[nk2] = chro2[nk2], chro2[j]
                vk1[v1], vk1[v2] = nk1, j
                vk2[v1], vk2[v2] = j, nk2
            new_chros.append(chro1)
            new_chros.append(chro2)
        return new_chros
 
    # 变异
    def mutate(self, new_chros):
        for chro in new_chros:
            # 50%概率变异
            n = random.randint(0, 1)
            if n == 0:
                self.gen.append(ind)
                continue
            index1 = random.randint(0, self.city.size - 2)
            index2 = random.randint(index1 + 1, self.city.size - 1)
            mu_chro = chro[index1:index2 + 1]
            random.shuffle(mu_chro)
            new_chro = chro[:index1]
            new_chro.extend(mu_chro)
            new_chro.extend(chro[index2 + 1: ])
            ind = Individuality(self.city)
            ind.initByChromosome(new_chro)
            self.gen.append(ind)

    # 选择迭代
    def select(self):
        self.mySort()
        self.gen = self.gen[:self.size]

    # 排序
    def mySort(self):
        for i in range(self.size * 2 - 1):
            for j in range(i + 1, self.size * 2):
                if self.gen[i].fitness > self.gen[j].fitness:
                    self.gen[i], self.gen[j] = self.gen[j], self.gen[i]
    
    # 进化一次
    def train(self, n):
        self.mutate(self.cross())
        self.select()
        print(f"第{n}次迭代最优个体适应度：{self.gen[0].fitness}")



if __name__ == '__main__':
    # 迭代次数
    count = 1000
    # 10个城市
    city = City(10)
    # 种群初始化，4组，每组10个，共40个个体
    train = Train(city, 10, 4)
    train.init_gen()
    for i in range(count):
        train.train(i + 1)
    train.gen[0].show(f"{count}次迭代最优解")

# 枚举验证方法
# def dfs(chro: list, city: City):
#     if len(chro) == city.size:
#         ind = Individuality(city)
#         ind.initByChromosome(chro)
#         return ind
#     inds = []
#     for i in range(city.size):
#         if i not in chro:
#             new_chro = chro.copy()
#             new_chro.append(i)
#             ind = dfs(new_chro, city)
#             inds.append(ind)
#     n = len(inds)
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             if(inds[j].fitness < inds[i].fitness):
#                 inds[i], inds[j] = inds[j], inds[i]
#     return inds[0]
