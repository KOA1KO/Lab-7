import numpy as np
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


def firstTask():
    countOfNums = 1000000

    list1 = []
    for i in range(countOfNums):
        list1.append(random.randint(0, 100))

    list2 = []
    for i in range(countOfNums):
        list2.append(random.randint(0, 100))

    np_array1 = np.array(list1)
    np_array2 = np.array(list2)

    start_time = time.perf_counter()
    result_list = [a * b for a, b in zip(list1, list2)]
    end_time = time.perf_counter()
    list_time = end_time - start_time
    print(f"Время выполнения операции поэлементного перемножения для стандартных списков: {list_time:.5f} секунд")

    start_time = time.perf_counter()
    result_np = np.multiply(np_array1, np_array2)
    end_time = time.perf_counter()
    np_time = end_time - start_time
    print(f"Время выполнения операции поэлементного перемножения для массивов NumPy: {np_time:.5f} секунд")


def secondTask():
    data = np.genfromtxt('data2.csv', delimiter=',')
    data = data[1:]

    Y = []
    for row in data:
        Y.append(row[0])

    plt.hist(Y, bins=20, color='b')
    plt.title("data2")
    plt.xlabel("number")
    plt.ylabel("ph")

    plt.show()

    # при написании этого кода опирался на данный источник https://russianblogs.com/article/17011016846/#直方图均衡化-1
    maxValue = max(Y)
    hist, bins = np.histogram(Y, 20, [0, maxValue])

    cumSumOfList = hist.cumsum()
    cdf_normalized = cumSumOfList * float(hist.max()) / cumSumOfList.max()

    plt.plot(cdf_normalized, color='r')
    plt.xlim([0, maxValue])
    plt.title("data2")
    plt.xlabel("number")
    plt.ylabel("ph")

    plt.show()

    # среднеквадратичное отклонение/стандартное отклонение
    cleanedList = [num for num in Y if str(num) != 'nan']
    print(f'Среднеквадратичное отклонение: {np.std(cleanedList)}')


def thirdTask():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.sin(x) * np.cos(x)
    z = np.sin(x) * np.cos(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('ось X')
    ax.set_ylabel('ось Y')
    ax.set_zlabel('ось Z')

    plt.show()


def additionalTask():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    line, = ax.plot(x, y)

    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0))
        return line,

    ani = FuncAnimation(fig, animate, frames=200, interval=20, blit=True)
    ani.save('animation.gif', writer='pillow', fps=60)

    plt.show()


if __name__ == '__main__':
    firstTask()
    secondTask()
    thirdTask()
    additionalTask()
