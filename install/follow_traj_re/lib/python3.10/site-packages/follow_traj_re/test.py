import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
# 生成数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# 绘图
plt.plot(x, y, label='sin(x)', color='blue', linestyle='-', linewidth=2)

# 添加标题和标签
plt.title('Test Plot of sin(x)')
plt.xlabel('x value')
plt.ylabel('sin(x)')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()