import matplotlib.pyplot as plt

# 将不同的参数组合和对应的准确率保存到两个列表中
components_list = list(range(10, 201, 10))
k_list = list(range(1, 11))
accuracy_lists = []

for line in open('result1.txt', 'r'):
    accuracy_lists.append(float(line.split('：')[-1]))

# print(accuracy_list)

fig = plt.figure(figsize=(10, 6))
plt.grid(linestyle='--', color='gray', linewidth=0.5)

# 绘制折线图
for k in k_list:
    accuracy_list = []
    for i in range(0, 20):
        accuracy_list.append(accuracy_lists[i * 10 + k - 1])
    # print(accuracy_list)
    plt.plot(components_list, accuracy_list, marker='o', label=f'k={k}')

plt.legend(loc='upper left')

plt.legend()
plt.xlabel('PCA components')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. PCA components and k')
plt.show()
