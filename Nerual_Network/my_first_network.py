import numpy as np

# 激活函数 RELU函数
def activation_RELU(inputs):
  return np.maximum(inputs,0)

# 权重生成函数
def create_weights(n_inputs,n_neurons):
  # 使用正态分布，生成一个随机矩阵
  return np.random.randn(n_inputs,n_neurons)

def create_biases(n_neurons):
  return np.random.randn(n_neurons)
'''-----------------------------------------------------------'''

inputs=np.array([2,3,5])
weights=np.array([[0.2],[-0.3],[0.5]])
b1=2.0

# 简单神经元，原本的n个神经元综合计算出一个新的神经元
sum1=np.dot(inputs,weights)+b1


# 权重矩阵，刚才只能n对1，现在我们让它能够将n个神经元计算出m个新的神经元
weights2=np.array([[0.2,0.4],[-0.3,-0.5],[0.5,0.6]])
sum2=np.dot(inputs,weights2)+b1
print(sum2)
print('-'*30)

# batch处理
inputs3=np.array([[2,3,5],[6,7,8],[9,10,2],[8,4,5]])
weights3=np.array([[0.2,0.4],[-0.3,-0.5],[0.5,0.6]])
sum3=np.dot(inputs3,weights3)+b1
print(sum3)