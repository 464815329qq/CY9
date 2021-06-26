#Load MNIST dataset for test
import tensorflow as tf # 导入 TF 库
from tensorflow import keras # 导入 TF 子库
from tensorflow.keras import layers, optimizers, datasets # 导入 TF 子库
(x,y),(x_val,y_val) = datasets.mnist.load_data() # 加载数据集
x,x_val = x/255.0,x_val/255.0


model = keras.Sequential([ # 3个非线性层的嵌套模型
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(10,activation='softmax')])

model.compile(optimizer = 'adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])

model.fit(x,y, batch_size=32, epochs=5, validation_data=(x_val,y_val), validation_freq=1)

model.summary()


