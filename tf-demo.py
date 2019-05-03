# coding=utf-8
import tensorflow as tf

# 模型版本号
model_version = 1
# 定义模型
x = tf.placeholder(tf.float32, shape=[None, 4], name="x")
y = tf.layers.dense(x, 10, activation=tf.nn.softmax)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 模型训练过程，省略
    # ......
    
    # 保存训练好的模型到"model/版本号"中
    tf.saved_model.simple_save(
        session=sess,
        export_dir="./model/{}".format(model_version),
        inputs={"x": x},
        outputs={"y": y}
    )