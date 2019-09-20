import tensorflow as tf
import PIL.Image as image
from get_train import Imagedata
from get_validate import Imagedata1
import numpy as np
from PIL import ImageFont,ImageDraw
import matplotlib.pyplot as plt

batch =100
font = ImageFont.truetype('1.ttf',15)
class Encoder_cnn:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,32],dtype=tf.float32,stddev=np.sqrt(1/32)))
        self.b1 = tf.Variable(tf.zeros(shape=[32],dtype=tf.float32))
        self.w2 = tf.Variable(tf.truncated_normal(shape=[3, 3,32, 64], dtype=tf.float32, stddev=np.sqrt(1 / 64)))
        self.b2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))
        self.w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=np.sqrt(1 / 128)))
        self.b3 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))
        self.w4 = tf.Variable(tf.truncated_normal(shape=[8*15*128, 128], dtype=tf.float32, stddev=np.sqrt(1 / 128)))
        self.b4 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))
    def forward(self,x):
        y1 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(net.x,self.w1,strides=[1,1,1,1],padding='SAME')+self.b1)))
        pool_y1 = tf.nn.max_pool(y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#60*120*32
        y2 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(pool_y1,self.w2,strides=[1,1,1,1],padding='SAME')+self.b2)))
        pool_y2 = tf.nn.max_pool(y2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')# 15*30*64
        y3 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(pool_y2, self.w3, strides=[1, 1, 1, 1], padding='SAME') + self.b3)))
        pool_y3 = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')# 8*15*128
        y = tf.reshape(pool_y3,[-1,8*15*128])
        self.out = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(y,self.w4)+self.b4))

        return self.out#100,128

class Decoder_rnn:
    def __init__(self):
        self.w1 =tf.Variable(tf.truncated_normal(shape=[128,10],dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros(shape=[10],dtype=tf.float32))
    def forward(self,x):
        # 100,128-->100,1,128
        y = tf.expand_dims(x,axis=1)
        # 100,1,128-->100,4,128
        y = tf.tile(y,(1,4,1))
        cell = tf.nn.rnn_cell.LSTMCell(128)
        init_state = cell.zero_state(batch,dtype=tf.float32)
        decoder_out,decoder_finalstate = tf.nn.dynamic_rnn(cell,y,initial_state=init_state)
        y = tf.reshape(y,[batch*4,128])
        y_out= tf.nn.relu((tf.matmul(y,self.w1)+self.b1))
        self.y_out = tf.reshape(y_out,[batch,4,10])

        return self.y_out

class  Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch, 60, 120, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch, 4, 10])
        self.encoder = Encoder_cnn()
        self.decoder = Decoder_rnn()
    def forward(self):
        y = self.encoder.forward(self.x)
        self.out = self.decoder.forward(y)
    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
net = Net()
net.forward()
net.backward()
data = Imagedata()
data1 =Imagedata1()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(50000):
        xs,ys = data.getcode(batch)
        error,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs,net.y:ys})
        if i%100 ==0:
            xss,yss = data1.getcode(batch)
            error1,out = sess.run([net.loss,net.out],feed_dict={net.x:xss,net.y:yss})
            output = np.argmax(out[0],axis=1)
            label = np.argmax(yss[0], axis=1)
            print('标签：',label)
            print('输出：',output)
            acc = np.mean(np.array(np.argmax(out,axis=2)==np.argmax(yss,axis=2),dtype=np.float32))
            print('准确率：',acc)
            arr = (np.array(xss)+0.5)*255
            pic = image.fromarray(np.uint8(arr[0]))
            img = ImageDraw.Draw(pic)
            img.text((5,5),text=str(output),font=font)
            plt.imshow(pic)
            plt.pause(0.1)


