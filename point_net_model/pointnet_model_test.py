import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()
import os, glob, trimesh, random
from os.path import isdir
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from math import sin, cos, pi

from tensorflow.python.client import device_lib ## 필요한 패키지 불러오기 ##
import keras.backend.tensorflow_backend as K ## 필요한 패키지 불러오기 ##
# print(device_lib.list_local_devices()) ## 사용 할 수 있는 연산기구 확인 ##

from keras.models import load_model


#########################################################################
#Load dataset
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

data_dir = r"C:\Users\.keras\datasets\ModelNet"


# mesh = trimesh.load(os.path.join(DATA_DIR, "apple/train/apple.off"))
# mesh.show()

# points = mesh.sample(2048)

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax.set_axis_off()
# plt.show()

def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    train_txt_file = []
    test_txt_file = []
    class_map = {}
    folders = os.listdir(r'C:\Users\.keras\datasets\ModelNet')
    # folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))
    
    k = 0
    flist = []
    for i in range(len(folders)):
        if isdir(data_dir + "\\" + folders[i]):
            flist.append(folders[i])
            k += 1
    
    for i, folder in enumerate(flist):        
        print("processing class: %s"%flist[i])
        # store folder name with ID so we can retrieve later
        class_map[i] = folder
        
        # gather all files
        train_files = glob.glob(data_dir + "\\" + folder +  "\\" + "train/*")
        test_files = glob.glob(data_dir + "\\" + folder +  "\\" + "test/*")
        
        train_txt_files = glob.glob(r'C:\Users\.keras\datasets\ModelNet\apple\train\*')
        test_txt_files = glob.glob(r'C:\Users\.keras\datasets\ModelNet\apple\test\*')
                       
        for f in train_files:            
            if folder != 'apple':
                train_points.append(trimesh.load(f).sample(num_points))
                train_labels.append(i)
        
        for f in train_txt_files:
            if folder == 'apple':            
                train_points.append(np.loadtxt(r"%s"%f))
                train_labels.append(i)                                     

        for f in test_files:            
            if folder != 'apple':
                test_points.append(trimesh.load(f).sample(num_points))
                test_labels.append(i)
        
        for f in test_txt_files:
            if folder == 'apple':      
                test_points.append(np.loadtxt(r"%s"%f))
                test_labels.append(i)   
            
            
    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)


def txtsample():   
    
    txt_files = glob.glob(r'C:\Users\Desktop\point net\Apple\*')
    last_label = max(train_labels)
    last_dict = max(CLASS_MAP)
    
    #txt 파일 불러와서 테스트 셋으로 만들기
    for i in range(len(txt_files) - 50):    
        txt_loaded = np.loadtxt(r"C:\Users\Desktop\point net\Apple\Apple_%d.txt"%i)
        train_points.append(txt_loaded)
        train_labels.append(last_label + 1)
        CLASS_MAP[last_dict+1] = 'apple'
    
    for i in range(len(txt_files) - 50, len(txt_files)):    
        txt_loaded = np.loadtxt(r"C:\Users\Desktop\point net\Apple\Apple_%d.txt"%i)
        test_points.append(txt_loaded)
        test_labels.append(last_label + 1)
        CLASS_MAP[last_dict+1] = 'apple'  



def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)
def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, 3))

with K.tf_ops.device('/device:GPU:0'):
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
# model.summary()
    

#########################################################################
#Train model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.load_weights("model_.h5")

mesh = trimesh.load_mesh(r"C:\Users\Desktop\point net\Models\Apple.off")

apple = np.array(mesh.sample(2048))


#테스트용 사과 불러오기
# mesh = trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\cloud_viewer\build\apple_passthrough_upsampling.ply")

#########################################################################
#2048사이즈로 맞추기 위해서 변환
points_ = list(mesh)
sampleList = random.sample(points_, 2048)
points_ = np.array(sampleList)

a=70*pi/180 #각도 (degree->radian)

#회전 행렬 적용 학습되었던 모양대로 바꿔주기
points_ = points_.T
points_ = (np.matrix([[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]])) * points_
points_ = points_.T
points_ = points_.reshape(1, 2048, 3)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points_[:, 0], points_[:, 1], points_[:, 2])

#########################################################################
test_list = list(test_dataset)
test_size = len(test_list)
answer = []
predic = []

count = np.zeros((10, 10), dtype=int)# 빈틀 만들기    

for i in range(test_size):
    points, labels = test_list[i]
    
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)
    
    for i in range(len(labels)):
        answer.append(labels[i].numpy())
        predic.append(preds[i].numpy())
    
answer = np.array(answer)
predic = np.array(predic)

#########################################################################
#result printing
line = np.zeros((10, 10), dtype=int)# 빈틀 만들기    
b = 0; a= 0

for j in range(0, 10):
    count = []
    for i in range(len(answer)):
        if answer[i] == j:
            count.append(predic[i])
    for k in range(len(count)):
        if count[k] == 0: line[j][0] += 1
        elif count[k] == 1: line[j][1] += 1 
        elif count[k] == 2: line[j][2] += 1
        elif count[k] == 3: line[j][3] += 1
        elif count[k] == 4: line[j][4] += 1
        elif count[k] == 5: line[j][5] += 1
        elif count[k] == 6: line[j][6] += 1
        elif count[k] == 7: line[j][7] += 1
        elif count[k] == 8: line[j][8] += 1
        elif count[k] == 9: line[j][9] += 1


print(CLASS_MAP)        
print(line)

#########################################################################
# # #Visualize predictions
data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

#테스트 셋을 맞추기 위해서 numpy로 변환
points = points.numpy()

# points[0] = points_
points[0] = apple

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title("pred: {:}, label: {:} \n".format(CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]))
    ax.set_axis_off()
plt.show()