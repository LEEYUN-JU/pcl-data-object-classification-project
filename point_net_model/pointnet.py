import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()
import os, glob, trimesh, json
import numpy as np
from os.path import isdir
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import sin, cos, pi

from tensorflow.python.client import device_lib ## 필요한 패키지 불러오기 ##
import keras.backend.tensorflow_backend as K ## 필요한 패키지 불러오기 ##

from keras.models import load_model

config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True #allow_growth 옵션: 초기에 메모리를 할당하지 않고 세션을 시작한 후에 더 많은 GPU 메모리가
#필요할때 메모리 영역을 확장한다.
# ex) config.gpu_options.per_process_gpu_memory_fraction = 0.6: 전체 GPU소모량을 정하고 싶을때
session = tf.compat.v1.Session(config=config) #세션 셜정

tf.compat.v1.random.set_random_seed(1234)

#########################################################################
#Load dataset
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

data_dir = r"C:\Users\.keras\datasets\ModelNet"

# mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
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

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)

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

#########################################################################
#Build a model

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.0001):
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
model.summary()
    

#########################################################################
#Train model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

print("컴파일끝!")

print("모델 피팅 시작!")
model.fit(train_dataset, epochs=20, validation_data=test_dataset)

model.save_weights('model_.h5')  #학습된 weight 저장
print("모델을 저장했습니다!")

#########################################################################
#Visualize predictions

test_list = list(test_dataset)
test_size = len(test_list)
answer = []
predic = []

for i in range(test_size):
    points, labels = test_list[i]
    
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)
    
    for i in range(len(labels)):
        answer.append(labels[i].numpy())
        predic.append(preds[i].numpy())
    
answer = np.array(answer)
predic = np.array(predic)

accuracy = np.mean(np.equal(answer,predic))

right = np.sum(answer *  predic== 1)

precision = right / np.sum(predic)

recall = right / np.sum(answer)

f1 = 2 * precision*recall/(precision+recall)

print('accuracy',accuracy)

print('precision', precision)

print('recall', recall)

print('f1', f1)

# print('accuracy', metrics.accuracy_score(answer,predic))
# print('precision', metrics.precision_score(answer,predic))
# print('recall', metrics.recall_score(answer,predic))
# print('f1', metrics.f1_score(answer,predic))

# print(metrics.classification_report(answer,predic))
# print(metrics.confusion_matrix(answer,predic))

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

#scatter : 산점도... 예 관련 뭐...
# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title("pred: {:}, label: {:} \n".format(CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]))
    ax.set_axis_off()
plt.show()