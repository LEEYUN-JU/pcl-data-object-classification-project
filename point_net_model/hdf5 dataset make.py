import h5py, os, glob, trimesh
import numpy as np

from os.path import isdir

data_dir = r"C:\Users\.keras\datasets\ModelNet40"

#f = h5py.File(r'C:\Users\Desktop\dgcnn-master\pytorch\data\train.hdf5', 'w')
#f_test = h5py.File(r'C:\Users\Desktop\dgcnn-master\pytorch\data\test.hdf5', 'w')
#쓰려고자 하는 파일 동시에 두개 열린게 되어버려서 오류남.
#오류 예시: RuntimeError: Unable to create link (name already exists)

train_points = []
train_labels = []
test_points = []
test_labels = []
class_map = {}
folders = os.listdir(r'C:\Users\.keras\datasets\ModelNet40')
# folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))


def make_data(): 
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
    
        for f in train_files:
            train_points.append(trimesh.load(f).sample(1024))
            train_labels.append(i)
    
        for f in test_files:
            test_points.append(trimesh.load(f).sample(1024))
            test_labels.append(i)

def make_file(purpose):
    f = h5py.File(r'C:\Users\Desktop\dgcnn-master\pytorch\data\%s.hdf5'%purpose, 'w')

    if purpose == 'train':
        # f.create_group("tr_cloud")
        # f.create_group("tr_labels")
    
        f.create_dataset("tr_cloud", data = np.array(train_points))
        f.create_dataset("tr_labels", data = np.array(train_labels))
        
    if purpose == 'test':
        # f.create_group("test_cloud")
        # f.create_group("test_label")
        # 작성시 RuntimeError: Unable to create link (name already exists) 에러 발생
        
        f.create_dataset("test_cloud", data = np.array(test_points))
        f.create_dataset("test_labels", data = np.array(test_labels))
        
    print (f.keys())
    
    for k in f.keys():
       dataset = f[k]
       print (dataset)
    f.close()

if __name__ == '__main__':
    make_data()
    
    print('data making is done')
    
    make_file('train')
    make_file('test')
    
    
    
        
    
    