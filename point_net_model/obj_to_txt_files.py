from matplotlib import pyplot as plt
import numpy as np
import trimesh, random, os, glob
from math import sin, cos, pi

def make_size(size, degree, axis):  
    
    ran_size = random.uniform(-3.0,3.0)
    
    ran_degree = random.randint(1, 365)
    
    ran_axis_x = random.randint(0, 2)
    ran_axis_y = random.randint(0, 2)
    ran_axis_z = random.randint(0, 2)
    ran_axis = []
    
    for i in range(100):
        while ran_size in size:
            ran_size = random.uniform(-3.0,3.0)
        size.append(ran_size)
        
        while ran_degree in degree:
            ran_degree = random.randint(1, 360)
        degree.append(ran_degree)
                
        ran_axis_x = random.randint(0, 1)
        ran_axis_y = random.randint(0, 1)
        ran_axis_z = random.randint(0, 1)
        ran_axis = [ran_axis_x, ran_axis_y, ran_axis_z]
        axis.append(ran_axis)
    
    size.sort()
    degree.sort()
    

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def read_banana_obj():
    banana = []
    with open(r"C:\Users\Desktop\point net\Data_set\obj_files_origin\banana.obj", mode="r") as f:
        for line in f:        
            if line.split(' ')[0] == 'v':            
                banana.append(line.split(' ')[1:4])
    return banana
    
def read_strawberry_obj():
    strawberrys = []
    with open(r"C:\Users\Desktop\point net\Data_set\obj_files_origin\strawberry.obj", mode="r") as f:
        for line in f:
            if line.split(' ')[0] == 'v':            
                strawberry = line.split(' ')[2:5]
                strawberry = [l.strip() for l in strawberry]
                strawberrys.append(strawberry)
    return strawberrys

def read_pear_obj():
    pears = []
    with open(r"C:\Users\Desktop\point net\Data_set\obj_files_origin\pear.obj", mode="r") as f:
        for line in f:            
            if line.split(' ')[0] == 'v':            
                pear = line.split(' ')[2:5]
                pear = [l.strip() for l in pear]
                pears.append(pear)
    return pears

def read_cube_obj():
    cubes = trimesh.load_mesh(r"C:\Users\Desktop\point net\Data_set\obj_files_origin\cube.ply")
    
    return cubes

def read_realsense():
    cube = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\cube\cube_test5.ply"))
    apple = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\apples\apple_test.ply"))
    banana = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\banana\banana_test.ply")) 
    
    return cube, apple, banana

def read_realsense_hand(objects2):
    folder_names = "_30_degree_inbox_cut"
    object_names = ["cube", "apple", "banana", "baseball", "pear", "tennis", "spam", "soup"]
    
    cube_list = []; apple_list = []; banana_list=[]; baseball_list=[]; pear_list = []; tennis_list=[]; spam_list=[]; soup_list=[];
    lists = [cube_list, apple_list, banana_list, baseball_list, pear_list, tennis_list, spam_list, soup_list]
    
    for k in range(0, 8):
            lists[k] = os.listdir(r"C:\Users\Desktop\PCL\Projects\python\%s\%s%s"%(object_names[k], object_names[k], folder_names))
            
            if k == 2:
                for j in range(0, 268):
                    objects2[k].append(list(trimesh.load_mesh(r"C:\Users3\Desktop\PCL\Projects\python\%s\%s%s\%s"%(object_names[k], object_names[k], folder_names, lists[k][j]))))
            
            else:
                for j in range(0, 360):
                    objects2[k].append(list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\%s\%s%s\%s"%(object_names[k], object_names[k], folder_names, lists[k][j]))))
            
    return objects2

def read_realsense_mix(objects):
    folder_names = ["_30_degree_inbox_cut", "_degree_cut"]
    object_names = ["cube", "apple", "banana", "baseball", "pear", "tennis", "spam", "soup"]
    
    
    for i in range(0, 2):
        cube_list = []; apple_list = []; banana_list=[]; baseball_list=[]; pear_list = []; tennis_list=[]; spam_list=[]; soup_list=[];
        lists = [cube_list, apple_list, banana_list, baseball_list, pear_list, tennis_list, spam_list, soup_list]
    
        for k in range(0, 8):
            lists[k] = os.listdir(r"C:\Users\Desktop\PCL\Projects\python\%s\%s%s"%(object_names[k], object_names[k], folder_names[i]))
            
            if i == 0:  index_num = 360
            if i == 0 and k == 2:   index_num = 260
            if i == 1:  index_num = 720            
            #if i == 2:  index_num = 600
             
            for j in range(0, index_num):
                objects[k].append(list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\%s\%s%s\%s"%(object_names[k], object_names[k], folder_names[i], lists[k][j]))))
                
    return objects


def apple_oneside():
    apple = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\apples\apple_0_translate.ply"))
    apple_list = random.sample(apple, dots)
    apple_list = np.array(apple_list)
    
    a=90.0*pi/180 #각도 (degree->radian)
    R=np.matrix([[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]]) #행렬 (x축)
    
    apple_list_T = apple_list.T
        
    apple_list_R = R * apple_list_T
    apple_list_R = apple_list_R.T
    apple_list_R = list(apple_list_R)
    
    return apple_list_R

def object_onside_apple():
    apple0 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\apples\apple_0_translate.ply"))
    apple1 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\apples\apple_1_translate.ply"))
    apple2 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\apples\apple_2_translate.ply"))
    apple3 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\apples\apple_3_translate.ply"))
    
    return apple0, apple1, apple2, apple3

def object_onside_banana():
    banana0 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\banana\banana_0_pass.ply"))
    banana1 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\banana\banana_1_translate.ply"))
    banana2 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\banana\banana_2_pass.ply"))
    banana3 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\banana\banana_3_pass.ply"))
    
    return banana0, banana1, banana2, banana3

def object_onside_cube():    
    cube0 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\cube\cube_0_translate.ply"))
    cube1 = list(trimesh.load_mesh(r"C:\Users\Desktop\PCL\Projects\python\cube\cube_1_pass.ply"))
    
    return cube0, cube1

def read_cad_half(objects): 
    
    objects[1] = trimesh.load_mesh(r"C:\Users\Desktop\half\obj_to_txt_apple_cut.ply")
    objects[2] = trimesh.load_mesh(r"C:\Users\Desktop\half\obj_to_txt_banana_cut.ply")
    objects[0] = trimesh.load_mesh(r"C:\Users\Desktop\half\obj_to_txt_cube_cut.ply")
    objects[3] = trimesh.load_mesh(r"C:\Users\Desktop\half\obj_to_txt_pear_cut.ply")
    
    return objects

def save_to_txt(object_name, folder_name):   
    
    if folder_name != 'cube_':
        for i in range(len(object_name)):
            object_name[i] = [float(x) for x in object_name[i]]
        
    for i in range(0, 100):       
                
        #수식...
        a=degree[i]*pi/180 #각도 (degree->radian)
        R=np.matrix([axis[i],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]]) #행렬 축 랜덤
        #R=np.matrix([[0,0,1],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]]) #행렬 (z축)
        
        if folder_name != 'cube_':
            object_name = list(object_name)
            object_name = random.sample(object_name, dots)
            object_name = np.array(object_name)
        elif folder_name == 'cube_':
            object_name = cube.sample(dots)
        
        
        object_name_T = object_name[i].T
        
        object_name_R = R * object_name_T
        object_name_R = object_name_R.T
        object_name_R = object_name_R * size[i]
        
        if i < 80:
            #createFolder(r"C:\Users\Desktop\point net\Data_set\for_oneside_model_making\cube\train")
            #np.savetxt(r"C:\Users\Desktop\point net\Data_set\for_oneside_model_making\cube\train\%s_%d.txt"%(folder_name, i), object_name_R)
            
            createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_hand\%s\train"%folder_name)
            np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_hand\%s\train\%s_%d.txt"%(folder_name, folder_name, i), object_name_R)
        
        else:
            #createFolder(r"C:\Users\Desktop\point net\Data_set\for_oneside_model_making\cube\test")
            #np.savetxt(r"C:\Users\Desktop\point net\Data_set\for_oneside_model_making\cube\test\%s_%d.txt"%(folder_name, i), object_name_R)
            
            
            createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_hand\%s\test"%folder_name)
            np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_hand\%s\test\%s_%d.txt"%(folder_name, folder_name, i), object_name_R)
       

def save_to_txt_hand(object_, folder_name):
    
    object_number = 0
    
    for p in range(0, 3):
        for k in range(0, 150):
            object_name = object_[k]
            object_number += 1
            
            if folder_name != 'cube_':
                for i in range(len(object_name)):
                    object_name[i] = [float(x) for x in object_name[i]]
                
                if folder_name != 'cube_':
                    object_name = list(object_name)
                    object_name = random.sample(object_name, dots)
                    object_name = np.array(object_name)
                elif folder_name == 'cube_':
                    object_name = cube.sample(dots)
                
            if k < 100:
                createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_hand2\%s\train"%folder_name)
                np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_hand2\%s\train\%s_%d.txt"%(folder_name, folder_name, object_number), object_name)
            
            else:
                createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_hand2\%s\test"%folder_name)
                np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_hand2\%s\test\%s_%d.txt"%(folder_name, folder_name, object_number), object_name)

    
def save_to_txt_hand_2048(object_, folder_name):
    
    for i in range(0, 8):
        for j in range(0, len(object_[i])):            
            for k in range(len(object_[i][j])):
                object_[i][j][k] = [float(x) for x in object_[i][j][k]]
                
    for i in range(0, 8):
        object_number = 0
        for p in range(0, 3):
            for j in range(0, len(object_[i])):            
                if len(object_[i][j]) >= 2048:
                    save_file = list(object_[i][j])
                    save_file = random.sample(save_file, 2048)
                    save_file = np.array(save_file)
                if len(object_[i][j]) < 2048:
                    save_file = list(object_[i][j])
                    a = random.sample(save_file, 1024)
                    b = random.sample(save_file, 1024)
                    save_file = np.array(a + b)
                    
                if j < 100:
                    createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_hand_2048\%s\train"%folder_name[i])
                    np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_hand_2048\%s\train\%s_%d.txt"%(folder_name[i], folder_name[i], object_number), save_file)
                
                elif j >= 100 and j < 140: 
                    createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_hand_2048\%s\test"%folder_name[i])
                    np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_hand_2048\%s\test\%s_%d.txt"%(folder_name[i], folder_name[i], object_number), save_file)
            
                object_number += 1
                
def timmer_1024(object_, folder_name):
    
    for i in range(0, 8):
        for j in range(0, len(object_[i])):            
            for k in range(len(object_[i][j])):                
                object_[i][j][k] = [float(x) for x in object_[i][j][k]]
                
    for i in range(0, 8):
        objnum = 360
        print(folder_name[i])

        #for p in range(0, 3):
        for j in range(0, len(object_[i])):
            a = []
            
            if len(object_[i][j]) >= 1024:
                save_file = list(object_[i][j])
                save_file = random.sample(save_file, 1024)
                save_file = np.array(save_file)
                
            if len(object_[i][j]) < 1024:
                
                save_file = []                
                                    
                for p in range(0, 10):
                    a.append(random.sample(list(object_[i][j]), 100))
                a.append(random.sample(list(object_[i][j]), 24))
                    
                for k in range(0, len(a)):
                    save_file += a[k]
                      
                save_file = np.array(save_file)
                
            if objnum < 960:                
                createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_degree_mix\%s\train"%folder_name[i])
                np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_degree_mix\%s\train\%s_%d.txt"%(folder_name[i], folder_name[i], objnum), save_file)
                objnum += 1
            
            elif objnum >= 960 and objnum < 1080: 
                createFolder(r"C:\Users\Desktop\point net\Data_set\realsense_degree_mix\%s\test"%folder_name[i])
                np.savetxt(r"C:\Users\Desktop\point net\Data_set\realsense_degree_mix\%s\test\%s_%d.txt"%(folder_name[i], folder_name[i], objnum), save_file)
                objnum += 1

def cad_to_txt(object_):
    folder_name = ["apple", "banana", "cube", "pear"]
        
    for i in range(0, 4):
        object_[i] = list(object_[i])
        for p in range(len(object_[i])):
            object_[i][p] = [float(x) for x in object_[i][p]]
    
    for j in range(0, 4):
        save_file = list(object_[j])        
        for k in range(0, 10):            
            save_file = random.sample(list(save_file), dots)
            save_file = np.array(save_file)
            
            createFolder(r"C:\Users\Desktop\half\%s"%folder_name[j])
            np.savetxt(r"C:\Users\Desktop\half\%s\%s_%d.txt"%(folder_name[j], folder_name[j], k), save_file)
            
    
def print_off_file(object_name):    
    mesh = trimesh.load(r"C:\Users\Desktop\point net\Data_set\ModelNet10\chair\train\chair_0001.off")
    mesh.show()
    
    points = mesh.sample(2048)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.set_axis_off()
    plt.show()

def drawing():
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(banana[:, 0], banana[:, 1], banana[:, 2])

if __name__ == "__main__":
    size = []
    degree = []
    axis = []

    dots = 1024
    list_size = 720

    make_size(size, degree, axis)
    
    cube = []; apple = []; banana = []; baseball = []; pear = []; tennis = []; spam = []; soup = [];
    cube2 = []; apple2 = []; banana2 = []; baseball2 = []; pear2 = []; tennis2 = []; spam2 = []; soup2 = [];
    
    objects = [cube, apple, banana, baseball, pear, tennis, spam, soup]
    objects2 = [cube2, apple2, banana2, baseball2, pear2, tennis2, spam2, soup2]
    
    objects = read_realsense_mix(objects)
    
    objects = read_realsense_hand(objects)
    objects2 = read_realsense_hand(objects2)
                                    
    ###################save txt to hand를 위한 리스트###############


    
    ###################3d 모델 ###############
    # # banana = read_banana_obj()
    # # strawberrys = read_strawberry_obj()
    # # pears = read_pear_obj()
    # cubes = read_cube_obj()

    
    # #apple = apple_oneside() #사과 단면 테스트 생성용
    #apple0, apple1, apple2, apple3 = object_onside_apple() #물체 단면 학습용
    #banana0, banana1, banana2, banana3 = object_onside_banana() #물체 단면 학습용
    #cube0, cube1 = object_onside_cube() #물체 단면 학습용
        
    # apples = ["apple0", "apple1", "apple2", "apple3"]
    # bananas = ["bananas0", "bananas1", "bananas2", "bananas3"]
    # cubes = ["cube0", "cube1"]

    # save_to_txt(apple3, apples[3])
    # save_to_txt(banana3, bananas[3])
    # save_to_txt(cube1, cubes[1])
    
    # folder_name = ["banana", "strawberry", "pear", "cube", "apple", "baseball", "pear", "tennis", "spam", "soup"]
    
    folder_name_2048 = ["cube", "apple", "banana", "baseball", "pear", "tennis", "spam", "soup"]
    
    
    timmer_1024(objects, folder_name_2048)
    timmer_1024(objects2, folder_name_2048)

    #drawing()
    
    #####################cad file half cutting and make txt files for testing ##########################
    # cad_files = [cube, apple, banana, pear]
    # cad_files = read_cad_half(cad_files)
    # cad_to_txt(cad_files)
