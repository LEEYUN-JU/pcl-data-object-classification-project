#pragma once
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <string>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl\registration\icp.h>
#include <pcl\filters\passthrough.h>
#include <typeinfo>       // operator typeid
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <vector>
#include <ctime>
#include <conio.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int user_data_, voidnum, filenum, mindis;
float minimum, maximum, thresh = 0;
string angle = "z";
string filename = "C:\\Users\\KIMM\\Desktop\\objects\\";
string filen = "";
string passthrought_file = "";
string filename_positive = "";
string filename_neg = "";
string file_first, file_second, file_third = "";
string object = "cube_30degree_inbox\\";
string file_view_name = "cube_";
string passthrough_folder = "cube_30_degree_inbox_cut\\";

int number, func, axis, saving, filenum_1, filenum_2, times = 0;
string loadingfile, plyname, savefile, filen_ = "";
float x, y, z, angle_, radius = 0;
string file_first_, file_second_, file_third_ = "";

void viewerOneOff_(pcl::visualization::PCLVisualizer& viewer);
void viewerPsycho_(pcl::visualization::PCLVisualizer& viewer);
void pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void* viewer_void);
void pointSavingEventOccurred(const pcl::visualization::KeyboardEvent& k_event, void* viewer_void);

void viewerLoad_(int filenum, string filen);

void PassThrough(float min, float max, string angle, int filenum, string filen);

void outlier_removal(int filenum, int mindis, float thresh, string filen);

void transfer();

void Search_Kdtree(string plyname, float x, float y, float z, float radius);
void search_octree(string plyname, float x, float y, float z, float radius);
void collect_rgb(string plyname, float x, float y, float z);

void multi_viewer(int filenum_1);

//for kd tree at once, select point with mouse and get x,y,z point save to points array and set center of object
float points[20][3] = {};
float temp_points[3] = {};

int main();

//txt to ply or pcd
ifstream fin("C:\\Users\\KIMM\\Desktop\\obj_to_txt\\pear\\train\\pear_0.txt");

int array_size = 2048;
char * array_ = new char[array_size];
float obj_to_txt[2048][3];
int position = 0;
