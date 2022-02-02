#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <iostream>
#include <vector>
#include <ctime>

//int main(int argc, char** argv)
int Search_Kdtree(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::io::loadPLYFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\apple_passthrough.ply", *cloud);
    pcl::io::loadPLYFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple_dataset\\apple_0.ply", *cloud);
    //pcl::io::loadPCDFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\gitBook_Tutorial_PCL-master\\Intermediate\\sample\\cloud_cluster_0.pcd", *cloud);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].r = 255;
        cloud->points[i].g = 255;
        cloud->points[i].b = 255;
    }

    //KdTree ������Ʈ ����
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);    //�Է� 


    //������(searchPoint) ���� ��� #1(x,y,z ��ǥ ����)
    pcl::PointXYZRGB searchPoint2;
    searchPoint2.x = 0.045f;
    searchPoint2.y = -0.035f;
    searchPoint2.z = -0.39f;

    //������(searchPoint) ���� ��� #2(3000��° ����Ʈ)
    //K nearest neighbor search
    //pcl::PointXYZRGB searchPoint2 = cloud->points[3000]; //Set the lookup point
    
    //���������� ����� ������ K��°������ ����Ʈ Ž�� (K nearest neighbor search)
    int K = 10;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    //������ ��ǥ ���
    std::cout << "K nearest neighbor search at (" << searchPoint2.x
        << " " << searchPoint2.y
        << " " << searchPoint2.z
        << ") with K=" << K << std::endl;

    if (kdtree.nearestKSearch(searchPoint2, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
        {
            cloud->points[pointIdxNKNSearch[i]].r =  0;
            cloud->points[pointIdxNKNSearch[i]].g = 255;
            cloud->points[pointIdxNKNSearch[i]].b = 0;
        }
    }

    // Ž���� ���� �� ���
    std::cout << "K = 10 ��" << pointIdxNKNSearch.size() << std::endl;

    // Neighbors within radius search
    pcl::PointXYZRGB searchPoint3 = cloud->points[2000]; //Set the lookup point
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    // ���������� ������ �ݰ泻 ����Ʈ Ž�� (Neighbor search within radius)
    float radius = 0.02; //Set the search radius Ž���� �ݰ� ����(Set the search radius)

    std::cout << "Neighbors within radius search at (" << searchPoint3.x
        << " " << searchPoint3.y
        << " " << searchPoint3.z
        << ") with radius=" << radius << std::endl;


    if (kdtree.radiusSearch(searchPoint3, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
            for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
            {
                cloud->points[pointIdxRadiusSearch[i]].r = 0;
                cloud->points[pointIdxRadiusSearch[i]].g = 0;
                cloud->points[pointIdxRadiusSearch[i]].b = 255;
            }
    }

    // Ž���� ���� �� ���
    std::cout << "Radius 0.02 nearest neighbors: " << pointIdxRadiusSearch.size() << std::endl;


    pcl::io::savePCDFile<pcl::PointXYZRGB>("Kdtree_AllinOne_hand.pcd", *cloud);

    return 0;
}