#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//int main(int argc, char** argv)
int Euclidean(int argc, char** argv)
{

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);

    // *.PCD ���� �б�
    pcl::io::loadPLYFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple_dataset\\apple_0.ply", *cloud);
    //pcl::io::loadPCDFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple_dataset_passthrough\\apple_0.pcd", *cloud);

    // ����Ʈ�� ���
    std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl; //*

    // Ž���� ���� KdTree ������Ʈ ���� //Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);  //KdTree ���� 


    std::vector<pcl::PointIndices> cluster_indices;       // ����ȭ�� ������� Index ����, ���� ����ȭ ��ü�� cluster_indices[0] ������ ���� 
    // ����ȭ ������Ʈ ����  
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setInputCloud(cloud);       // �Է�   
    ec.setClusterTolerance(0.002);  // 2cm  
    ec.setMinClusterSize(100);     // �ּ� ����Ʈ �� 
    ec.setMaxClusterSize(150);   // �ִ� ����Ʈ ��
    ec.setSearchMethod(tree);      // ������ ������ Ž�� ��� ���� 
    ec.extract(cluster_indices);   // ����ȭ ���� 

    // Ŭ�����ͺ� ���� ����, ���, ���� 
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // ����Ʈ�� ���
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;

        // Ŭ�����ͺ� �̸� ���� �� ���� 
        std::stringstream ss;
        ss << "apple_0_euclidean" << j << ".pcd";
        pcl::PCDWriter writer;
        writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_cluster, false); //*
        j++;
    }

    return (0);
}