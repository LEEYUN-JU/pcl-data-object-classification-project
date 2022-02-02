#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>


//Filtering a PointCloud using a PassThrough filter
//http://pointclouds.org/documentation/tutorials/passthrough.php#passthrough

int PassThrough(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

	// *.PCD ���� �б� 
	//pcl::io::loadPCDFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\gitBook_Tutorial_PCL-master\\Beginner\\sample\\tabletop.pcd", *cloud);
	pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple.ply", *cloud);

	// ����Ʈ�� ���
	std::cout << "Loaded :" << cloud->width * cloud->height << std::endl;

	// ������Ʈ ���� 
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud(cloud);                //�Է� 
	pass.setFilterFieldName("z");             //������ ��ǥ �� (eg. Z��)
	//pass.setFilterLimits(0.70, 1.5);          //������ �� (�ּ�, �ִ� ��)
	pass.setFilterLimits(-3, 3);          //������ �� (�ּ�, �ִ� ��)
	//pass.setFilterLimitsNegative (true);     //������ �� �� 
	pass.filter(*cloud_filtered);             //���� ���� 

	// ����Ʈ�� ���
	std::cout << "Filtered :" << cloud_filtered->width * cloud_filtered->height << std::endl;

	// ���� 
	//pcl::io::savePCDFile<pcl::PointXYZRGB>("applke_passthrough.pcd", *cloud_filtered); //Default binary mode save
	pcl::io::savePLYFile<pcl::PointXYZRGB>("applke_passthrough.ply", *cloud_filtered); //Default binary mode save


	return (0);
}