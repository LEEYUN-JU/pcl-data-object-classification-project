#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

int icp_registration(int argc, char** argv)
//int main(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::io::loadPLYFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apples\\apple_0_translate.ply", *cloud_in);
	pcl::io::loadPLYFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apples\\apple_1_roatation.ply", *cloud_out);
	/*pcl::io::loadPCDFile<pcl::PointXYZ>("room_scan1.pcd", *cloud_in);
	pcl::io::loadPCDFile<pcl::PointXYZ>("room_scan2.pcd", *cloud_out);*/


	pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_out);
	pcl::PointCloud<pcl::PointXYZRGB> Final;
	icp.align(Final);

	std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;
	pcl::io::savePCDFile("Registration_final.pcd", Final);

	return (0);
}