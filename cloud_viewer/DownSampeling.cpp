#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

int DownSampling(int argc, char** argv)
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());


    //pcl::PCDReader reader;
    pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\apple_passthrough.ply", *cloud);
    //reader.read("table_scene_lms400.pcd", *cloud); // Remember to download the file first!

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height  //cloud_filtered->points.size()
        << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(3.0f, 2048.0f, 1.0f); //The size of the body is 1 * 1 cm 
    sor.filter(*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
        << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;    
    
    pcl::PCDWriter writer;
    writer.write("apple_passthrough_downsampled.pcd", *cloud_filtered,
        Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false);

    pcl::io::savePLYFile("apple_passthrough_downsampled.ply", *cloud_filtered); //Default binary mode save
    return (0);
}