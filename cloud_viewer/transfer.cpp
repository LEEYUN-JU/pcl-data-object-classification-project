#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl\registration\icp.h>
#include <pcl\filters\passthrough.h>

int transfer()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);    
    pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apples\\", *cloud);
    //pcl::io::loadPCDFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apples\\apple_down_pass_outfilter.pcd", *cloud);
    //pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apples\\apple_0_pass.ply", *cloud);
    //pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\apple_passthrough.ply", *cloud);

    /* Reminder: how transformation matrices work :

           |-------> This column is the translation
    | 1 0 0 x |  \
    | 0 1 0 y |   }-> The identity 3x3 matrix (no rotation) on the left
    | 0 0 1 z |  /
    | 0 0 0 1 |    -> We do not use this line (and it has to stay 0,0,0,1)

    METHOD #1: Using a Matrix4f
    This is the "manual" method, perfect to understand but error prone !
  */
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

    // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    float theta = M_PI / 20; // The angle of rotation in radians

    transform_1(0, 0) = std::cos(theta);
    transform_1(0, 1) = -sin(theta);
    transform_1(1, 0) = sin(theta);
    transform_1(1, 1) = std::cos(theta);
    //    (row, column)

    // Define a translation of 2.5 meters on the x axis.
    transform_1(3, 3) = 2.5;

    // Print the transformation
    printf("Method #1: using a Matrix4f\n");
    std::cout << transform_1 << std::endl;

    /*  METHOD #2: Using a Affine3f
    This method is easier and less error prone
  */
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

    //평행 이동
    // Define a translation of 2.5 meters on the x axis.
        
    transform_2.translation() << 0.0, 0.0, 0.0;
    
    //회전식
    // The same rotation matrix as before; theta radians around Z axis        
    transform_2.rotate(Eigen::AngleAxisf(((180.0 * M_PI) / 180), Eigen::Vector3f::UnitX()));  
       
    // Print the transformation
    printf("\nMethod #2: using an Affine3f\n");
    std::cout << transform_2.matrix() << std::endl;

    // Executing the transformation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // You can either apply transform_1 or transform_2; they are the same
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform_2);

    // Visualization
    printf("\nPoint cloud colors :  white  = original point cloud\n"
        "                        red  = transformed point cloud\n");
    pcl::visualization::PCLVisualizer viewer("Matrix transformation example");

    // Define R,G,B colors for the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler(cloud, 255, 255, 255);
    // We add the point cloud to the viewer and pass the color handler
    viewer.addPointCloud(cloud, source_cloud_color_handler, "original_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler(transformed_cloud, 230, 20, 20); // Red
    viewer.addPointCloud(transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    //viewer.setPosition(800, 400); // Setting visualiser window position

    while (!viewer.wasStopped()) { // Display the visualiser until 'q' key is pressed
        viewer.spinOnce();
    }
      
    pcl::PointCloud<pcl::PointXYZRGB> cloud2;
   
    pcl::copyPointCloud(*transformed_cloud, cloud2);

    ////cloud2.points.resize(cloud2.size());
    //for (size_t i = 0; i < cloud2.size(); i++) {
    //    cloud2.points[i].x = transformed_cloud. points[i].x;
    //    cloud_rgb.points[i].y = cloud_xyz.points[i].y;
    //    cloud_rgb.points[i].z = cloud_xyz.points[i].z;
    //}
    //
        
    pcl::io::savePLYFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apples\\apple.ply", cloud2);
    printf("파일을 저장했습니다.\n");
       
    return 0;          

    
}