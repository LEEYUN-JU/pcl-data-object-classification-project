#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>


int user_data;
    
void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (1.0, 0.5, 1.0);
    pcl::PointXYZ o;
    o.x = 1.0;
    o.y = 0;
    o.z = 0;
    //viewer.addSphere (o, 0.25, "sphere", 0);
    std::cout << "i only run once" << std::endl;
    
}
    
void viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape ("text", 0);
    viewer.addText (ss.str(), 200, 300, "text", 0);
    
    //FIXME: possible race condition here:
    user_data++;
}
    
int cloud_viewer()
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>); 

    //���� ���� �ε�
    //pcl::io::loadPCDFile("C:\\Users\\Robot 7113\\Desktop\\gitBook_Tutorial_PCL-master\\Beginner\\sample\\tabletop.pcd", *cloud);
    //pcl::io::loadPLYFile ("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\applke.ply", *cloud);
    //pcl::io::loadPCDFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\StatisticalOutlierRemoval.pcd", *cloud);
    //pcl::io::loadPCDFile("C:\\Users\\Robot 7113\\Desktop\\gitBook_Tutorial_PCL-master\\Beginner\\sample\\table_scene_lms400.pcd", *cloud);

    //pass_through ���� ���� �ε�
    //pcl::io::loadPCDFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\tabletop_passthrough.pcd", *cloud);
    pcl::io::loadPLYFile ("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\applke_passthrough.ply", *cloud);
       
    
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    
    //blocks until the cloud is actually rendered
    viewer.showCloud(cloud);
    
    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer
    
    //This will only get called once
    viewer.runOnVisualizationThreadOnce (viewerOneOff);
    
    //This will get called once per visualization iteration
    viewer.runOnVisualizationThread (viewerPsycho);
    while (!viewer.wasStopped ())
    {
    //you can also do cool processing here
    //FIXME: Note that this is running in a separate thread from viewerPsycho
    //and you should guard against race conditions yourself...
    user_data++;
    }
    return 0;
}