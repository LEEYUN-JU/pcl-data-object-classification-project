#include "main.h"


void viewerOneOff_(pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor(1.0, 0.5, 1.0);
    pcl::PointXYZ o;
    o.x = 1.0;
    o.y = 0;
    o.z = 0;
    //viewer.addSphere (o, 0.25, "sphere", 0);
    std::cout << "i only run once" << std::endl;

}

void viewerPsycho_(pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape("text", 0);
    viewer.addText(ss.str(), 200, 300, "text", 0);

    //FIXME: possible race condition here:
    user_data_++;
}

void pointSavingEventOccurred(const pcl::visualization::KeyboardEvent& k_event, void* viewer_void) 
{
    if (k_event.getKeyCode() == 49 && k_event.isShiftPressed() == FALSE)    
    {        
        points[times][0] = temp_points[0];
        points[times][1] = temp_points[1];
        points[times][2] = temp_points[2];

        times++;
        std::cout << "포인트를 저장했습니다." << std::endl;
    }
}

void pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
    std::cout << "[INOF] Point picking event occurred." << std::endl;

    float x, y, z;
    if (event.getPointIndex() == -1)
    {
        return;
    }
    event.getPoint(x, y, z);
    std::cout << "[INOF] Point coordinate ( " << x << ", " << y << ", " << z << ")" << std::endl;
    
    temp_points[0] = x;
    temp_points[1] = y;
    temp_points[2] = z;
}

void viewerLoad_(int filenum, string filen)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

    switch (filenum)
    {
    case(1):
        filename = filename + "timmer\\" + object + filen + ".ply";
        std::cout << filename;
        pcl::io::loadPLYFile(filename, *cloud);
        break;
    case(2):
        filename = filename + "timmer_cut\\" + passthrough_folder + filen + ".ply";
        std::cout << filename;
        pcl::io::loadPLYFile(filename, *cloud);
        break;
	case(3) :
		filename = filename + filen + ".ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
	case(4) : 
		filename = filename + "30_degree_inbox\\" + object + filen + ".ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
	case(5) :
		filename = filename + "30_degree_inbox_cut\\" + passthrough_folder + filen + "_cut.ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
	case(6) :
		filename = filename + "outfilter_remove\\" + filen + "_outfilter.pcd";
		std::cout << filename;
		pcl::io::loadPCDFile(filename, *cloud);
		break;
	case(7) :
		filename = filename + "outfilter_remove\\" + filen + "_neg.pcd";
		std::cout << filename;
		pcl::io::loadPCDFile(filename, *cloud);
		break;
	case(8) :
		filename = filename + "kd_tree\\" + filen + ".ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
	case(9) :
		filename = "C:\\Users\\KIMM\\Desktop\\objects\\car\\car_" + filen + ".ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
    }    
	
    pcl::visualization::CloudViewer viewer("Cloud Viewer");

    //blocks until the cloud is actually rendered
    viewer.showCloud(cloud);

    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer

    //This will only get called once
    viewer.runOnVisualizationThreadOnce(viewerOneOff_);

    //This will get called once per visualization iteration
    viewer.runOnVisualizationThread(viewerPsycho_);

    viewer.registerPointPickingCallback(pointPickingEventOccurred, (void*)&viewer);
    viewer.registerKeyboardCallback(pointSavingEventOccurred, (void*)&viewer);
    while (!viewer.wasStopped())
    {
        //you can also do cool processing here
        //FIXME: Note that this is running in a separate thread from viewerPsycho
        //and you should guard against race conditions yourself...
        user_data_++;        
    }
    
    filename = "C:\\Users\\KIMM\\Desktop\\objects\\";
    printf("%d", cloud);
}


void PassThrough(float min, float max, string angle, int filenum, string filen)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    switch (filenum)
    {
    case(1):
        filename = filename + "timmer\\"+ object + filen + ".ply";
        std::cout << filename;
        pcl::io::loadPLYFile(filename, *cloud);
        break;
    case(2):
        filename = filename + "timmer_cut\\" + passthrough_folder + filen + ".ply";
        std::cout << filename;
        pcl::io::loadPLYFile(filename, *cloud);
        break;
	case(3) :
		filename = filename + filen + ".ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
	case(4) :
		filename = filename + "30_degree_inbox\\" + object + filen + ".ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
	case(5) :
		filename = filename + "30_degree_inbox_cut\\" + passthrough_folder + filen + "_cut.ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
		break;
    
    }
    // *.PCD 파일 읽기 
    //pcl::io::loadPCDFile<pcl::PointXYZRGB>("C:\\Users\\Robot 7113\\Desktop\\gitBook_Tutorial_PCL-master\\Beginner\\sample\\tabletop.pcd", *cloud);
    //pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple.ply", *cloud);
    //pcl::io::loadPLYFile("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\apple_passthrough.ply", *cloud);

    // 포인트수 출력
    std::cout << "Loaded :" << cloud->width * cloud->height << std::endl;

    // 오브젝트 생성 
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);                //입력 
    pass.setFilterFieldName(angle);             //적용할 좌표 축 (eg. Z축)
    //pass.setFilterLimits(0.70, 1.5);          //적용할 값 (최소, 최대 값)                     
    pass.setFilterLimits(min, max);          //적용할 값 (최소, 최대 값)
    //pass.setFilterLimitsNegative (true);     //적용할 값 외 
    pass.filter(*cloud_filtered);             //필터 적용 

    // 포인트수 출력
    std::cout << "Filtered :" << cloud_filtered->width * cloud_filtered->height << std::endl;

    // 저장
	switch (filenum)
	{
		case(1) :
			passthrought_file = "C:\\Users\\KIMM\\Desktop\\objects\\timmer_cut\\" + passthrough_folder + filen + ".ply";
			pcl::io::savePLYFile<pcl::PointXYZRGB>(passthrought_file, *cloud_filtered); //Default binary mode save
			break;
		case(2) :
			passthrought_file = "C:\\Users\\KIMM\\Desktop\\objects\\timmer_cut\\" + passthrough_folder + filen + ".ply";
			pcl::io::savePLYFile<pcl::PointXYZRGB>(passthrought_file, *cloud_filtered); //Default binary mode save
			break;
		case(3) :
			passthrought_file = "C:\\Users\\KIMM\\Desktop\\objects\\" + filen + "_cut.ply";
			pcl::io::savePLYFile<pcl::PointXYZRGB>(passthrought_file, *cloud_filtered); //Default binary mode save
			break;
		case(4) :
			passthrought_file = "C:\\Users\\KIMM\\Desktop\\objects\\30_degree_inbox_cut\\" + passthrough_folder + filen + "_cut.ply";
			pcl::io::savePLYFile<pcl::PointXYZRGB>(passthrought_file, *cloud_filtered); //Default binary mode save
			break;
		case(5) :
			passthrought_file = "C:\\Users\\KIMM\\Desktop\\objects\\30_degree_inbox_cut\\" + passthrough_folder + filen + "_cut.ply";
			pcl::io::savePLYFile<pcl::PointXYZRGB>(passthrought_file, *cloud_filtered); //Default binary mode save
			break;
	printf("파일을 저장했습니다.\n");
	}
        
    

    filename = "C:\\Users\\KIMM\\Desktop\\objects\\";
}

void outlier_removal(int filenum, int mindis, float thresh, string filen)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    switch (filenum)
    {
    case(1):
		filename = filename + "degree_cut\\" + passthrough_folder + filen + "_cut.ply";
		std::cout << filename;
		pcl::io::loadPLYFile(filename, *cloud);
        
        
    
    }
    //pcl::io::loadPLYFile<pcl::PointXYZ>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\apple_passthrough.ply", *cloud);
    //pcl::io::loadPCDFile<pcl::PointXYZ>("C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\table_scene_lms400.pcd", *cloud);
    std::cout << "Loaded " << cloud->width * cloud->height << std::endl;

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(mindis);           //분석시 고려한 이웃 점 수(50개)
    sor.setStddevMulThresh(thresh); //Outlier로 처리할 거리 정보 
    sor.filter(*cloud_filtered);

    std::cout << "Filtered " << cloud_filtered->width * cloud_filtered->height << std::endl;
	    
    filename_positive = "C:\\Users\\KIMM\\Desktop\\objects\\outfilter_remove\\" + filen + "_outfilter.pcd";
    pcl::io::savePCDFile<pcl::PointXYZRGB>(filename_positive, *cloud_filtered);
   
    sor.setNegative(true);
    sor.filter(*cloud_filtered);
    filename_neg = "C:\\Users\\KIMM\\Desktop\\objects\\outfilter_remove\\" + filen + + "_neg.pcd";
    pcl::io::savePCDFile<pcl::PointXYZRGB>(filename_neg, *cloud_filtered);

	filename = "C:\\Users\\KIMM\\Desktop\\objects\\";
}

void transfer()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::cout << "파일명을 입력해 주세요: ";
    std::cin >> plyname;

    loadingfile = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\" + object + plyname;
    pcl::io::loadPLYFile(loadingfile, *cloud);
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

    std::cout << "기능을 선택해 주세요. (0 : 평행 이동 1 : 회전) : ";
    std::cin >> func;

    if (func == 0)
    {
        //평행 이동
        // Define a translation of 2.5 meters on the x axis.
        std::cout << "x: ";
        std::cin >> x;
        std::cout << "y: ";
        std::cin >> y;
        std::cout << "z: ";
        std::cin >> z;
        transform_2.translation() << x, y, z;
    }
    else if (func == 1)
    {
        //회전식
        // The same rotation matrix as before; theta radians around Z axis
        std::cout << "angle: ";
        std::cin >> angle_;

        std::cout << "1: x, 2: y, 3: z ";
        std::cin >> axis;

        if (axis == 1) { transform_2.rotate(Eigen::AngleAxisf(((angle_ * M_PI) / 180), Eigen::Vector3f::UnitX())); }
        if (axis == 2) { transform_2.rotate(Eigen::AngleAxisf(((angle_ * M_PI) / 180), Eigen::Vector3f::UnitY())); }
        if (axis == 3) { transform_2.rotate(Eigen::AngleAxisf(((angle_ * M_PI) / 180), Eigen::Vector3f::UnitZ())); }

    }

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

    std::cout << "저장? 1: yes 2:no ";
    std::cin >> saving;

    if (saving == 1)
    {
        std::cout << "저장할 파일명을 입력해 주세요: ";
        std::cin >> plyname;
        savefile = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\" + object + plyname;
        pcl::io::savePLYFile<pcl::PointXYZRGB>(savefile, cloud2);
        printf("파일을 저장했습니다.\n");        
    }
}

void Search_Kdtree(string plyname, float x, float y, float z, float radius)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    loadingfile = filename + "30_degree\\" + object + plyname + ".ply";
    pcl::io::loadPLYFile<pcl::PointXYZRGB>(loadingfile, *cloud);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].r = 255;
        cloud->points[i].g = 255;
        cloud->points[i].b = 255;
    }

    //KdTree 오브젝트 생성
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);    //입력 


    //기준점(searchPoint) 설정 방법 #1(x,y,z 좌표 지정)
    pcl::PointXYZRGB searchPoint2;
   /* std::cout << "x: ";
    std::cin >> searchPoint2.x;
    std::cout << "y: ";
    std::cin >> searchPoint2.y;
    std::cout << "z: ";
    std::cin >> searchPoint2.z;*/

    searchPoint2.x = 0.0f;
    searchPoint2.y = 0.0f;
    searchPoint2.z = 0.0f;

    //기준점(searchPoint) 설정 방법 #2(3000번째 포인트)
    //K nearest neighbor search
    //pcl::PointXYZRGB searchPoint2 = cloud->points[3000]; //Set the lookup point

    //기준점에서 가까운 순서중 K번째까지의 포인트 탐색 (K nearest neighbor search)    
    int K = 3;

    //std::cout << "k: ";
    //std::cin >> K;

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    //기준점 좌표 출력
    std::cout << "K nearest neighbor search at (" << searchPoint2.x
        << " " << searchPoint2.y
        << " " << searchPoint2.z
        << ") with K=" << K << std::endl;

    if (kdtree.nearestKSearch(searchPoint2, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
        {
            cloud->points[pointIdxNKNSearch[i]].r = 255;
            cloud->points[pointIdxNKNSearch[i]].g = 255;
            cloud->points[pointIdxNKNSearch[i]].b = 53;
        }
    }

    // 탐색된 점의 수 출력
    std::cout << "K = 10 ：" << pointIdxNKNSearch.size() << std::endl;

    // Neighbors within radius search
    pcl::PointXYZRGB searchPoint3;

    searchPoint3.x = x;
    searchPoint3.y = y;
    searchPoint3.z = z;

    //pcl::PointXYZRGB searchPoint3 = cloud->points[2000]; //Set the lookup point
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    // 기준점에서 지정된 반경내 포인트 탐색 (Neighbor search within radius)    
    //float radius = 0.0f; //Set the search radius 탐색할 반경 설정(Set the search radius) default = 0.02

    std::cout << "Neighbors within radius search at (" << searchPoint3.x
        << " " << searchPoint3.y
        << " " << searchPoint3.z
        << ") with radius=" << radius << std::endl;


    if (kdtree.radiusSearch(searchPoint3, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
            for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
            {
                cloud->points[pointIdxRadiusSearch[i]].r = 255;
                cloud->points[pointIdxRadiusSearch[i]].g = 0;
                cloud->points[pointIdxRadiusSearch[i]].b = 0;
            }
    }

    // 탐색된 점의 수 출력
    std::cout << "Radius 0.02 nearest neighbors: " << pointIdxRadiusSearch.size() << std::endl;

    //저장할 포인트 클라우드 점
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_save(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < pointIdxRadiusSearch.size(); i++) 
    {
        cloud_save->push_back(cloud->points[pointIdxRadiusSearch[i]]);
    }

    savefile = "C:\\Users\\KIMM\\Desktop\\objects\\kd_tree\\" + plyname + ".ply";
    pcl::io::savePLYFile<pcl::PointXYZRGB>(savefile, *cloud_save);

}

void search_octree(string plyname, float x, float y, float z, float radius)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	loadingfile = filename + "30_degree\\" + object + plyname + ".ply";
	pcl::io::loadPLYFile<pcl::PointXYZRGB>(loadingfile, *cloud);

	float resolution = radius; //set the size of Voxel
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree(resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();

	pcl::PointXYZRGB searchPoint;
	searchPoint.x = x;
	searchPoint.y = y;
	searchPoint.z = z;

	//neighbors within voxel search

	std::vector<int> pointIdxVec;

	if (octree.voxelSearch(searchPoint, pointIdxVec))
	{
		std::cout << "Neighbors within vocelsearch at (" << searchPoint.x
			<< " " << searchPoint.y
			<< " " << searchPoint.z << ")"
			<< std::endl;

		for (size_t i = 0; i < pointIdxVec.size(); ++i)
		{
			std::cout << " " << cloud->points[pointIdxVec[i]].x
				<< " " << cloud->points[pointIdxVec[i]].y
				<< " " << cloud->points[pointIdxVec[i]].z
				<< std::endl;
		}
	}

	//저장할 포인트 클라우드 점
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_save(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (size_t i = 0; i < pointIdxVec.size(); i++)
	{
		cloud_save->push_back(cloud->points[pointIdxVec[i]]);
	}

	savefile = "C:\\Users\\KIMM\\Desktop\\objects\\kd_tree\\" + plyname + ".ply";
	pcl::io::savePLYFile<pcl::PointXYZRGB>(savefile, *cloud_save); 
		

	//k nearest neighbor search

	/*int K = radius;

	std::vector<int> pointIdxNKNSearch;
	std::vector<float> pointNKNSquaredDistance;

	std::cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with K=" << K << std::endl;

	if (octree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i) 
		{
			std::cout << " " << cloud->points[pointIdxNKNSearch[i]].x
				<< " " << cloud->points[pointIdxNKNSearch[i]].y
				<< " " << cloud->points[pointIdxNKNSearch[i]].z
				<< "(squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;

		}
	}

	//저장할 포인트 클라우드 점
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_save(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (size_t i = 0; i < pointIdxNKNSearch.size(); i++)
	{
		cloud_save->push_back(cloud->points[pointIdxNKNSearch[i]]);
	}

	savefile = "C:\\Users\\KIMM\\Desktop\\objects\\kd_tree\\" + plyname + ".ply";
	pcl::io::savePLYFile<pcl::PointXYZRGB>(savefile, *cloud_save);*/

}

void collect_rgb(string plyname, float x, float y, float z)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	loadingfile = filename + "degree\\" + object + plyname + ".ply";
	pcl::io::loadPLYFile<pcl::PointXYZRGB>(loadingfile, *cloud);

	pcl::PointXYZRGB searchPoint;
	int color_num = 0;

	for (size_t i = 0; i < cloud->size(); i++) 
	{		
		if (cloud->points[i].x == points[0][0] && cloud->points[i].y == points[0][1] && cloud->points[i].z == points[0][2])
		{			
			color_num = i;	
		}		
	}
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_save(new pcl::PointCloud<pcl::PointXYZRGB>);

	for (size_t i = 0; i < cloud->size(); i++) 
	{
		float a = cloud->points[i].r;
		float b = cloud->points[color_num].r;
		if (cloud->points[i].r == cloud->points[color_num].r && cloud->points[i].g == cloud->points[color_num].g && cloud->points[i].b == cloud->points[color_num].b)
		{
			
			cloud_save->push_back(cloud->points[i]);
		}
	}

	savefile = "C:\\Users\\KIMM\\Desktop\\objects\\kd_tree\\" + plyname + ".ply";
	pcl::io::savePLYFile<pcl::PointXYZRGB>(savefile, *cloud_save);
}

void multi_viewer(int filenum_1)
{    
    std::cout << "파일 이름을 입력해 주세요: ";
    std::cin >> plyname;
    string files_name[10] = {};

    for (int j = 0; j < 10; j++) 
    {
        files_name[j] = plyname + to_string(filenum_1);
        filenum_1++;         
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr apple0(new pcl::PointCloud<pcl::PointXYZ>());
    string loadingfile0 = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple_kdtree\\" + files_name[0] + ".ply";
    pcl::io::loadPLYFile<pcl::PointXYZ>(loadingfile0, *apple0);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr apple1(new pcl::PointCloud<pcl::PointXYZ>());
    string loadingfile1 = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\apple_kdtree\\" + files_name[1] + ".ply";
    pcl::io::loadPLYFile<pcl::PointXYZ>(loadingfile1, *apple1);

    pcl::visualization::PCLVisualizer viewer("multi-viewer");

    // We add the point cloud to the viewer and pass the color handler
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_0(apple0, 255, 255, 255);
    viewer.addPointCloud(apple0, handler_0, "apple_0");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(apple1, 230, 20, 20);
    viewer.addPointCloud(apple1, handler_1, "apple_1");


    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    //viewer.setPosition(800, 400); // Setting visualiser window position

    while (!viewer.wasStopped()) { // Display the visualiser until 'q' key is pressed
        viewer.spinOnce();
    }


}

//int main_main()
int main() 
{
    while (1) 
    {
        system("cls");
        printf("1: 뷰어 2: 설정창 3: 아웃필터 리무버 4: 종료 5:정합 6: 회전 및 이동 7:범위 값 설정 다중창 8:다중 파일 뷰어\n ");
        std::cout << "숫자를 입력해 주세요: ";        
        std::cin >> voidnum;
        if (voidnum != 1 &&  voidnum != 2 && voidnum != 3 && voidnum != 5 && voidnum != 6 && voidnum != 7 && voidnum != 8 && voidnum != 9 && voidnum != 10)
        {
            printf("숫자를 다시 입력해 주세요.\n");
        }

        switch (voidnum)
        {
            case(1):
            {
                printf("뷰어를 로드합니다.\n");
                std::cout << "1: timmer\\.ply 2: timmer_cut\\.ply";
                std::cin >> filenum;
                std::cout << "파일 이름을 입력해 주세요 : ";
                std::cin >> filen;
				filen = filen;
				//filen = file_view_name + filen;
                viewerLoad_(filenum, filen);
                break;
            }

            case(2):
            {
                printf("범위 값 설정 창을 로드합니다.\n");
                std::cout << "설정축: ";
                std::cin >> angle;
                std::cout << "최소값: ";
                std::cin >> minimum;
                std::cout << "최대값: ";
                std::cin >> maximum;
                std::cout << "1: timmer\\.ply 2: timmer_cut\\.ply 번호를 입력해주세요 : ";
                std::cin >> filenum;
                std::cout << "파일 이름을 입력해 주세요 : ";
                std::cin >> filen;
				filen = file_view_name + filen;

                if (angle.compare("x") == 0 || angle.compare("y") == 0 || angle.compare("z") == 0)
                    PassThrough(minimum, maximum, angle, filenum, filen);
                else
                    printf("축 설정이 잘못되었습니다.\n");
                system("pause");
                break;
            }

            case(3):
            {
                std::cout << "파일 번호를 입력해 주세요 ";
                std::cin >> filenum;                
                std::cout << "파일명을 입력해 주세요 :  ";
                std::cin >> filen;
				filen = file_view_name + filen;                
                std::cout << "거리를 입력하세요 :  ";
                std::cin >> mindis;
                std::cout << "thresh를 입력하세요 :  ";
                std::cin >> thresh;
                outlier_removal(filenum, mindis, thresh, filen);
                break;
            }

            case(4):
            {
                printf("프로그램을 종료합니다.\n");
                return 0;
            }
        
			//registration 정합
            case(5):
            {
                std::cout << "파일명을 입력하세요 :  ";
                std::cin >> file_first;
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3(new pcl::PointCloud<pcl::PointXYZRGB>);
                //filen = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\" + file_first + ".ply";
                filen = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\" + object + file_first + ".ply";
                pcl::io::loadPLYFile(filen, *cloud3);

                std::cout << "파일명을 입력하세요 :  ";
                std::cin >> file_second;
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud4(new pcl::PointCloud<pcl::PointXYZRGB>);
                //filen = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\cloud_viewer\\build\\" + file_second + ".ply";
                filen = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\" + object + file_second + ".ply";
                pcl::io::loadPLYFile(filen, *cloud4);

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud5(new pcl::PointCloud<pcl::PointXYZRGB>);

                //pcl::copyPointCloud(*cloud5, *cloud3);

                for (size_t j = 0; j < 2; j++)
                {
                    if (j == 0) {
                        for (size_t i = 0; i < cloud3->size(); i++) {
                            cloud5->push_back(cloud3->points[i]);
                        }
                    }
                    if (j == 1) {
                        for (size_t i = 0; i < cloud4->size(); i++) {
                            cloud5->push_back(cloud4->points[i]);
                        }
                    }
                }

                std::cout << "파일명을 입력하세요 :  ";
                std::cin >> file_third;
                filen = "C:\\Users\\Robot 7113\\Desktop\\PCL\\Projects\\python\\" + object + file_third + ".ply";
                pcl::io::savePLYFile<pcl::PointXYZRGB>(filen, *cloud5);
                //pcl::io::savePLYFile<pcl::PointXYZRGB>("apple_001.ply", *cloud5); //Default binary mode save
                printf("파일을 저장했습니다.\n");

                break;
            }
        
            case(6): 
                {
                transfer();
                break;
                }

            case(7):
            {
                printf("범위 값 설정 다중창을 로드합니다.\n");
                std::cout << "설정축: ";
                std::cin >> angle;
                std::cout << "최소값: ";
                std::cin >> minimum;
                std::cout << "최대값: ";
                std::cin >> maximum;
                std::cout << "1: timmer\\.ply 2: timmer_cut\\.ply 번호를 입력해주세요 : ";
                std::cin >> filenum; //2번
                //std::cout << "파일 이름을 입력해 주세요 : ";
                //std::cin >> filen;
                filen = file_view_name;
                std::cout << "파일 번호를 입력해 주세요 : ";
                std::cin >> filenum_1 >> filenum_2;

                for (int i = filenum_1; i < filenum_2; i++)
                {
                    string filen_multi_passthrough = filen + to_string(i);
                    if (angle.compare("x") == 0 || angle.compare("y") == 0 || angle.compare("z") == 0)
                        PassThrough(minimum, maximum, angle, filenum, filen_multi_passthrough);
                    else
                        printf("축 설정이 잘못되었습니다.\n");
                }
            
                break;
            }

            case(8): 
            {
                memset(points, 0, sizeof(points));
                times = 0;
                printf("연속 뷰어를 로드합니다.\n");
                std::cout << "1: timmer\\.ply 2: timmer_cut\\.ply";
                std::cin >> filenum;
                //std::cout << "파일 이름을 입력해 주세요 : ";
                //std::cin >> filen;
                filen = file_view_name;
                std::cout << "파일 번호를 입력해 주세요 : ";
                std::cin >> filenum_1 >> filenum_2;

                for (int i = filenum_1; i < filenum_2; i++)
                {
                    string filen_multi_viewer = filen + to_string(i);
                    viewerLoad_(filenum, filen_multi_viewer);
                }
                
                break;
            }

            case(9):
            {
                //std::cout << "파일 이름을 입력해 주세요 : ";
                //std::cin >> filen;
                filen = file_view_name;

				//for manual version
                /*std::cout << "좌표계를 입력해 주세요" << std::endl;
                std::cout << "x: ";
                std::cin >> x;
                std::cout << "y: ";
                std::cin >> y;
                std::cout << "z: ";
                std::cin >> z;*/

                std::cout << "지름을 입력해 주세요 : ";
                std::cin >> radius;


                std::cout << "파일 번호를 입력해 주세요 : ";
                std::cin >> filenum_1 >> filenum_2;

                int time = 0;

                for (int i = filenum_1; i < filenum_2; i++)
                {
                    string filen_kdtree = filen + to_string(i);
                    //Search_Kdtree(filen_kdtree, x, y, z, radius);
                    Search_Kdtree(filen_kdtree, points[time][0], points[time][1], points[time][2], radius);
                    time++;
                }

                system("pause");
                memset(points, 0, sizeof(points));
                std::cout << "포인트를 초기화 합니다." << std::endl;
                break;
            }

            case(10): 
            {
                std::cout << "파일 번호를 입력해 주세요 : ";
                std::cin >> filenum_1;

                multi_viewer(filenum_1);
            }

			case(11) : //convert txt file to point cloud and export ply file
			{
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud11(new pcl::PointCloud<pcl::PointXYZRGB>);

				if (fin.is_open())
				{
					cout << "File Opened successfully! Reading data from file into array" << endl;
					int line_num = 0;
					long double num = 0;

					pcl::PointXYZRGB point;

					while (!fin.eof())
					{						
						fin.getline(array_, 100);
						cout << array_ << endl;
						
						int xyz = 0;
						char* tok0 = strtok(array_, " ");						

						while (tok0 != NULL) 
						{
							num = atof(tok0);
							num = num * 0.005;
							obj_to_txt[line_num][xyz] = num;
							tok0 = strtok(NULL, " ");
							xyz++;
						}

						point.x = obj_to_txt[line_num][0];
						point.y = obj_to_txt[line_num][1];
						point.z = obj_to_txt[line_num][2];

						cloud11->push_back(point);

						line_num++;				

					}

					fin.close();
					
					filen = "C:\\Users\\KIMM\\Desktop\\objects\\obj_to_txt_pear.ply";
					pcl::io::savePLYFile<pcl::PointXYZRGB>(filen, *cloud11);

					system("pause");
				}
			}

			case(12) : 
			{
				std::cout << "파일 이름을 입력해 주세요 : ";
				std::cin >> filen;
				string filen_octree = file_view_name + filen;

				//for manual version
				std::cout << "지름을 입력해 주세요 : ";
				std::cin >> radius;
				
				collect_rgb(filen_octree, temp_points[0], temp_points[1], temp_points[2]);

				system("pause");
				break;
			}
					
        }
    }

    return 0;
}