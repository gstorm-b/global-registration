#include <thread>
#include <mutex>
#include <iostream>
// #include <fstream>
#include "PPF.h"
#include "configReader.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Please provide json file path";
        return -1;
    }

    ConfigReader cfg;
    std::string err;
    if (!cfg.loadFromFile(argv[1], &err)) {
        std::cerr << "Load failed: " << err << "\n";
        return -1;
    }

    std::cout << "Configuration reader load file from: " << argv[1] << std::endl;
    std::string model_path = cfg.get<std::string>("model_path", "");
    std::string model_pcd_path = cfg.get<std::string>("model_pcd_path", "");

    DescriptorPPF* descr(new DescriptorPPF());
    descr->setModelPath(model_path);
    descr->setModelPcdPath(model_pcd_path);


    auto _3D_Matching_Lambda = [&descr]() {
		// descr->loadModel();
        descr->loadModelPCD();
        descr->createSimScene();

        
		// descr->_3D_Matching();
	};
	std::thread _3D_Matching_Thread(_3D_Matching_Lambda);
	std::cout << "Processing Thread Started ... !" << std::endl;

    // Start visualizing from different thread
	while (!descr->customViewer.viewer->wasStopped()) {
		descr->customViewer.viewer->spinOnce(300);
		std::this_thread::sleep_for(std::chrono::microseconds(300000));
	}
	 
	//Wait for thread to finish before closing the program
	if (_3D_Matching_Thread.joinable()) {
		_3D_Matching_Thread.join();
    }

    if (cfg.get<bool>("model_write_to_pcd", false)) {
        descr->saveToPCD(cfg.get<std::string>("model_write_path", "model_output.pcd"));
    }

    return 0;
}