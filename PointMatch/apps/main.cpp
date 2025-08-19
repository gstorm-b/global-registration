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

    DescriptorPPF* descr(new DescriptorPPF());
    std::string file_path = argv[1];

    auto _3D_Matching_Lambda = [&descr, file_path]( ) {
        ConfigReader cfg;
        std::string err;
        if (!cfg.loadFromFile(file_path, &err)) {
            std::cerr << "Load failed: " << err << "\n";
            return -1;
        }

        std::cout << "Configuration reader load file from: " << file_path << std::endl;
        std::string model_path = cfg.get<std::string>("model_path", "");
        std::string model_pcd_path = cfg.get<std::string>("model_pcd_path", "");

        descr->setModelPath(model_path);
        descr->setModelPcdPath(model_pcd_path);

		descr->loadModel();
        // descr->loadModelPCD();
        descr->createSimScene(cfg);
        descr->match(cfg);
		// descr->_3D_Matching();

        if (cfg.get<bool>("model_write_to_pcd", false)) {
        descr->saveToPCD(cfg.get<std::string>("model_write_path", "model_output.pcd"));
    }

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

    return 0;
}