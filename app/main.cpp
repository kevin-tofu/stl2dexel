
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <omp.h>
#include <igl/readSTL.h>
#include <igl/AABB.h>
#include "dexel/dexel.hpp"

using namespace dexel_sim;


int main(int argc, char** argv){
    CLI::App app("Example CLI");
    std::string stl_path = "./test.stl";
    app.add_option("-s,--stl", stl_path, "Set Path for stl file");

    CLI11_PARSE(app, argc, argv);
    std::cout << "stl_path: " << stl_path << std::endl;
    
    // FILE* stl_file = fopen(stl_path.c_str(), "rb");
    std::ifstream stl_file(stl_path, std::ios::binary);
    if (!stl_file) {
        std::cerr << "Failed to open STL file: " << stl_path << std::endl;
        return 1;
    }
    Eigen::MatrixXd V;  // Coordinates of Vertices
    Eigen::MatrixXi F;  // Indeces of Faces
    Eigen::MatrixXd N;  // NOrmal
    
    if (!igl::readSTL(stl_file, V, F, N)) {
        std::cerr << "Failed to load STL file: " << stl_path << std::endl;
        // fclose(stl_file);
        return 1;
    }
    // fclose(stl_file);

    std::cout << "Successfully loaded: " << stl_path << std::endl;
    std::cout << "Vertices: " << V.rows() << std::endl;
    std::cout << "Triangles: " << F.rows() << std::endl;
    std::cout << "Normals: " << N.rows() << std::endl;

    // double x_min, x_max, y_min, y_max;
    // getBoundingBox(
    //     V, x_min, x_max, y_min, y_max
    // );
    int x_steps = 100;
    int y_steps = 100;
    int z_steps = 100;
    // auto dexel_map = generateDexelMap(V, F, x_min, x_max, x_steps, y_min, y_max, y_steps);

    // auto dexel_map = generateDexelMap<double>(
    //     V, F,
    //     x_min, x_max, x_steps,
    //     y_min, y_max, y_steps
    // );

    dexel_sim::DexelMap<double> dexel_map(V, F, x_steps, y_steps);
    dexel_sim::TriDexelMap<double> tridexel_map(V, F, x_steps, y_steps, z_steps);

    return 0;
}
