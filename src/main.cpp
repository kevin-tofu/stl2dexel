
#include <algorithm>
#include <iostream>
#include <vector>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <omp.h>
#include <igl/readSTL.h>
#include <igl/AABB.h>


class Dexel {
    public:
        std::vector<double> z_values;
    
        void add(double z) {
            z_values.push_back(z);
        }
    
        void sort() {
            std::sort(z_values.begin(), z_values.end());
        }
    
        std::vector<double> getZValues() const {
            return z_values;
        }
    
        void print() const {
            for (double z : z_values) {
                std::cout << z << " ";
            }
            std::cout << std::endl;
        }
    };
    

void getBoundingBox(
    const Eigen::MatrixXd& V,
    double& x_min, double& x_max, double& y_min, double& y_max
) {
    x_min = y_min = std::numeric_limits<double>::max();
    x_max = y_max = std::numeric_limits<double>::lowest();

    for (int i = 0; i < V.rows(); ++i) {
        x_min = std::min(x_min, V(i, 0));
        x_max = std::max(x_max, V(i, 0));
        y_min = std::min(y_min, V(i, 1));
        y_max = std::max(y_max, V(i, 1));
    }
}

void findIntersections(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double x, double y,
    Dexel& dexel
)
{
    for (int i = 0; i < F.rows(); ++i)
    {
        Eigen::Vector3d v0 = V.row(F(i, 0));
        Eigen::Vector3d v1 = V.row(F(i, 1));
        Eigen::Vector3d v2 = V.row(F(i, 2));

        Eigen::Vector2d p0(v0.x(), v0.y());
        Eigen::Vector2d p1(v1.x(), v1.y());
        Eigen::Vector2d p2(v2.x(), v2.y());

        Eigen::Matrix2d A;
        A << (p1 - p0), (p2 - p0);
        Eigen::Vector2d b(x - p0.x(), y - p0.y());

        if (A.determinant() == 0) continue;

        Eigen::Vector2d lambdas = A.colPivHouseholderQr().solve(b);
        double u = lambdas[0], v = lambdas[1], w = 1.0 - u - v;

        if (u >= 0 && v >= 0 && w >= 0) {
            double z = w * v0.z() + u * v1.z() + v * v2.z();
            dexel.add(z);
        }
    }

    dexel.sort();
}

void findIntersections_AABB(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double x, double y,
    Dexel& dexel
) {
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(V, F);  // Initialize AABB Tree

    for (int i = 0; i < F.rows(); ++i) {
        // Vertice
        Eigen::Vector3d v0 = V.row(F(i, 0));
        Eigen::Vector3d v1 = V.row(F(i, 1));
        Eigen::Vector3d v2 = V.row(F(i, 2));

        // filtering with BBOX (preprocess)
        double x_min = std::min({v0.x(), v1.x(), v2.x()});
        double x_max = std::max({v0.x(), v1.x(), v2.x()});
        double y_min = std::min({v0.y(), v1.y(), v2.y()});
        double y_max = std::max({v0.y(), v1.y(), v2.y()});

        // Check if (x, y) is on the triagnle on AABB
        if (x < x_min || x > x_max || y < y_min || y > y_max) {
            continue;  // Skip if it is out
        }

        // Compute intersection
        Eigen::Vector2d p0(v0.x(), v0.y());
        Eigen::Vector2d p1(v1.x(), v1.y());
        Eigen::Vector2d p2(v2.x(), v2.y());

        Eigen::Matrix2d A;
        A << (p1 - p0), (p2 - p0);
        Eigen::Vector2d b(x - p0.x(), y - p0.y());

        if (A.determinant() == 0) continue;

        Eigen::Vector2d lambdas = A.colPivHouseholderQr().solve(b);
        double u = lambdas[0], v = lambdas[1], w = 1.0 - u - v;

        if (u >= 0 && v >= 0 && w >= 0) {
            double z = w * v0.z() + u * v1.z() + v * v2.z();
            dexel.add(z);
        }
    }

    dexel.sort();
}



void findIntersections_OMP(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double x, double y,
    Dexel& dexel
) {
    #pragma omp parallel for
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3d v0 = V.row(F(i, 0));
        Eigen::Vector3d v1 = V.row(F(i, 1));
        Eigen::Vector3d v2 = V.row(F(i, 2));

        double x_min = std::min({v0.x(), v1.x(), v2.x()});
        double x_max = std::max({v0.x(), v1.x(), v2.x()});
        double y_min = std::min({v0.y(), v1.y(), v2.y()});
        double y_max = std::max({v0.y(), v1.y(), v2.y()});

        if (x < x_min || x > x_max || y < y_min || y > y_max) {
            continue;
        }

        Eigen::Vector2d p0(v0.x(), v0.y());
        Eigen::Vector2d p1(v1.x(), v1.y());
        Eigen::Vector2d p2(v2.x(), v2.y());

        Eigen::Matrix2d A;
        A << (p1 - p0), (p2 - p0);
        Eigen::Vector2d b(x - p0.x(), y - p0.y());

        if (A.determinant() == 0) continue;

        Eigen::Vector2d lambdas = A.colPivHouseholderQr().solve(b);
        double u = lambdas[0], v = lambdas[1], w = 1.0 - u - v;

        if (u >= 0 && v >= 0 && w >= 0) {
            double z = w * v0.z() + u * v1.z() + v * v2.z();
            #pragma omp critical
            dexel.add(z);
        }
    }

    dexel.sort();
}

std::vector<std::vector<Dexel>> generateDexelMap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, 
    double x_min, double x_max, int x_steps,
    double y_min, double y_max, int y_steps)
{
    double dx = (x_max - x_min) / (x_steps - 1);
    double dy = (y_max - y_min) / (y_steps - 1);

    std::vector<std::vector<Dexel>> dexel_map(x_steps, std::vector<Dexel>(y_steps));

    // Generate Dexel by scanning
    for (int i = 0; i < x_steps; ++i) {
        double x = x_min + i * dx;
        for (int j = 0; j < y_steps; ++j) {
            double y = y_min + j * dy;
            // findIntersections(V, F, x, y, dexel_map[i][j]);
            findIntersections_AABB(V, F, x, y, dexel_map[i][j]);
            // findIntersections_OMP(V, F, x, y, dexel_map[i][j]);
        }
    }

    return dexel_map;
}


int main(int argc, char** argv){
    CLI::App app("Example CLI");
    std::string stl_path = "./test.stl";
    app.add_option("-s,--stl", stl_path, "Set Path for stl file");

    CLI11_PARSE(app, argc, argv);
    std::cout << "stl_path: " << stl_path << std::endl;
    
    FILE* stl_file = fopen(stl_path.c_str(), "rb");
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

    double x_min, x_max, y_min, y_max;
    getBoundingBox(
        V, x_min, x_max, y_min, y_max
    );
    int x_steps = 100;
    int y_steps = 100;
    // auto dexel_map = generateDexelMap(V, F, x_min, x_max, x_steps, y_min, y_max, y_steps);

    auto dexel_map = generateDexelMap(
        V, F,
        x_min, x_max, x_steps,
        y_min, y_max, y_steps
    );

    return 0;
}
