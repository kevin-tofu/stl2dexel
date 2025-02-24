#ifndef DEXEL_TPP
#define DEXEL_TPP


#include "dexel.hpp"
// #include "str2dexel.hpp"


namespace dexel_sim {
    void getBoundingBox(
        const Eigen::MatrixXd& V,
        double& x_min, double& x_max,
        double& y_min, double& y_max,
        double& z_min, double& z_max
    ) {
        x_min = y_min = z_min = std::numeric_limits<double>::max();
        x_max = y_max = z_max = std::numeric_limits<double>::lowest();

        for (int i = 0; i < V.rows(); ++i) {
            x_min = std::min(x_min, V(i, 0));
            x_max = std::max(x_max, V(i, 0));
            y_min = std::min(y_min, V(i, 1));
            y_max = std::max(y_max, V(i, 1));
            z_min = std::min(z_min, V(i, 1));
            z_max = std::max(z_max, V(i, 1));
        }
    }

    template <typename T>
    void findIntersections_OMP_XY(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        double x, double y,
        Dexel<T>& dexel
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


    enum class Plane { XY, XZ, YZ };
    template <typename T>
    void findIntersections_OMP(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        double coord1, double coord2,
        Plane plane,
        Dexel<T>& dexel
    ) {
        #pragma omp parallel for
        for (int i = 0; i < F.rows(); ++i) {
            Eigen::Vector3d v0 = V.row(F(i, 0));
            Eigen::Vector3d v1 = V.row(F(i, 1));
            Eigen::Vector3d v2 = V.row(F(i, 2));

            // 平面ごとに使用する座標を選択
            Eigen::Vector2d p0, p1, p2;
            double val0, val1, val2;

            switch (plane) {
                case Plane::XY:
                    p0 = {v0.x(), v0.y()};
                    p1 = {v1.x(), v1.y()};
                    p2 = {v2.x(), v2.y()};
                    val0 = v0.z();
                    val1 = v1.z();
                    val2 = v2.z();
                    break;
                case Plane::XZ:
                    p0 = {v0.x(), v0.z()};
                    p1 = {v1.x(), v1.z()};
                    p2 = {v2.x(), v2.z()};
                    val0 = v0.y();
                    val1 = v1.y();
                    val2 = v2.y();
                    break;
                case Plane::YZ:
                    p0 = {v0.y(), v0.z()};
                    p1 = {v1.y(), v1.z()};
                    p2 = {v2.y(), v2.z()};
                    val0 = v0.x();
                    val1 = v1.x();
                    val2 = v2.x();
                    break;
            }

            double x_min = std::min({p0.x(), p1.x(), p2.x()});
            double x_max = std::max({p0.x(), p1.x(), p2.x()});
            double y_min = std::min({p0.y(), p1.y(), p2.y()});
            double y_max = std::max({p0.y(), p1.y(), p2.y()});

            if (coord1 < x_min || coord1 > x_max || coord2 < y_min || coord2 > y_max) {
                continue;
            }

            Eigen::Matrix2d A;
            A << (p1 - p0), (p2 - p0);
            Eigen::Vector2d b(coord1 - p0.x(), coord2 - p0.y());

            if (std::abs(A.determinant()) < 1e-10) continue;

            Eigen::Vector2d lambdas = A.colPivHouseholderQr().solve(b);
            double u = lambdas[0], v = lambdas[1], w = 1.0 - u - v;

            if (u >= 0 && v >= 0 && w >= 0) {
                double intersectionValue = w * val0 + u * val1 + v * val2;
                #pragma omp critical
                dexel.add(intersectionValue);
            }
        }

        dexel.sort();
    }

    template <typename T>
    std::vector<std::vector<Dexel<T>>> generateDexelMap_XY(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F, 
        double x_min, double x_max, int x_steps,
        double y_min, double y_max, int y_steps
    ){
        double dx = (x_max - x_min) / (x_steps - 1);
        double dy = (y_max - y_min) / (y_steps - 1);

        std::vector<std::vector<Dexel<T>>> dexel_map(
            x_steps, std::vector<Dexel<T>>(y_steps)
        );

        // Generate Dexel by scanning
        for (int i = 0; i < x_steps; ++i) {
            double x = x_min + i * dx;
            for (int j = 0; j < y_steps; ++j) {
                double y = y_min + j * dy;
                // findIntersections(V, F, x, y, dexel_map[i][j]);
                // findIntersections_AABB(V, F, x, y, dexel_map[i][j]);
                findIntersections_OMP_XY(V, F, x, y, dexel_map[i][j]);
            }
        }

        return dexel_map;
    }

    template <typename T>
    std::vector<std::vector<Dexel<T>>> generateDexelMap(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F, 
        double x_min, double x_max, int x_steps,
        double y_min, double y_max, int y_steps,
        Plane plane
    ){
        double dx = (x_max - x_min) / (x_steps - 1);
        double dy = (y_max - y_min) / (y_steps - 1);

        std::vector<std::vector<Dexel<T>>> dexel_map(
            x_steps, std::vector<Dexel<T>>(y_steps)
        );

        // Generate Dexel by scanning
        for (int i = 0; i < x_steps; ++i) {
            double x = x_min + i * dx;
            for (int j = 0; j < y_steps; ++j) {
                double y = y_min + j * dy;
                // findIntersections(V, F, x, y, dexel_map[i][j]);
                // findIntersections_AABB(V, F, x, y, dexel_map[i][j]);
                findIntersections_OMP(
                    V, F, x, y,
                    plane,
                    dexel_map[i][j]
                );
            }
        }

        return dexel_map;
    }


    template <typename T>
    void Dexel<T>::add(T z) {
        z_values.push_back(z);
    }

    template <typename T>
    void Dexel<T>::sort() {
        std::sort(z_values.begin(), z_values.end());
    }

    template <typename T>
    std::vector<T> Dexel<T>::getZValues() const {
        return z_values;
    }

    template <typename T>
    void Dexel<T>::print() const {
        for (const T& z : z_values) {
            std::cout << z << " ";
        }
        std::cout << std::endl;
    }

    template <typename T>
    DexelMap<T>::DexelMap(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F, 
        int x_steps,
        int y_steps
    ){
        double x_min, x_max, y_min, y_max, z_min, z_max;
        getBoundingBox(
            V, x_min, x_max, y_min, y_max, z_min, z_max
        );

        this->dexel_map = generateDexelMap_XY<T>(
            V, F,
            x_min, x_max, x_steps,
            y_min, y_max, y_steps
        );
    }

    template <typename T>
    DexelMap<T>::~DexelMap(){
    }

    template <typename T>
    TriDexelMap<T>::TriDexelMap(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F, 
        int x_steps,
        int y_steps,
        int z_steps
    ){
        double x_min, x_max, y_min, y_max, z_min, z_max;
        getBoundingBox(
            V, x_min, x_max, y_min, y_max, z_min, z_max
        );
        // this->tridexel_array = {
        //     generateDexelMap<T>(V, F, x_min, x_max, x_steps, y_min, y_max, y_steps, Plane::XY),
        //     generateDexelMap<T>(V, F, x_min, x_max, x_steps, z_min, z_max, z_steps, Plane::XZ),
        //     generateDexelMap<T>(V, F, y_min, y_max, y_steps, z_min, z_max, z_steps, Plane::YZ)
        // };
        // auto xy = generateDexelMap<T>(V, F, x_min, x_max, x_steps, y_min, y_max, y_steps, Plane::XY);
        // auto xz = generateDexelMap<T>(V, F, x_min, x_max, x_steps, z_min, z_max, z_steps, Plane::XZ);
        // auto yz = generateDexelMap<T>(V, F, y_min, y_max, y_steps, z_min, z_max, z_steps, Plane::YZ);
        auto xy = DexelMap<T>(V, F, x_steps, y_steps, Plane::XY);
        auto xz = DexelMap<T>(V, F, x_steps, z_steps, Plane::XZ);
        auto yz = DexelMap<T>(V, F, y_steps, z_steps, Plane::YZ);
        this->tridexel_array.push_back(xy);
        this->tridexel_array.push_back(xz);
        this->tridexel_array.push_back(yz);
    }
    template <typename T>
    TriDexelMap<T>::~TriDexelMap(){}
}

#endif