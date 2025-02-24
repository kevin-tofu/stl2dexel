#ifndef DEXEL_HPP
#define DEXEL_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>


namespace dexel_sim {
    template <typename T = double>
    class Dexel {
    public:
        std::vector<T> z_values;

        void add(T z);
        void sort();
        std::vector<T> getZValues() const;
        void print() const;
    };

    template <typename T = double>
    class DexelMap {
        public:
            DexelMap(
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXi& F, 
                int x_steps,
                int y_steps
            );
            ~DexelMap();
        private:
            std::vector<std::vector<Dexel<T>>> dexel_map;
        };
        

    template <typename T>
    class TriDexelMap {
        public:
            TriDexelMap(
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXi& F, 
                int x_steps,
                int y_steps,
                int z_steps
            );
            ~TriDexelMap();
        private:
            std::vector<DexelMap<T>> tridexel_array;
            // std::array<DexelMap<T>, 3> tridexel_array;
    };
};

#include "dexel.tpp"

#endif // DEXEL_HPP
