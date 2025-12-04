#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <string>
#include <thrust/device_vector.h>

#include "Atoms.hpp"
#include "NeighbourList.hpp"

class Predictor {
    public:
        Predictor(const std::string& model_path);

        void load_model(const std::string& model_path);
        void convet_atoms(Atoms& atoms, float cutoff);
        void predict();
    
    private:
        // 変換時のバッファ
        thrust::device_vector<int> valid_linear_indices;

        // インプット元
        thrust::device_vector<int> x;
        thrust::device_vector<int> source_index;
        thrust::device_vector<int> target_index;
        thrust::device_vector<float> edge_weight;
};

#endif