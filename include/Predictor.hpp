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
        void convet_atoms(const Atoms& atoms);
        void predict();
    
    private:
        // インプット元
        thrust::device_vector<int> x;
        thrust::device_vector<int> edge_index;
        thrust::device_vector<float> edge_weight;
};

#endif