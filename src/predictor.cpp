#include "Predictor.hpp"

namespace {

}

Predictor::Predictor(const std::string& model_path) {
    load_model(model_path);
}

void Predictor::load_model(const std::string& model_path) {
    // load model
}

void Predictor::convet_atoms(const Atoms& atoms) {
    const thrust::device_vector<float>& d_x = atoms.get_x();
    const thrust::device_vector<float>& d_y = atoms.get_y();
    const thrust::device_vector<float>& d_z = atoms.get_z();

    
}