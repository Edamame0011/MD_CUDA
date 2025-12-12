#ifndef PREDICTOR_TRT_HPP
#define PREDICTOR_TRT_HPP

#include <string>
#include <thrust/device_vector.h>
#include <NvInfer.h>

#include "Atoms.hpp"
#include "NeighbourList.hpp"
#include "Predictor.hpp"

class Predictor_TRT : public Predictor {
    public:
        Predictor_TRT(const std::string& model_path);

        void load_model(const std::string& model_path);
        void convet_atoms(Atoms& atoms, float cutoff);
        void predict(Atoms& atoms) override;

        void printInfo(nvinfer1::ICudaEngine* engine);
    private:
        // 変換時のバッファ
        thrust::device_vector<int> valid_linear_indices;

        // インプット元
        thrust::device_vector<int> x;
        thrust::device_vector<int> source_index;
        thrust::device_vector<int> target_index;
        thrust::device_vector<float> edge_weight;

        // TensorRTオブジェクト
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
};

#endif