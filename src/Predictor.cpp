#include "Predictor.hpp"
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <vector_types.h>
#include <thrust/copy.h>
#include <NvInfer.h>
#include <iostream>
#include <fstream>

namespace {
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    } logger;

    struct Filter {
        const float* d_x;
        const float* d_y;
        const float* d_z;
        int num_atoms;
        float margin_sq;
        Filter(const float* _d_x, const float* _d_y, const float* _d_z, int _num_atoms,float _margin_sq) : d_x(_d_x), d_y(_d_y), d_z(_d_z), num_atoms(_num_atoms), margin_sq(_margin_sq) {}
        __host__ __device__ bool operator() (int idx) const {
            int i = idx / num_atoms;
            int j = idx % num_atoms;

            if (i == j) return false;

            float x1 = d_x[i];
            float y1 = d_y[i];
            float z1 = d_z[i];

            float x2 = d_x[j];
            float y2 = d_y[j];
            float z2 = d_z[j];

            // 距離の計算
            float dx = x1 - x2;
            float dy = y1 - y2;
            float dz = z1 - z2;

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return dist_sq <= margin_sq;
        }
    };

    struct Result {
        const float* d_x;
        const float* d_y;
        const float* d_z;
        int num_atoms;
        Result(const float* _d_x, const float* _d_y, const float* _d_z, int _num_atoms) : d_x(_d_x), d_y(_d_y), d_z(_d_z), num_atoms(_num_atoms) {}
        __host__ __device__ auto operator() (int idx) {
            int i = idx / num_atoms;
            int j = idx % num_atoms;
            
            // 距離の計算
            float x1 = d_x[i];
            float y1 = d_y[i];
            float z1 = d_z[i];

            float x2 = d_x[j];
            float y2 = d_y[j];
            float z2 = d_z[j];

            // 距離の計算
            float dx = x1 - x2;
            float dy = y1 - y2;
            float dz = z1 - z2;

            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            return thrust::make_tuple(i, j, dist);
        }
    };

    std::string getDataTypeString(nvinfer1::DataType type) {
        switch (type) {
            case nvinfer1::DataType::kFLOAT: return "FP32";
            case nvinfer1::DataType::kHALF: return "FP16";
            case nvinfer1::DataType::kINT8: return "INT8";
            case nvinfer1::DataType::kINT32: return "INT32";
            case nvinfer1::DataType::kBOOL: return "BOOL";
            default: return "Unknown";
        }
    }
}

Predictor::Predictor(const std::string& model_path) {
    load_model(model_path);
}

void Predictor::load_model(const std::string& model_path) {
    // ファイルをバイナリで読み込む
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Error reading engine file" << std::endl;
        throw;
    }
    std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // TensorRTオブジェクトの初期化
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();
}

void Predictor::printInfo(nvinfer1::ICudaEngine* engine) {
    int nbIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; i ++) {
        const char* name = engine->getIOTensorName(i);

        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType type = engine->getTensorDataType(name);

        bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT);

        std::cout << "Index " << i << ": "
                  << (isInput? "[input]" : "[output]")
                  << "Name: " << name
                  << ", Type: " << getDataTypeString(type)
                  << ", Shape: (";
        for (int d = 0; d < dims.nbDims; d ++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << ",";
        }
        std::cout << ")" << std::endl;
    }
}

void Predictor::convet_atoms(Atoms& atoms, float cutoff) {
    int num_atoms = atoms.get_num_atoms();
    this->x = atoms.get_atomic_numbers();
    float cutoff_sq = cutoff * cutoff;

    float* d_x = atoms.x_ptr();
    float* d_y = atoms.y_ptr();
    float* d_z = atoms.z_ptr();

    // フィルターでマスク配列を作成
    auto end_ptr = thrust::copy_if(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(num_atoms * num_atoms), 
        valid_linear_indices.begin(), 
        Filter(d_x, d_y, d_z, num_atoms, cutoff_sq)
    );

    // source_idx, target_idxに変換
    int num_edges = end_ptr - valid_linear_indices.begin();

    source_index.resize(num_edges);
    target_index.resize(num_edges);

    thrust::transform(
        valid_linear_indices.begin(), 
        valid_linear_indices.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(source_index.begin(), target_index.begin(), edge_weight.begin())), 
        Result(d_x, d_y, d_z, num_atoms)
    );
}