#ifndef PREDICTOR_LIBTORCH_HPP
#define PREDICTOR_LIBTORCH_HPP

#include "Predictor.hpp"

#include <torch/script.h>
#include <torch/torch.h>
#include <thrust/device_vector.h>

#include <string>
#include "Atoms.hpp"
#include "NeighbourList.hpp"

class Predictor_libtorch : public Predictor {
    public:
        Predictor_libtorch(Atoms& atoms, const std::string& model_path);
        void load_model(const std::string& model_path);
        void convert_atoms(Atoms& atoms, NeighbourList& NL);
        void predict(Atoms& atoms, NeighbourList& NL) override;
    private:
        torch::jit::script::Module model;

        // 入力テンソル
        thrust::device_vector<int> d_edge_index;
        thrust::device_vector<float> d_edge_weight;

        // カットオフ距離以内にある原子のインデックスを保存するバッファ
        thrust::device_vector<int> d_valid_indices;

        // ホスト側の変数
        int num_edges;
};

#endif