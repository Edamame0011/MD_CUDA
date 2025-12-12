#ifndef NEIGHBOURLIST_HPP
#define NEIGHBOURLIST_HPP

#include "Atoms.hpp"
#include <thrust/device_vector.h>

class NeighbourList {
    public:
        NeighbourList(float _cutoff, float _margin) : cutoff(_cutoff), margin(_margin) {}

        // ポインタの取得（読み取り専用）
        const int* target_indices_ptr() const { return thrust::raw_pointer_cast(d_valid_indices.data()); }
        const float* config_x_ptr() const { return thrust::raw_pointer_cast(d_config_x.data()); }
        const float* config_y_ptr() const { return thrust::raw_pointer_cast(d_config_y.data()); }
        const float* config_z_ptr() const { return thrust::raw_pointer_cast(d_config_z.data()); }

        // ゲッター
        float get_cutoff() { return cutoff; }
        float get_margin() { return margin; }
        thrust::device_vector<int>& get_valid_indices() { return d_valid_indices; }

        // 隣接リストの生成
        void generate(Atoms& atoms);

        // 隣接リストのチェックをし、必要があれば再生性
        void check(Atoms& atoms);

    private:
        // GPU上の配列
        thrust::device_vector<int> d_valid_indices;
        thrust::device_vector<float> d_config_x;
        thrust::device_vector<float> d_config_y;
        thrust::device_vector<float> d_config_z;

        // ホスト側の変数
        float margin;
        float cutoff;
};

#endif