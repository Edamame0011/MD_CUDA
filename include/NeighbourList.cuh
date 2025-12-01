#include "Atoms.cuh"
#include <thrust/device_vector.h>

class NeighbourList {
    public:
        NeighbourList();

        // ポインタの取得（読み取り専用）
        const std::size_t* target_indices_ptr() const { return thrust::raw_pointer_cast(d_target_indices.data()); }
        const std::size_t* source_indices_ptr() const { return thrust::raw_pointer_cast(d_source_indices.data()); }
        const std::size_t* config_ptr() const { return thrust::raw_pointer_cast(d_config.data()); }

        // 隣接リストの生成
        void generate(const Atoms& atoms);

        // 隣接リストのチェックをし、必要があれば再生性
        void update(const Atoms& atoms);

    private:
        thrust::device_vector<std::size_t> d_target_indices;
        thrust::device_vector<std::size_t> d_source_indices;
        thrust::device_vector<float> d_config;
}