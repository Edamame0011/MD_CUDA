#include "Predictor.hpp"
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <vector_types.h>
#include <thrust/copy.h>

namespace {
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
            
        }
    };
}

Predictor::Predictor(const std::string& model_path) {
    load_model(model_path);
}

void Predictor::load_model(const std::string& model_path) {
    // load model
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