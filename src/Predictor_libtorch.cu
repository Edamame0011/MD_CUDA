#include "Predictor_libtorch.hpp"
#include <thrust/transform_reduce.h>
#include <vector_types.h>
#include <thrust/copy.h>

namespace {
    struct Filter {
        const float* d_x;
        const float* d_y;
        const float* d_z;
        int num_atoms;
        float Lbox;
        float Linv;
        float cutoff_sq;
        Filter(const float* _d_x, const float* _d_y, const float* _d_z, int _num_atoms, float _Lbox, float _Linv, float _cutoff_sq) : d_x(_d_x), d_y(_d_y), d_z(_d_z), num_atoms(_num_atoms), Lbox(_Lbox), Linv(_Linv), cutoff_sq(_cutoff_sq) {}
        __host__ __device__ bool operator() (int idx) const {
            int i = idx / num_atoms;
            int j = idx % num_atoms;

            if (j <= i) return false;

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

            dx -= Lbox * floorf(dx * Linv + 0.5);
            dy -= Lbox * floorf(dy * Linv + 0.5);
            dz -= Lbox * floorf(dz * Linv + 0.5);

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return dist_sq <= cutoff_sq;
        }
    };

    struct Result {
        const float* d_x;
        const float* d_y;
        const float* d_z;
        int num_atoms;
        float Lbox;
        float Linv;
        Result(const float* _d_x, const float* _d_y, const float* _d_z, int _num_atoms, float _Lbox, float _Linv) : d_x(_d_x), d_y(_d_y), d_z(_d_z), num_atoms(_num_atoms), Lbox(_Lbox), Linv(_Linv) {}
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

            dx -= Lbox * floorf(dx * Linv + 0.5);
            dy -= Lbox * floorf(dy * Linv + 0.5);
            dz -= Lbox * floorf(dz * Linv + 0.5);

            return thrust::make_tuple(j, i, dx, dy, dz);
        }
    };

    struct Copy {
        int64_t *source_former, *source_latter, *target_former, *target_latter;
        float *w_x_former, *w_x_latter, *w_y_former, *w_y_latter, *w_z_former, *w_z_latter;
        Copy(
            int64_t* _source_former, 
            int64_t* _target_latter, 
            int64_t* _target_former, 
            int64_t* _source_latter, 
            float* _w_x_former, 
            float* _w_x_latter, 
            float* _w_y_former, 
            float* _w_y_latter, 
            float* _w_z_former, 
            float* _w_z_latter
        ) : 
        source_former(_source_former), 
        target_latter(_target_latter),  
        target_former(_target_former), 
        source_latter(_source_latter), 
        w_x_former(_w_x_former), 
        w_x_latter(_w_x_latter), 
        w_y_former(_w_y_former), 
        w_y_latter(_w_y_latter), 
        w_z_former(_w_z_former), 
        w_z_latter(_w_z_latter) {}
        __host__ __device__ void operator() (int idx) {
            // source indexのコピー
            target_latter[idx] = source_former[idx];
            // target indexのコピー
            source_latter[idx] = target_former[idx];
            // dx, dy, dzを反転させてコピー
            w_x_latter[idx] = - w_x_former[idx];
            w_y_latter[idx] = - w_y_former[idx];
            w_z_latter[idx] = - w_z_former[idx];
        }
    };
}

Predictor_libtorch::Predictor_libtorch(Atoms& atoms, const std::string& model_path) {
    int N = atoms.get_num_atoms();
    d_valid_indices.resize(N * N, 0);
    d_edge_weight.resize(3 * N * N, 0);
    d_edge_index.resize(N * N, 0);

    load_model(model_path);
}

void Predictor_libtorch::load_model(const std::string& model_path) {
    try{
        model = torch::jit::load(model_path);
        model.to(torch::kCUDA);
        std::cout << "モデルをロードしました：" << model_path << std::endl; 
    }
    catch(c10::Error& e){
        std::cerr << "モデルの読み込みに失敗しました。" << std::endl
                  << e.what() << std::endl;
        throw;
    }
}

void Predictor_libtorch::convert_atoms(Atoms& atoms, NeighbourList& NL) {
    int num_atoms = atoms.get_num_atoms();
    float Lbox = atoms.get_Lbox();
    float cutoff = NL.get_cutoff();
    float cutoff_sq = cutoff * cutoff;

    float* d_x = atoms.x_ptr();
    float* d_y = atoms.y_ptr();
    float* d_z = atoms.z_ptr();

    // カットオフ距離以内にある原子のインデックスを取得
    auto end_ptr = thrust::copy_if(
        NL.get_valid_indices().begin(), 
        NL.get_valid_indices().end(), 
        d_valid_indices.begin(), 
        Filter(d_x, d_y, d_z, num_atoms, Lbox, 1 / Lbox, cutoff_sq)
    );

    // データ形式の変換・距離の計算
    int num_pairs = end_ptr - d_valid_indices.begin();
    // 双方向グラフなため2倍
    num_edges = 2 * num_pairs;

    d_edge_index.resize(2 * num_edges);
    d_edge_weight.resize(3 * num_edges);

    // 前半部分の書き込み
    thrust::transform(
        d_valid_indices.begin(), 
        end_ptr, 
        thrust::make_zip_iterator(
            thrust::make_tuple(
                d_edge_index.begin(), 
                d_edge_index.begin() + num_edges, 
                d_edge_weight.begin(), 
                d_edge_weight.begin() + num_edges, 
                d_edge_weight.begin() + num_edges + num_edges
            )
        ), 
        Result(d_x, d_y, d_z, num_atoms, Lbox, 1 / Lbox)
    );

    // 後半部分にコピー
    auto idx_ptr = thrust::raw_pointer_cast(d_edge_index.data());
    auto w_ptr = thrust::raw_pointer_cast(d_edge_weight.data());

    thrust::for_each(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(num_pairs), 
        Copy(
            idx_ptr, 
            idx_ptr + num_edges + num_pairs, 
            idx_ptr + num_edges, 
            idx_ptr + num_pairs, 
            w_ptr, 
            w_ptr + num_pairs, 
            w_ptr + num_edges, 
            w_ptr + num_edges + num_pairs, 
            w_ptr + 2 * num_edges, 
            w_ptr + 2 * num_edges + num_pairs
        )
    );
/*
    // target_indexの前半をsource_indexの後半にコピー
    thrust::copy(
        idx_ptr + num_edges, 
        idx_ptr + num_edges + num_pairs, 
        idx_ptr + num_pairs
    );

    // source_indexの前半をtarget_indexの後半にコピー
    thrust::copy(
        idx_ptr, 
        idx_ptr + num_pairs, 
        idx_ptr + num_edges + num_pairs
    );

    // 距離ベクトルを反転させてコピー
    // dxを反転させて書き込み
    thrust::transform(
        w_ptr, 
        w_ptr + num_pairs, 
        w_ptr + num_pairs, 
        thrust::negate<float>()
    );

    // dyを反転させて書き込み
    thrust::transform(
        w_ptr + num_edges, 
        w_ptr + num_edges + num_pairs, 
        w_ptr + num_edges + num_pairs, 
        thrust::negate<float>()
    );

    // dzを反転させて書き込み
    thrust::transform(
        w_ptr + 2 * num_edges, 
        w_ptr + 2 * num_edges + num_pairs, 
        w_ptr + 2 * num_edges + num_pairs, 
        thrust::negate<float>()
    );
*/
}

void Predictor_libtorch::predict(Atoms& atoms, NeighbourList& NL) {
    convert_atoms(atoms, NL);

    int num_atoms = atoms.get_num_atoms();

    // torch::Tensorオブジェクトの作成
    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor x = torch::from_blob(
        atoms.atomic_numbers_ptr(), 
        {num_atoms}, 
        options.dtype(torch::kInt64)
    );

    torch::Tensor edge_index = torch::from_blob(
        thrust::raw_pointer_cast(d_edge_index.data()), 
        {2, num_edges}, 
        options.dtype(torch::kInt64)
    );

    torch::Tensor edge_weight = torch::from_blob(
        thrust::raw_pointer_cast(d_edge_weight.data()), 
        {3, num_edges}, 
        options.dtype(torch::kFloat32)
    );

    edge_weight = edge_weight.t();
    edge_weight.requires_grad_(true);

    // 推論
    model.eval();

    try {
        auto result_iv = model.forward({x, edge_index, edge_weight});
        auto result_tuple = result_iv.toTuple();
        auto elements = result_tuple->elements();
        
        torch::Tensor energy = elements[0].toTensor().to(torch::kFloat32).detach();
        torch::Tensor forces = elements[1].toTensor().to(torch::kFloat32).detach();

        forces = forces.t().contiguous();

        // libtorch側のポインター
        float* energy_ptr = energy.data_ptr<float>();
        float* forces_ptr = forces.data_ptr<float>();

        // thrust側のポインター
        float* thrust_force_ptr_x = atoms.force_x_ptr();
        float* thrust_force_ptr_y = atoms.force_y_ptr();
        float* thrust_force_ptr_z = atoms.force_z_ptr();
        float potential_energy;

        // 値のコピー
        cudaMemcpy(&potential_energy, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(thrust_force_ptr_x, forces_ptr, num_atoms * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(thrust_force_ptr_y, forces_ptr + num_atoms, num_atoms * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(thrust_force_ptr_z, forces_ptr + 2 * num_atoms, num_atoms * sizeof(float), cudaMemcpyDeviceToDevice);

        atoms.set_potential_energy(potential_energy);
    }
    catch(const c10::Error& e) {
        std::cerr << "モデルの推論に失敗しました。" << std::endl 
                  << e.what() << std::endl;
        throw;
    }
}