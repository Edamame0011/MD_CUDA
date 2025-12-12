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
        float cutoff_sq;
        Filter(const float* _d_x, const float* _d_y, const float* _d_z, int _num_atoms,float _cutoff_sq) : d_x(_d_x), d_y(_d_y), d_z(_d_z), num_atoms(_num_atoms), cutoff_sq(_cutoff_sq) {}
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

            return dist_sq <= cutoff_sq;
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
}

Predictor_libtorch::Predictor_libtorch(Atoms& atoms) {
    int N = atoms.get_num_atoms();
    d_valid_indices.reserve(N * N);
    d_edge_index.reserve(N * N / 2);
    d_edge_index.reserve(N * N / 2);
}

void Predictor_libtorch::load_model(const std::string& model_path) {
    try{
        model = torch::jit::load(model_path);
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
        Filter(d_x, d_y, d_z, num_atoms, cutoff_sq)
    );

    // データ形式の変換・距離の計算
    num_edges = end_ptr - d_valid_indices.begin();

    d_edge_index.resize(2 * num_edges);
    d_edge_weight.resize(num_edges);

    thrust::transform(
        d_valid_indices.begin(), 
        d_valid_indices.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(d_edge_index.begin(), d_edge_index.begin() + num_edges, d_edge_weight.begin())), 
        Result(d_x, d_y, d_z, num_atoms)
    );
}

void Predictor_libtorch::predict(Atoms& atoms, NeighbourList& NL) {
    convert_atoms(atoms, NL);

    int num_atoms = atoms.get_num_atoms();

    // torch::Tensorオブジェクトの作成
    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor x = torch::from_blob(
        atoms.atomic_numbers_ptr(), 
        {num_atoms}, 
        options.dtype(torch::kInt32)
    );

    torch::Tensor edge_index = torch::from_blob(
        thrust::raw_pointer_cast(d_edge_index.data()), 
        {2, num_edges}, 
        options.dtype(torch::kInt32)
    );

    torch::Tensor edge_weight = torch::from_blob(
        thrust::raw_pointer_cast(d_edge_weight.data()), 
        {num_edges}, 
        options.dtype(torch::kFloat32)
    );

    // 推論
    model.eval();
    torch::NoGradGuard no_grad;

    try {
        auto result_iv = model.forward({x, edge_index, edge_weight});
        auto result_tuple = result_iv.toTuple();
        auto elements = result_tuple->elements();
        
        torch::Tensor energy = elements[0].toTensor().to(torch::kFloat32).detach();
        torch::Tensor forces = elements[1].toTensor().to(torch::kFloat32).detach();

        energy = energy.contiguous();
        forces = forces.contiguous();

        // libtorch側のポインター
        float* energy_ptr = energy.data_ptr<float>();
        float* force_ptr_x = forces[0].data_ptr<float>();
        float* force_ptr_y = forces[1].data_ptr<float>();
        float* force_ptr_z = forces[2].data_ptr<float>();

        // thrust側のポインター
        float* thrust_force_ptr_x = atoms.force_x_ptr();
        float* thrust_force_ptr_y = atoms.force_y_ptr();
        float* thrust_force_ptr_z = atoms.force_z_ptr();
        float potential_energy;

        // 値のコピー
        cudaMemcpy(&potential_energy, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(thrust_force_ptr_x, force_ptr_x, num_atoms * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(thrust_force_ptr_y, force_ptr_y, num_atoms * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(thrust_force_ptr_z, force_ptr_z, num_atoms * sizeof(float), cudaMemcpyDeviceToDevice);

        atoms.set_potential_energy(potential_energy);
    }
    catch(const c10::Error& e) {
        std::cerr << "モデルの推論に失敗しました。" << std::endl 
                  << e.what() << std::endl;
        throw;
    }
}