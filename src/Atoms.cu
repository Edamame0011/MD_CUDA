#include "Atoms.hpp"
#include "constant.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

namespace {
    struct PosUpdate {
        float dt;
        PosUpdate(float _dt) : dt(_dt) {}
        template <typename Tuple>
        __host__ __device__ auto operator() (const Tuple& pos, const Tuple& vel) const {
            return thrust::make_tuple(
                thrust::get<0>(pos) + dt * thrust::get<0>(vel), 
                thrust::get<1>(pos) + dt * thrust::get<1>(vel), 
                thrust::get<2>(pos) + dt * thrust::get<2>(vel)
            );
        }
    };

    struct VelUpdate {
        float dt;
        float conversion_factor;
        VelUpdate(float _dt, float _conversion_factor) : dt(_dt), conversion_factor(_conversion_factor) {}
        template <typename Tuple>
        __host__ __device__ void operator() (Tuple t) {
            thrust::get<0>(t) += 0.5 * dt * (thrust::get<3>(t) / thrust::get<6>(t)) * conversion_factor;
            thrust::get<1>(t) += 0.5 * dt * (thrust::get<4>(t) / thrust::get<6>(t)) * conversion_factor;
            thrust::get<2>(t) += 0.5 * dt * (thrust::get<5>(t) / thrust::get<6>(t)) * conversion_factor;
        }
    };

    struct Multiply {
        template <typename Tuple>
        __host__ __device__ float operator() (const Tuple& t) const {
            return thrust::get<0>(t) * thrust::get<1>(t);
        }
    };

    struct RemoveDrift {
        float avg_x, avg_y, avg_z;
        RemoveDrift(float _avg_x, float _avg_y, float _avg_z) : avg_x(_avg_x), avg_y(_avg_y), avg_z(_avg_z) {}
        template <typename Tuple>
        __host__ __device__ auto operator() (const Tuple& vel) const {
            return thrust::make_tuple(
                thrust::get<0>(vel) - avg_x, 
                thrust::get<1>(vel) - avg_y, 
                thrust::get<2>(vel) - avg_z
            );
        }
    };

    struct ApplyPBC {
        float Lbox;
        ApplyPBC(float _Lbox) : Lbox(_Lbox) {}
        template <typename Tuple>
        __host__ __device__ void operator() (Tuple t) {
            float shift_x = floorf(thrust::get<0>(t) / Lbox + 0.5);
            float shift_y = floorf(thrust::get<1>(t) / Lbox + 0.5);
            float shift_z = floorf(thrust::get<2>(t) / Lbox + 0.5);

            thrust::get<0>(t) -= Lbox * shift_x;
            thrust::get<1>(t) -= Lbox * shift_y;
            thrust::get<2>(t) -= Lbox * shift_z;

            thrust::get<3>(t) += (int)shift_x;
            thrust::get<4>(t) += (int)shift_y;
            thrust::get<5>(t) += (int)shift_z;
        }
    };

    struct CalcKinEnergy {
        template <typename Tuple>
        __host__ __device__ float operator() (const Tuple& t) {
            float vel_x = thrust::get<0>(t);
            float vel_y = thrust::get<1>(t);
            float vel_z = thrust::get<2>(t);
            
            return 0.5 * thrust::get<3>(t) * (vel_x * vel_x + vel_y * vel_y + vel_z * vel_z);
        }
    };

    //文字列からLatticeを見つける
    std::string find_lattice(std::string input) {
        //開始位置のキーワード
        std::string start_tag = "Lattice=\"";

        //開始位置
        std::size_t start_position = input.find(start_tag);

        if(start_position != std::string::npos){
            start_position = start_position + start_tag.length();

            //開始位置から次の"を探す
            std::size_t end_position = input.find('"', start_position);

            if(end_position != std::string::npos){
                //抜き出す部分の長さ
                std::size_t length = end_position - start_position;

                //文字列の抜き出し
                std::string result = input.substr(start_position, length);

                return result;
            }

            else{ 
                throw std::runtime_error("終了のダブルクオーテーションが見つかりません。"); 
            }
        }

        else{ 
            throw std::runtime_error("ファイルにLatticeデータが含まれていません。"); 
        }
    }
    //原子種類と原子番号を関連づけるmap
    std::map<std::string, int> atom_number_map = {
            {"H",  1},
            {"He", 2},
            {"Li", 3},
            {"Be", 4},
            {"B",  5},
            {"C",  6},
            {"N",  7},
            {"O",  8},
            {"F",  9},
            {"Ne", 10},
            {"Na", 11},
            {"Mg", 12},
            {"Al", 13},
            {"Si", 14},
            {"P",  15},
            {"S",  16},
            {"Cl", 17},
            {"Ar", 18},
            {"K",  19},
            {"Ca", 20}
        };

    //原子種類と原子質量を関連づけるmap
    std::map<std::string, double> atom_mass_map = {
            {"H",   1.0080},
            {"He",  4.0026},
            {"Li",  6.94},
            {"Be",  9.0122},
            {"B",   10.81},
            {"C",   12.011},
            {"N",   14.007},
            {"O",   15.999},
            {"F",   18.998},
            {"Ne",  20.180},
            {"Na",  22.990},
            {"Mg",  24.305},
            {"Al",  26.982},
            {"Si",  28.0855},
            {"P",   30.974},
            {"S",   32.06},
            {"Cl",  35.45},
            {"Ar",  39.95},
            {"K",   39.098},
            {"Ca",  40.078}
        };
}

Atoms::Atoms(std::string data_path) {
    std::ifstream file(data_path);

    if(!file.is_open()) {
        std::cerr << "構造ファイルを開けません。" << std::endl;
        throw;
    }

    std::string line;
    int num_atoms;
    std::getline(file, line);
    num_atoms = std::stoi(line);
    d_box_x.resize(num_atoms, 0);
    d_box_y.resize(num_atoms, 0);
    d_box_z.resize(num_atoms, 0);
    d_vel_x.resize(num_atoms, 0);
    d_vel_y.resize(num_atoms, 0);
    d_vel_z.resize(num_atoms, 0);

    std::getline(file, line);
    std::array<float, 3> lattice_x, lattice_y, lattice_z;
    // latticeの部分を読み込む
    // コメントからlatticeの部分を抜き出す
    std::string lattice = find_lattice(line);
    // 文字列をストリームに変換
    std::istringstream iss(lattice);
    iss >> lattice_x[0] >> lattice_x[1] >> lattice_x[2] >> 
           lattice_y[0] >> lattice_y[1] >> lattice_y[2] >>
           lattice_z[0] >> lattice_z[1] >> lattice_z[2];
    // とりあえず正方格子を想定
    float box_size = lattice_x[0];         

    // 原子の情報を保持する変数
    thrust::host_vector<int> h_atomic_numbers(num_atoms);
    thrust::host_vector<float> h_x(num_atoms);
    thrust::host_vector<float> h_y(num_atoms);
    thrust::host_vector<float> h_z(num_atoms);
    thrust::host_vector<float> h_force_x(num_atoms);
    thrust::host_vector<float> h_force_y(num_atoms);
    thrust::host_vector<float> h_force_z(num_atoms);
    thrust::host_vector<float> h_masses(num_atoms);

    int i = 0;

    while(std::getline(file, line)) {
        std::string atom_type;

        std::istringstream iss(line);

        iss >> atom_type >> h_x[i] >> h_y[i] >> h_z[i] >> h_force_x[i] >> h_force_y[i] >> h_force_z[i];
        
        h_atomic_numbers[i] = atom_number_map[atom_type];
        h_masses[i] = atom_mass_map[atom_type];

        i ++;
    }

    // デバイスに転送
    d_x = h_x;
    d_y = h_y;
    d_z = h_z;
    d_force_x = h_force_x;
    d_force_y = h_force_y;
    d_force_z = h_force_z;
    d_masses = h_masses;
    d_atomic_numbers = h_atomic_numbers;
}

void Atoms::update_positions(float dt) {
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin(), d_z.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end(), d_z.end()));
    auto zip_vel_begin = thrust::make_zip_iterator(thrust::make_tuple(d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin()));
    thrust::transform(zip_begin, zip_end, zip_vel_begin, zip_begin, PosUpdate(dt));
}

void Atoms::update_velocities(float dt) {
    // zip
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin(), 
            d_force_x.begin(), d_force_y.begin(), d_force_z.begin(),    
            d_masses.begin()
        )
    );
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            d_vel_x.end(), d_vel_y.end(), d_vel_z.end(), 
            d_force_x.end(), d_force_y.end(), d_force_z.end(),    
            d_masses.end()
        )
    );

    // update
    thrust::for_each(zip_begin, zip_end, VelUpdate(dt, conversion_factor));
}

void Atoms::remove_drift() {
    // zip
    auto zip_begin_x = thrust::make_zip_iterator(thrust::make_tuple(d_vel_x.begin(), d_masses.begin()));
    auto zip_end_x = thrust::make_zip_iterator(thrust::make_tuple(d_vel_x.end(), d_masses.end()));

    auto zip_begin_y = thrust::make_zip_iterator(thrust::make_tuple(d_vel_y.begin(), d_masses.begin()));
    auto zip_end_y = thrust::make_zip_iterator(thrust::make_tuple(d_vel_y.end(), d_masses.end()));

    auto zip_begin_z = thrust::make_zip_iterator(thrust::make_tuple(d_vel_z.begin(), d_masses.begin()));
    auto zip_end_z = thrust::make_zip_iterator(thrust::make_tuple(d_vel_z.end(), d_masses.end()));

    // calc drift
    float weighted_sum_x = thrust::transform_reduce(zip_begin_x, zip_end_x, Multiply(), 0.0f, thrust::plus<float>());
    float weighted_sum_y = thrust::transform_reduce(zip_begin_y, zip_end_y, Multiply(), 0.0f, thrust::plus<float>());
    float weighted_sum_z = thrust::transform_reduce(zip_begin_z, zip_end_z, Multiply(), 0.0f, thrust::plus<float>());

    float mass_sum = thrust::reduce(d_masses.begin(), d_masses.end(), 0.0f, thrust::plus<float>());

    float avg_x = weighted_sum_x / mass_sum;
    float avg_y = weighted_sum_y / mass_sum;
    float avg_z = weighted_sum_z / mass_sum;

    // remove drift
    auto zip_vel_begin = thrust::make_zip_iterator(thrust::make_tuple(d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin()));
    auto zip_vel_end = thrust::make_zip_iterator(thrust::make_tuple(d_vel_x.end(), d_vel_y.end(), d_vel_z.end()));
    thrust::transform(zip_vel_begin, zip_vel_end, zip_vel_begin, RemoveDrift(avg_x, avg_y, avg_z));
}

void Atoms::apply_pbc() {
    // zip
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_x.begin(), d_y.begin(), d_z.begin(), 
        d_box_x.begin(), d_box_y.begin(), d_box_z.begin()
    ));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(
        d_x.end(), d_y.end(), d_z.end(), 
        d_box_x.end(), d_box_y.end(), d_box_z.end()
    ));

    // apply pbc
    thrust::for_each(zip_begin, zip_end, ApplyPBC(Lbox));
}

float Atoms::calc_kinetic_energy() {
    // zip
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin(), d_masses.begin()
    ));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(
        d_vel_x.end(), d_vel_y.end(), d_vel_z.end(), d_masses.end()
    ));

    // 運動エネルギーの計算
    float kinetic_energy = thrust::transform_reduce(
        zip_begin, 
        zip_end, 
        CalcKinEnergy(), 
        0.0, 
        thrust::plus<float>()
    );

    return kinetic_energy / conversion_factor;
}