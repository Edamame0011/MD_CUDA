#include "Simulator.hpp"
#include "constant.h"
#include <thrust/host_vector.h>
#include <iomanip>

#include <nvtx3/nvtx3.hpp>

void Simulator::run_nve(const float tsim) {
    int steps = (int)tsim / dt;
    steps += current_steps;

    predictor.predict(atoms, NL);
    output();

    const auto logbin = std::pow(10.0f, 1.0f / 9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    while (current_steps < steps) {
        step_nve();
        current_steps ++;
        if (dt * current_steps > checker) {
            output();
            checker *= logbin;
        }
    }
}

void Simulator::step_nve() {
    {
        nvtx3::scoped_range r("update_velocities_1");
        atoms.update_velocities(dt);
    }
    
    {
        nvtx3::scoped_range r("update_positions");
        atoms.update_positions(dt);
    }
    
    {   
        nvtx3::scoped_range r("apply_pbc");
        atoms.apply_pbc();
    }
    
    {
        nvtx3::scoped_range r("NL_check");
        NL.check(atoms);
    }
    
    {
        nvtx3::scoped_range r("predict");
        predictor.predict(atoms, NL);
    }
    
    {
        nvtx3::scoped_range r("update_velocities_2");
        atoms.update_velocities(dt);
    }
}

void Simulator::init_simulation() {
    current_steps = 0;

    atoms.apply_pbc();

    // 隣接リストの作成
    NL.generate(atoms);

    // ログの見出しを出力しておく
    std::cout << "time (fs), kinetic energy (eV), potential energy (eV), total energy (eV), temperature (K)" << std::endl;
}

void Simulator::set_initial_temperature(const float temperature, std::mt19937& mt) {
    // デバイスから質量を転送
    thrust::host_vector<float> masses = atoms.get_masses();
    // 平均0、分散1のガウス分布
    std::normal_distribution<float> dist_trans(0.0, 1);
    int N = atoms.get_num_atoms();

    // ホスト側の配列
    thrust::host_vector<float> h_vel_x(N);
    thrust::host_vector<float> h_vel_y(N);
    thrust::host_vector<float> h_vel_z(N);

    for (int i = 0; i < atoms.get_num_atoms(); i ++) {
        // 分散を調節
        h_vel_x[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / masses[i]);
        h_vel_y[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / masses[i]);
        h_vel_z[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / masses[i]);
    }

    // デバイスに速度を送信
    atoms.set_vel_x(h_vel_x);
    atoms.set_vel_y(h_vel_y);
    atoms.set_vel_z(h_vel_z);

    // 全体速度の除去
    atoms.remove_drift();
}

void Simulator::output() {
    float K = atoms.calc_kinetic_energy();
    float U = atoms.get_potential_energy();
    int dof = 3 * atoms.get_num_atoms();
    float temperature = 2 * K / (dof * boltzmann_constant);
    std::cout << std::setprecision(15) << std::scientific << current_steps * dt << ", "
                                                          << K << ", "
                                                          << U << ", "
                                                          << K + U << ", "
                                                          << temperature << std::endl;
}