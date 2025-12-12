#include "Atoms.hpp"
#include "NeighbourList.hpp"
#include "Predictor_libtorch.hpp"
#include "Simulator.hpp"
#include "constant.h"

#include <string>
#include <chrono>

int main() {
    // シミュレーション定数の設定
    float dt = 0.5;
    float cutoff = 5.0;
    float margin = 1.0;
    float temperature = 300;

    std::string data_path = "./data/sample_NS2.xyz";
    std::string model_path = "./models/deployed_model_Na2O-SiO2.pt";

    std::mt19937 mt(123456789);

    Atoms atoms(data_path);
    NeighbourList NL(cutoff, margin);
    Predictor_libtorch predictor(atoms, model_path);

    Simulator simulator(atoms, NL, predictor, dt);
    simulator.set_initial_temperature(temperature, mt);
    simulator.init_simulation();

    // 時間の計測
    auto start = std::chrono::steady_clock::now();

    simulator.run_nve(1e+3);

    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
}   