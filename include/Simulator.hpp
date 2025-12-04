#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <string>
#include <thrust/device_vector.h>

#include "Atoms.hpp"
#include "Predictor.hpp"

class Simulator {
    public:
        void run_nve(const float tsim);
        void run_nvt(const float tsim, const float temperature);
        void run_anneal(const float cooling_rate, const float start_temperature, const float target_temperature);

    private:
        // シミュレーションの補助用関数
        void set_initial_temperature(const float temperature);
        void init_simulation();
        void step_nve();
        void step_nvt();

        Atoms atoms;
        NeighbourList NL;
        Predictor predictor;

        // シミュレーション設定
        float dt;

        // シミュレーションの状態
        int num_steps;
};

#endif