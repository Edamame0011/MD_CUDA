#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <string>
#include <thrust/device_vector.h>
#include <random>

#include "Atoms.hpp"
#include "Predictor_libtorch.hpp"
#include "NeighbourList.hpp"

class Simulator {
    public:
        Simulator(Atoms& _atoms, NeighbourList& _NL, Predictor_libtorch& _predictor, float _dt) : atoms(_atoms), NL(_NL), predictor(_predictor), dt(_dt) {}
        void run_nve(const float tsim);
//        void run_nvt(const float tsim, const float temperature);
//        void run_anneal(const float cooling_rate, const float start_temperature, const float target_temperature);
        void set_initial_temperature(const float temperature, std::mt19937& mt);
        void init_simulation();


    private:
        // シミュレーションの補助用関数
        void step_nve();
        void output();
//        void step_nvt();

        Atoms& atoms;
        NeighbourList& NL;
        Predictor_libtorch& predictor; 


        // シミュレーション設定
        float dt;

        // シミュレーションの状態
        int current_steps;
        float current_temperature;
};

#endif