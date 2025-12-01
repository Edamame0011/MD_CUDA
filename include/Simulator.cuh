#include <string>
#include <thrust/device_vector.h>

#include "Atoms.cuh"
#include "Predictor.cuh"

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

        // シミュレーションの状態
        int num_steps;
};

// GPU側の定数として保持するパラメータ
struct SimParams {
    int num_atoms;  // 粒子数
    float Lbox;     // シミュレーションボックスのサイズ
    float dt;       // 時間刻み幅
    float cutoff;   // カットオフ距離
    float margin;   // マージン距離

    float boltzmann_constant;   // ボルツマン低数
    float conversion_factor;    // 単位変換係数
};

extern __const__ SimParams c_params;