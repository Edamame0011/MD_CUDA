#include <thrust/device_vector.h>
#include <string>

class Atoms {
    public:
        Atoms(std::string path);

        // 生ポインタの取得
        // 位置
        float* x_ptr() const { return thrust::raw_pointer_cast(d_x.data()); }
        float* y_ptr() const { return thrust::raw_pointer_cast(d_y.data()); }
        float* z_ptr() const { return thrust::raw_pointer_cast(d_z.data()); }

        // 速度
        float* vel_x_ptr() const { return thrust::raw_pointer_cast(d_vel_x.data()); }
        float* vel_y_ptr() const { return thrust::raw_pointer_cast(d_vel_y.data()); }
        float* vel_z_ptr() const { return thrust::raw_pointer_cast(d_vel_z.data()); }

        // 力
        float* force_x_ptr() const { return thrust::raw_pointer_cast(d_force_x.data()); }
        float* force_y_ptr() const { return thrust::raw_pointer_cast(d_force_y.data()); }
        float* force_z_ptr() const { return thrust::raw_pointer_cast(d_force_z.data()); }

        // 質量・原子番号は読み取り専用
        const float* masses_ptr() const { return thrust::raw_pointer_cast(d_masses.data()); }
        const uint8_t* atomic_numbers_ptr() const { return thrust::raw_pointer_cast(d_atomic_numbers.data()); }

        // セッター
        // 座標
        void set_x(thrust::device_vector<float>& x) { this->d_x.swap(x); }
        void set_y(thrust::device_vector<float>& y) { this->d_x.swap(y); }
        void set_z(thrust::device_vector<float>& z) { this->d_x.swap(z); }

        // 速度
        void set_vel_x(thrust::device_vector<float>& vel_x) { this->d_vel_x.swap(vel_x); }
        void set_vel_y(thrust::device_vector<float>& vel_y) { this->d_vel_y.swap(vel_y); }
        void set_vel_z(thrust::device_vector<float>& vel_z) { this->d_vel_z.swap(vel_z); }

        // 力
        void set_force_x(thrust::device_vector<float>& force_x) { this->d_force_x.swap(force_x); }
        void set_force_y(thrust::device_vector<float>& force_y) { this->d_force_y.swap(force_y); }
        void set_force_z(thrust::device_vector<float>& force_z) { this->d_force_z.swap(force_z); }

        // 更新
        void update_positions();
        void update_velocities();

        // その他
        void remove_drift();

    private:
        // 原子のデータ
        // 座標
        thrust::device_vector<float> d_x;   // {N, }
        thrust::device_vector<float> d_y;   // {N, }
        thrust::device_vector<float> d_z;   // {N, }

        // 速度
        thrust::device_vector<float> d_vel_x;   // {N, }
        thrust::device_vector<float> d_vel_y;   // {N, }
        thrust::device_vector<float> d_vel_z;   // {N, }

        // 力
        thrust::device_vector<float> d_force_x; // {N, }
        thrust::device_vector<float> d_force_y; // {N, }
        thrust::device_vector<float> d_force_z; // {N, }

        thrust::device_vector<float> d_masses;    // {N, }
        thrust::device_vector<uint8_t> d_atomic_numbers;  // {N, }

        // エネルギー
        thrust::device_vector<float> d_potential_energy;    // {1, }

        thrust::device_vector<int> box;

        // ホスト側で保持する値
        int num_atoms;  // 原子数
        float Lbox;     // シミュレーションボックスのサイズ
};