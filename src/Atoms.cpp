#include "Atoms.cuh"
#include "constant.cuh"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

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
    VelUpdate(float _dt) : dt(_dt) {}
    template <typename Tuple>
    __host__ __device__ void operator() (Tuple& t) {
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
    __host__ __device__ void operator() (Tuple& t) {
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
    thrust::for_each(zip_begin, zip_end, VelUpdate(dt));
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