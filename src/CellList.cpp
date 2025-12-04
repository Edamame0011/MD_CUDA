#include "CellList.hpp"
#include <thrust/host_vector.h>

namespace {
    int icellNo(int ix, int iy, int iz, int M) {
        int idx = ((ix + M) % M) + ((iy + M) % M) * M + ((iz + M) % M) * M * M;
        return idx;
    }

    struct Link {
        int M;
        Link(int _M): M(_M) {}
        template <typename Tuple>
        __host__ __device__ int operator() (const Tuple& t) const {
            int dmyrxi = thrust::get<0>(t) - floorf(thrust::get<0>(t) + 0.5);
            int dmyryi = thrust::get<1>(t) - floorf(thrust::get<1>(t) + 0.5);
            int dmyrzi = thrust::get<2>(t) - floorf(thrust::get<2>(t) + 0.5);
            return (int)((dmyrxi + 0.5) / (float)M) + (int)((dmyryi + 0.5) / (float)M) * M + (int)((dmyrzi + 0.5) / (float)M) * M * M;
        } 
    };
}

CellList::CellList(int _M) : M(_M) {
    init_maps();
}

void CellList::init_maps() {
    // create the maps of the cell list
    int mapsize = 13 * M * M * M;
    thrust::host_vector<int> h_map(mapsize);
    int imap = 0;

    for (int iz = 0; iz < M; iz ++) {
        for (int iy = 0; iy < M; iy ++) {
            for (int ix = 0; ix < M; ix ++) {
                imap = icellNo(ix, iy, iz, M) * 13;
                h_map[imap + 1] = icellNo(ix+1, iy, iz, M);
                h_map[imap + 2] = icellNo(ix+1, iy+1, iz, M);
                h_map[imap + 3] = icellNo(ix, iy+1, iz, M);
                h_map[imap + 4] = icellNo(ix-1, iy+1, iz, M);
                h_map[imap + 5] = icellNo(ix+1, iy, iz-1, M);
                h_map[imap + 6] = icellNo(ix+1, iy+1, iz-1, M);
                h_map[imap + 7] = icellNo(ix, iy+1, iz-1, M);
                h_map[imap + 8] = icellNo(ix-1, iy+1, iz-1, M);
                h_map[imap + 9] = icellNo(ix+1, iy, iz+1, M);
                h_map[imap + 10] = icellNo(ix+1, iy+1, iz+1, M);
                h_map[imap + 11] = icellNo(ix, iy+1, iz+1, M);
                h_map[imap + 12] = icellNo(ix-1, iy+1, iz+1, M);
                h_map[imap + 13] = icellNo(ix, iy, iz+1, M);
            }
        }
    }

    // transfer to device
    d_map = h_map;
}

void CellList::init_cell(const Atoms& atoms) {
    float Lbox = atoms.get_Lbox();
    Lcell = Lbox / (float)M;
}

void CellList::link(const Atoms& atoms) {
    const thrust::device_vector<float>& pos_x = atoms.get_x();
    const thrust::device_vector<float>& pos_y = atoms.get_y();
    const thrust::device_vector<float>& pos_z = atoms.get_z();

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(pos_x.begin(), pos_y.begin(), pos_z.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(pos_x.end(), pos_y.end(), pos_z.end()));

    thrust::transform(zip_begin, zip_end, icell.begin(), Link(M));

    for (int i = 0; i < atoms.get_num_atoms(); i ++) {
        list[i] = head[icell[i]];
        head[icell[i]] = i;
    }
}