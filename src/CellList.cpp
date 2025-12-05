#include "CellList.hpp"
#include <thrust/host_vector.h>

namespace {
    int icellNo(int ix, int iy, int iz, int M) {
        int idx = ((ix + M) % M) + ((iy + M) % M) * M + ((iz + M) % M) * M * M;
        return idx;
    }

    struct Link {
        const float *d_x, *d_y, *d_z;
        int* list, *head;
        int M;
        Link(
            float* _d_x, 
            float* _d_y, 
            float* _d_z, 
            int* _list, 
            int* _head, 
            int _M
        ) : d_x(_d_x), d_y(_d_y), d_z(_d_z), list(_list), head(_head), M(_M) {}
        __host__ __device__ void operator() (const int idx) const {
            int dmyrxi = d_x[idx] - floorf(d_x[idx] + 0.5);
            int dmyryi = d_y[idx] - floorf(d_y[idx] + 0.5);
            int dmyrzi = d_z[idx] - floorf(d_z[idx] + 0.5);
            int icell = (int)((dmyrxi + 0.5) / (float)M) + (int)((dmyryi + 0.5) / (float)M) * M + (int)((dmyrzi + 0.5) / (float)M) * M * M;

            list[idx] = head[icell];
            head[icell] = idx;
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

void CellList::link(Atoms& atoms) {
    thrust::for_each(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(atoms.get_num_atoms()),
        Link(
            atoms.x_ptr(), 
            atoms.y_ptr(), 
            atoms.z_ptr(), 
            thrust::raw_pointer_cast(list.data()), 
            thrust::raw_pointer_cast(head.data()), 
            M
        ) 
    );
}