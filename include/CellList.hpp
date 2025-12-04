#ifndef CELLLIST_HPP
#define CELLLIST_HPP

#include <thrust/device_vector.h>
#include "Atoms.hpp"

class CellList {
    public:
        CellList(int _Lcell);
        void init_maps();
        void init_cell(const Atoms& atoms);

        void link(const Atoms& atoms);

    private:
        int M;
        float Lcell;

        thrust::device_vector<int> d_map;

        thrust::device_vector<int> icell;
        thrust::device_vector<int> head;
        thrust::device_vector<int> list;
};

#endif