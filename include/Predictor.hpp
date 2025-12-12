#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"

class Predictor {
    public:
        virtual void predict(Atoms& atoms, NeighbourList& NL) = 0;
        virtual ~Predictor() {}
};

#endif