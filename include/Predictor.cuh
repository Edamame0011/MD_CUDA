#include <string>
#include <thrust/device_vector.h>

#include "Atoms.cuh"
#include "NeighbourList.cuh"

class Predictor {
    public:
        Predictor(std::string model_path);

        void load_model(std::string model_path);
        void predict(Atoms& atoms, const NeighbourList& NL);
    
    private:
        // インプット元
        thrust::device_vector<int> x;
        thrust::device_vector<int> edge_index;
        thrust::device_vector<float> edge_weight;
};