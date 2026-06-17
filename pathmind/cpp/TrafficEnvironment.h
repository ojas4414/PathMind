
#pragma once
#include <vector>
#include <tuple>



class trafficEnv{
    private:
        std::vector<float>occupancy;
        std::vector<float>car_volume;
        std::vector<float>signal_states;
        int current_step;
        int max_steps;
    
    public:
        std::vector<float> reset();
        std::tuple< std::vector<float>,float,bool> step(int action);
        trafficEnv(int max_steps);


};