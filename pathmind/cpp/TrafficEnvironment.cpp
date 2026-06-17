#include "TrafficEnvironment.h"


trafficEnv:: trafficEnv(int max_steps){
    car_volume=std::vector<float>(10,0.0f);
    occupancy=std::vector<float>(10,0.0f);
    signal_states=std::vector<float>(10,0.0f);
    current_step= 0;
    this->max_steps=max_steps;



    
};

std::vector<float>  trafficEnv::reset(){
    car_volume=std::vector<float>(10,0.0f);
    occupancy=std::vector<float>(10,0.0f);
    signal_states=std::vector<float>(10,0.0f);
    current_step= 0;
    std::vector<float> obs;
    obs.insert(obs.end(), signal_states.begin(), signal_states.end());
    obs.insert(obs.end(), car_volume.begin(), car_volume.end());
    obs.insert(obs.end(), occupancy.begin(), occupancy.end());
    return obs;


    


};

std::tuple<std::vector<float>, float, bool> trafficEnv :: step(int action){
    bool done = false;
    float reward = 0.0f;

    for (int i=0;i<10;i++){
        car_volume[i]+=0.2f;
    }
    signal_states[action]=1.0f;
    car_volume[action] = std::max(0.0f, car_volume[action] - 5.0f); 









    for (float v : car_volume) reward += v;
    reward = -reward;
    current_step++;
    if (current_step==max_steps){
        done=true;
    } else{
        done=false;

    }
    std::vector<float> obs;
    obs.insert(obs.end(), signal_states.begin(), signal_states.end());
    obs.insert(obs.end(), car_volume.begin(), car_volume.end());
    obs.insert(obs.end(), occupancy.begin(), occupancy.end());
    return std::make_tuple(obs, reward, done);

    



};