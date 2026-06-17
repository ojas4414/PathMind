#include "TrafficEnvironment.h"
#include <pybind11/pybind11.h>
#include  <pybind11/stl.h>

namespace py=pybind11;


PYBIND11_MODULE(traffic_env,m){
    py::class_<trafficEnv>(m,"trafficEnv")
    .def(py::init<int>())
    .def("reset",&trafficEnv::reset)//same as pandas chaingin in python df.dropna().reset_index().rename(columns={...})//
    .def("step",&trafficEnv::step);

}