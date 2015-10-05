#ifndef _GRAPH_HPP_
#define _GRAPH_HPP_

#include "stdio.h"
#include "main.hpp"
#include "common.hpp"
#include <string>
#include <vector>
#include <map>

class GRAPH {
    /*All the units in the graph*/
    std::vector<UNIT*> units;
    /*Batch size of training samples*/
    int size_batch;
    CL_ENV* env;
    /*Number of training classes.*/
    int num_classes;

public:
    /*Initialize a graph object based on given network file. Return true on success, false otherwise.*/
    bool init(CL_ENV* _env, std::string path_to_network_file, std::string path_to_train_spec, std::string path_to_solver_param);

    /*Validate the graph: returnt true if the graph is valid, false otherwise.*/
    bool validate();

    /*Propagate parameter settings along the graph to initialize all units.*/
    bool build();
    
    std::vector<UNIT*> get_units(){return units;}
};

#endif
