#ifndef _SCHEDULE_HPP_
#define _SCHEDULE_HPP_

#include "common.hpp"
#include "main.hpp"
#include <vector>
#include <string>


typedef struct FORWARD_BACKWARD_PARAM{
	UNIT* unit;
	DEVICE* dev;
	bool is_forward;
	FORWARD_BACKWARD_PARAM(UNIT* _unit, DEVICE* _dev, bool _is_forward){
		unit = _unit;
		dev = _dev;
		is_forward = _is_forward;
	}
}FORWARD_BACKWARD_PARAM;



class COMPUTE_NODE {
public:
    float wt; 		/*wall time to run this process*/
    DEVICE* dev; 	/*device to run this process*/
    UNIT* unit; 	/*The unit associated. NULL if this is a 'super-node' denoting the terminal*/
    const bool is_forward; /*true if this is a forward pass, false if it is backward pass.*/

    COMPUTE_NODE(UNIT* u, const bool flag) : is_forward(flag) {
        unit = u;
        dev = unit->get_env().devices[0];
        wt = 1.0;
    }

    /*use some strategies to determine which devices should run this computation: need to be improved later.*/
    void set_device() {
        dev = unit->get_env().devices[0];
        wt = 1.0;
    }
    
    /*run the computation*/
    void run(){
    	if(is_forward)
    		unit->forward(*dev);
    	else
    		unit->backward(*dev);
    }
};


class SCHEDULER{
	std::vector< std::vector <float> > cg;			/*Computation Graph (CG): cg[i][j] is the computational cost (in milli-seconds) from node i to j*/
    std::vector<COMPUTE_NODE*> seralized;			/*Seralized computation order*/
    std::vector<COMPUTE_NODE*> compute_nodes;		/*The list of objects of computation nodes*/
    size_t pos;										/*The next seralized node to be returned.*/
    std::map<int, int> mapper;						/*map uint numeric identifier to index in the cg.*/
    std::vector<UNIT*> units;						/*A list of all units in the system.*/
    float* dist_buf;								/*This is a buffer used for quick sort.*/
    int* order_buf;									/*This is a buffer used for quick sort.*/
    std::vector<appsdk::SDKThread*> input_threads;	/*The threads that are running the INPUT units.*/
public:

	SCHEDULER(const std::vector<UNIT*>& _units);

	/*Generate linear-order scheduling scheme.*/
	void update_schedule();
	
	/*Return the next computation node to run based on the linear-order schedule*/
	COMPUTE_NODE* get_next();
	
	/*Start all the input units*/
	void start_input_units();
	
	/*Stop all the input units*/
	void stop_input_units();
	
};


#endif
