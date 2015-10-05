#include "schedule.hpp"

using namespace std;
using namespace appsdk;

/*Floyd's algorithm to compute the shortest path: used to generate serialized schedule based on computation graph.*/
void floyds(vector< vector <float> >& b) {
    const size_t N = b.size();
    int i, j, k;
    for (k = 0; k < N; k++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if ((b[i][k] + b[k][j] < b[i][j])) {
                    b[i][j] = b[i][k] + b[k][j];
                }
            }
        }
    }
}


/*Initialize the SCHEDULER object.*/
SCHEDULER::SCHEDULER(const vector<UNIT*>& _units) {
	units = _units;
    pos = -1; /* -1 indicates current 'seralized' is invalid.*/
    
    /*create mapping from unit to cg index*/
    mapper[0] = 0; /*terminal*/
    int index = 1;
    for (int i = 0; i < units.size(); i++) {
        if (units[i]->get_type() == UNIT_TYPE_INPUT) {
            /*all input units are collectively denoted by terminal "0", which are not needed to be scheduled. When a forward-backward pass completes, control reaches the terminal (inputs).*/
        } else if (units[i]->get_type() == UNIT_TYPE_SOFTMAX) {			/*SOFTMAX only has forward node*/
            mapper[units[i]->get_nid()] = index++;
            compute_nodes.push_back(new COMPUTE_NODE(units[i], true));	/*'true' because it is a forward pass*/
        } else {														/*All the other units have both forward and backward passes.*/
            mapper[units[i]->get_nid()] = index++;						/*forward pass computation*/
            mapper[(-1) * units[i]->get_nid()] = index++; 				/*backward pass computation (nid is negated)*/
            compute_nodes.push_back(new COMPUTE_NODE(units[i], true));
            compute_nodes.push_back(new COMPUTE_NODE(units[i], false));
        }
    }
    
    /*allocate memory for cg graph*/
    vector<float> tmp;
    for (int i = 0; i < index; i++) {
        tmp.clear();
        for (int j = 0; j < index; j++) {
            tmp.push_back(INF);
        }
        cg.push_back(tmp);
    }
    for (int i = 0; i < index; i++)
        cg[i][i] = 0;

	/*allocate memory for sorting*/
    dist_buf = new float [index - 1];
    order_buf = new int [index - 1];
    
}

/*update adjacent matrix cg based on the walltime of each computation unit, and then generate a new seralized schedule.*/
void SCHEDULER::update_schedule() {
    for (int i = 0; i < units.size(); i++) {
    	/*Update Input units: they only have backward links*/
        if (units[i]->get_type() == UNIT_TYPE_INPUT) {
            /*Update the weight of link from successors of input node to terminal node, by assigning the walltime of backward pass computation.*/
            vector<UNIT*> succ = units[i]->get_succ();		/*Successors of the input unit*/
            int index;
            for (int j = 0; j < succ.size(); j++) {
                index = mapper[(-1) * succ[j]->get_nid()]; /*index of the backward node of this unit in the cg graph*/
                cg[index][0] = succ[j]->get_wall_time(-1); /*assign the backward-pass wall time of this unit as the weight of the edge linking to the terminal*/
            }
        } 
        /*Update Softmax units: they only have forward links*/
        else if (units[i]->get_type() == UNIT_TYPE_SOFTMAX) {
            /*link the unit to all predecessors*/
            vector<UNIT*> pred = units[i]->get_pred();
            int index1 = mapper[units[i]->get_nid()]; /*index of the SOFTMAX unit in cg*/
            int index2; /*index of the predecessors units (backward) of this SOFTMAX in cg*/
            for (int j = 0; j < pred.size(); j++) {
                index2 = mapper[(-1)*pred[j]->get_nid()];
                cg[index1][index2] = units[i]->get_wall_time(1); /*assign the forward-pass wall time of SOFTMAX unit as the weight of the edge linking to the predecessors*/
            }
        } 
        /*Update all the other units: they have both forward and backward links*/
        else {
            vector<UNIT*> pred = units[i]->get_pred();
            vector<UNIT*> succ = units[i]->get_succ();
            int index1 = mapper[units[i]->get_nid()]; /*index of the unit for forward in cg*/
            int index2 = mapper[(-1) * units[i]->get_nid()]; /*index of the unit for backward in cg*/
            int index;
            /*link the forward node to all successors*/
            for (int j = 0; j < succ.size(); j++) {
                index = mapper[succ[j]->get_nid()]; /*index of the successors for forward pass*/
                cg[index1][index] = units[i]->get_wall_time(1); /*assign the forward-pass wall time of the unit as the weight of the edge linking to the successors*/
            }
            /*link the backward node to all predecessors*/
            for (int j = 0; j < pred.size(); j++) {
                if (pred[j]->get_type() == UNIT_TYPE_INPUT)
                    continue; /*If the predecessor is INPUT, then skip it because this unit needs to link to 'terminal' instead of INPUT.*/
                index = mapper[(-1) * pred[j]->get_nid()]; /*index of the predecessors for backward pass*/
                cg[index2][index] = units[i]->get_wall_time(-1); /*assign the backward-pass wall time of the unit as the weight of the edge linking to the predecessors*/
            }
        }

    }
    

    /*Calculate the shortest path from any node to the terminal "0", based on cg*/
    vector< vector <float> > dist = cg;
    floyds(dist);
    
    
    /*generate linear-order schedule by sorting the nodes according to path length to the terminal node*/
    for (int i = 0; i < compute_nodes.size(); i++)
        dist_buf[i] = dist[i + 1][0];
    seralized.clear();
    quick_sort(dist_buf, compute_nodes.size(), order_buf, 0, true);
    for (int i = 0; i < compute_nodes.size(); i++)
        seralized.push_back(compute_nodes[order_buf[i]]);
        

#ifdef DEBUG3
    printf("[mapping] \nTerminal  ");
    for(int i = 0; i<compute_nodes.size();i++)
    	printf("%s[%d]  ", compute_nodes[i]->unit->get_id().c_str(),compute_nodes[i]->is_forward);
    puts("\n[dist]");
    for(int i = 0; i<dist.size();i++){
    	for(int j = 0; j<dist.size(); j++){
    		printf("%d ",(int)dist[i][j]);
    	}
    	puts("");
    }
    int i;
    printf("[seralized] "); fflush(stdout);
    for (i = 0; i < seralized.size() - 1; i++){
        printf("%s[%d](%.2f) -> ", seralized[i]->unit->get_id().c_str(), (int) seralized[i]->is_forward, dist[order_buf[i] + 1][0]);
        fflush(stdout);
    }
    printf("%s[%d](%.2f)\n", seralized[i]->unit->get_id().c_str(), (int) seralized[i]->is_forward, dist[order_buf[i] + 1][0]);
    fflush(stdout);
    puts("---------------------------------------");
#endif

    /*set the serailized as valid*/
    pos = 0;
}


/*Get the next computation nodes to execute.*/
COMPUTE_NODE* SCHEDULER::get_next() {
	if(pos<0) return NULL;		/*no valid seralized schedule*/
    else if (pos < seralized.size()) {
    	pos++;
        return seralized[pos-1];
    } else {					/*reach the end of computation graph (terminal): revoke the current schedule.*/
        pos = -1;
        return NULL; //no more task to run.
    }
}

/*A thread to run either forward or backward pass.*/
void* thread_run_forward_backward(void* _param){
	FORWARD_BACKWARD_PARAM* param = (FORWARD_BACKWARD_PARAM*)_param;
	if(param->is_forward){
		param->unit->forward(*param->dev);		/*Run the forward pass: (unit,device)*/
	}else{
		param->unit->backward(*param->dev);		/*Run the backward pass: (unit,device)*/
	}
}

/*start all the input units as independent threads.*/
void SCHEDULER::start_input_units(){
	for (int i = 0; i < units.size(); i++){
		if (units[i]->get_type() == UNIT_TYPE_INPUT){
			SDKThread* t = new SDKThread;
			FORWARD_BACKWARD_PARAM* p = new FORWARD_BACKWARD_PARAM(units[i],units[i]->get_env().devices[0],true);
			t->create(thread_run_forward_backward,p);
			input_threads.push_back(t);
		}
	}
}


void SCHEDULER::stop_input_units(){
	
}





