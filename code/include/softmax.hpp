#ifndef _MAT_SOFTMAX_HPP_
#define _MAT_SOFTMAX_HPP_
#include "main.hpp"
#include "common.hpp"
#include <string>


class SOFTMAX:public UNIT{
	int* batch_labels;					/*batch_labels is the label where batch_labels[n] is index (e.g. 0,1,...) of the true class of n-th label..*/
public:
	SOFTMAX(const std::string& _id, const CL_ENV& _env):UNIT(_id,_env,UNIT_TYPE_SOFTMAX){
	}
	
	void init();
	
	void forward(DEVICE& device);
	
	void backward(DEVICE& device);
	
	void set_label(int* unit_input_label){
		batch_labels = unit_input_label;
	}
};
#endif
