#ifndef _INPUT_HPP_
#define _INPUT_HPP_
#include "main.hpp"
#include "common.hpp"
#include <string>
#include <vector>


class INPUT:public UNIT{
	size_t _N, _H, _W, _D;					/*Actual size of the problem before bits alginment*/
	std::vector<std::string> sample_paths;	/*path to the training samples on the hard drive*/
	size_t pos;								/*the current position to read the training samples.*/
	size_t num_samples;
	std::vector<int> sample_labels;
public:
	INPUT(const std::string& _id, const CL_ENV& _env, const std::vector<std::string>& paths, const std::vector<int>& labels):UNIT(_id,_env,UNIT_TYPE_INPUT){
		sample_paths = paths;
		sample_labels = labels;
		pos = 0;
		labels_exposed_to_softmax = 0;
	}
	
	void init();
	
	/*Read a batch of data into pinned memory area.*/
	void forward(DEVICE& device);
	
	
	/*INPUT unit has no backward pass*/
	void backward(DEVICE& device){}
	
	int* labels_exposed_to_softmax;
	
};
#endif
