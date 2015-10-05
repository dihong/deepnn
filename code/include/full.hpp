#ifndef _MAT_MUL_HPP_
#define _MAT_MUL_HPP_
#include "main.hpp"
#include "common.hpp"
#include <string>


/*This struct stores device and unit dependent information*/
typedef struct DEV{
	float kt; 							/*kernel time to compute using this device.*/
	float gflops;
	int tsk, tsm, tsn, wptm, wptn, rtsm, rtsn;  //optimal parameters.
	int width;							/*OpenCL vector loading width*/
	cl_kernel kernel;					/*Built OpenCL kernel*/
	size_t global_work_size[2];			/*Optimal global work size to run the unit*/
	size_t local_work_size[2];			/*Optimal local work size to run the unit*/
}DEV;

class FULL:public UNIT{
	DEV* Paramsforward;					/*forward-pass parameters.*/
	DEV* ParamsdEdX;					/*dEdX parameters*/
	DEV* ParamsdEdW;					/*dEdW parameters*/
	DEV* ParamsdEdB;					/*dEdB parameters*/
	int activation;						/*TANH or RELU*/
	int P, Q, N; 						/*Size of the problem (aligned)*/
	int _P, _Q, _N;						/*Actual size of the problem before bits alginment*/
	MATRIX<float> *W, *B, *dEdW, *dEdB;	/*Unit-specific parameters*/
	MATRIX<int> *y_offset, *y_stride;	/*(offset,stride) pair of outputs*/
	MATRIX<float> * Y;					/*A copy of output. It is useful for backward pass derivative calculation.*/
	SOLVER* sol;						/*Solver to update the model parameter*/
	bool get_optimal_parameters_forward(const DEVICE& dev, int& tsk, int& tsm, int & tsn, int& wptm, int& wptn, int& width, float& kernel_time, float& gflops);
	bool get_optimal_parameters_dEdX   (const DEVICE& dev, int& tsk, int& tsm, int & tsn, int& wptm, int& wptn, int& width, float& kernel_time, float& gflops);
	bool get_optimal_parameters_dEdW(const DEVICE& dev, int& tsk, int& tsm, int & tsn, int& wptm, int& wptn, int& width, float& kernel_time, float& gflops);
	bool get_optimal_parameters_dEdB(const DEVICE& dev, int& rtsm, int& rtsn, float& kernel_time);
	std::string generate_header_dEdB(const DEVICE& dev, const int& rtsm, const int& rtsn);
	std::string generate_header_forward(const int& tsm, const int & tsn, const int& tsk, const int& wptm, const int& wptn, const int& width, const int & activation);
	std::string generate_header_dEdX(const int& tsn, const int & tsp, const int& tsq,  const int& wptn, const int& wptp, const int& width);
	std::string generate_header_dEdW(const int& tsm, const int & tsn, const int& tsk,  const int& wptm, const int& wptn, const int& width);
	/*Create kernels for all devices*/
	void create_kernel(std::string kernel_name, std::string header);
public:
	FULL(const std::string& _id, const CL_ENV& _env):UNIT(_id,_env,UNIT_TYPE_FULL){
		activation = RELU;
	}
	
	/*Initialize the unit:
		1) Get the optimal parameter settings for each device, based on size of the problem.
		2) Get the running time information for the unit on each device.
		3) Get the resource requirement for the problem.
		4) Create OpenCL program and kernel for each device.
	*/
	void init();
	
	void forward(DEVICE& device);
	
	void backward(DEVICE& device);
	
	
	void set_activation(const int & a){
		activation = a;
	}
	
	void set_Q(const int& QQ){
		if(QQ<1){
			printf("FULL::set_Q: invalid Q value '%d'. It must be positive integer.\n",QQ);
			exit(-1);
		}
		_Q	=  	QQ;
		Q 	= 	roundup(_Q);
	}
	
	void set_N(const int& NN){
		if(NN<1){
			printf("FULL::set_N: invalid N value '%d'. It must be positive integer.\n",NN);
			exit(-1);
		}
		_N	=  	NN;
		N 	= 	roundup(_N);
	}
};
#endif
