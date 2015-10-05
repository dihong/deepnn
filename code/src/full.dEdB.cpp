#include "full.hpp"
#include <vector>
#include "main.hpp"
#include "common.hpp"


using namespace std;

string FULL::generate_header_dEdB(const DEVICE& dev, const int& rtsm, const int& rtsn) {
    if (dev.MaxWorkgroupSize % 64 == 0) {
        char ret [300];
        if (activation == TANH)
            sprintf(ret, "#define TANH\n#define NUM_SUCC %d\n#define DEDB\n#define RTSM %d\n#define RTSN %d\n", (int)succ.size(), rtsm, rtsn);
        else
            sprintf(ret, "#define RELU\n#define NUM_SUCC %d\n#define DEDB\n#define RTSM %d\n#define RTSN %d\n", (int)succ.size(), rtsm, rtsn);
        return ret;
    } else {
        printf("FULL::generate_header_dEdB: device not supported. Requre mod(MaxWorkgroupSize,32) = 0. MaxWorkgroupSize = %ld\n", dev.MaxWorkgroupSize);
        exit(-1);
    }
}

/*Check: dEdB = dEdY.*f'(z) */
float check_dEdB(float* dEdB, float* Y, int _N, int _P, int _Q, int activation, vector<PMEM*> outputs, int* y_offset, int* y_stride) {

    float err = 0;
    int N = roundup(_N);
    int P = roundup(_P);
    int Q = roundup(_Q);
    
    float tmp;
    if (activation == TANH) {
        for (int i = 0; i < _N; i++)
            for (int j = 0; j < _Q; j++) {
            	tmp = 0;
            	for(int k = 0; k<outputs.size(); k++){
            		tmp += outputs[k]->data1[y_offset[k]+i*y_stride[k]+j]*(1 - Y[i * Q + j] * Y[i * Q + j]);
            	}
                tmp = fabs(dEdB[i * Q + j] - tmp);
                if (tmp > err) err = tmp;
            }
    } else {
        for (int i = 0; i < _N; i++)
            for (int j = 0; j < _Q; j++) {
            	tmp = 0;
            	if (Y[i * Q + j] > 0){
		        	for(int k = 0; k<outputs.size(); k++){
		        		tmp += outputs[k]->data1[y_offset[k]+i*y_stride[k]+j];
		        	}
            	}
                tmp = fabs(dEdB[i * Q + j] - tmp);
                if (tmp > err) err = tmp;
            }
    }

    return err;
}

bool FULL::get_optimal_parameters_dEdB(const DEVICE& dev, int& rtsm, int& rtsn, float& kernel_time) {

    /*generate candidate parameters*/
    if (dev.MaxWorkgroupSize % 64 == 0) {
        rtsn = 64;
        rtsn = rtsn > Q ? Q : rtsn;
        rtsm = dev.MaxWorkgroupSize / rtsn;
        rtsm = rtsm > N ? N : rtsm;
    } else {
        printf("FULL::get_optimal_parameters_dEdB: device not supported. Requre mod(MaxWorkgroupSize,32) = 0. MaxWorkgroupSize = %ld\n", dev.MaxWorkgroupSize);
        return false;
    }

    kernel_time = 10000;
    cl_int status;
    
	/*Create input buffer of successors and write random data into them (dEdY)*/
	for(int k = 0; k<outputs.size(); k++){
		outputs[k]->create_input_buffer();		//create data2 buffer.
		outputs[k]->set_data1();				//create data1 buffer.
		for (int i = 0; i < _N; i++)
		    for (int j = 0; j < _Q; j++)
		        outputs[k]->data1[y_offset->data1[k] + i * y_stride->data1[k] + j] = (0.1 + rand()) / RAND_MAX;
		outputs[k]->set_data2();				//copy data1 to data2.
	}
    
    /*Move dEdB and Y into device memory.*/
	dEdB->set_data2();
	dEdB->migrate(dev);
	for (int i = 0; i < _N; i++)
		for (int j = 0; j < _Q; j++)
			Y->data1[i*Q+j] = (0.1 + rand()) / RAND_MAX;
    Y->set_data2();
    Y->migrate(dev);

    /*Create program object */
    string source = readKernelFile("src/full.cl");
    string header = generate_header_dEdB(dev, rtsm, rtsn).c_str();
    string code = header + source;
    const char* constCode = code.c_str();
    cl_program program = clCreateProgramWithSource(env.context, 1, &constCode, 0, &status);
    if (status != CL_SUCCESS) {
        printf("FULL::get_optimal_parameters_dEdB: unable to create program.  %s.\n", getErrorString(status));
        puts(constCode);
        exit(-1);
    }

    /*Build program. */
    clBuildProgram(program, 1, &dev.id, NULL, NULL, NULL);

    /*Creae kernel*/
    cl_kernel kernel = clCreateKernel(program, "dEdB", &status);
    if (status != CL_SUCCESS) {
        printf("FULL::get_optimal_parameters_dEdB: unable to create kernel.  %s (%s).\n", getErrorString(status), "dEdB");
        puts(constCode);
        exit(-1);
    }

    /*Set kernel arguments*/
    clSetKernelArg(kernel, 0, sizeof (Y->data2), (void *) &Y->data2); 										/*Y*/
    clSetKernelArg(kernel, 1, sizeof (dEdB->data2), (void *) &dEdB->data2); 									/*dEdB*/
    clSetKernelArg(kernel, 2, sizeof (int), (void *) &Q);
    clSetKernelArg(kernel, 3, sizeof (int), (void *) &_N);
    clSetKernelArg(kernel, 4, sizeof (int), (void *) &_Q);
    clSetKernelArg(kernel, 5, sizeof (y_offset->data2), (void *) &y_offset->data2);
    clSetKernelArg(kernel, 6, sizeof (y_stride->data2), (void *) &y_stride->data2);
    for(int ii = 0; ii<outputs.size();ii++)		//set valid output buffers.
    	clSetKernelArg(kernel, 7+ii, sizeof (outputs[ii]->data2), (void *) &outputs[ii]->data2);
    for(int ii = outputs.size(); ii<10;ii++)		//fill the rest arguments with dummy argument.
    	clSetKernelArg(kernel, 7+ii, sizeof (dEdB->data2), (void *) &dEdB->data2);
	
    /*run kernel*/
    double kt = 0;
    double fp = 0;
    float err = -1;
    int t;
    cl_event ndrEvt;
    for (t = 0; t < 1;) {
        size_t global_work_size[] = {(size_t)(Q), (size_t)(N)};
        size_t local_work_size[] = {(size_t)(rtsn),(size_t)(rtsm)};
        status = clEnqueueNDRangeKernel(
                dev.compute_q,
                kernel,
                2, //work dimension.
                NULL, //global work offset.
                global_work_size,
                local_work_size,
                0, //number of events in wait list.
                NULL, //event wait list.
                &ndrEvt);
        if (status != CL_SUCCESS) {
            printf("FULL::get_optimal_parameters_dEdB: unable to run the kernel. %s.\n", getErrorString(status));
            exit(-1);
        }
        clFlush(dev.compute_q);
        clWaitForEvents(1, &ndrEvt);
        /*Get profiling info.*/
        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_START, sizeof (cl_ulong), &start, NULL);
        clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_END, sizeof (cl_ulong), &end, NULL);
        t++;
        kt += (cl_double) (end - start)*(cl_double) (1e-09);
#ifdef DEBUG0
        if (t == 1) {
            /*Get the dEdB back from device*/
            dEdB->set_data1();

            /*Validate the results*/
            err = check_dEdB(dEdB->data1, Y->data1, _N, _P, _Q, activation, outputs, y_offset->data1, y_stride->data1);
            if (err > 1e-6) {
                printf("P = %d, Q = %d, N = %d, rtsm = %d, rtsn = %d\t", P, Q, N, rtsm, rtsn);
                printf("FULL::get_optimal_parameters_dEdB: Incorrect results. Err = %.6f. Debugging info has been written to dumps.txt.\n", err);
				char msg [100];
				sprintf(msg,"FULL::get_optimal_parameters_dEdB@%s [Y]",get_id().c_str());
				Y->print_data1(msg);
				sprintf(msg,"FULL::get_optimal_parameters_dEdB@%s [offset]",get_id().c_str());
				y_offset->print_data1(msg);
				sprintf(msg,"FULL::get_optimal_parameters_dEdB@%s [stride]",get_id().c_str());
				y_stride->print_data1(msg);     
				sprintf(msg,"FULL::get_optimal_parameters_dEdB@%s [dEdB]",get_id().c_str());
				dEdB->print_data1(msg);                
				for(int tt = 0; tt<outputs.size(); tt++){
					sprintf(msg,"FULL::get_optimal_parameters_dEdB@%s [dEdY][%d]",get_id().c_str(),tt);
					outputs[tt]->print_data1(msg);
				}
                exit(-1);
            }
        }
#endif

    }
    kt /= t;

    clReleaseEvent(ndrEvt);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    if (kernel_time > kt) {
        kernel_time = kt;
    }
#ifdef DEBUG3
    printf("P = %d, Q = %d, N = %d, rtsm = %d, rtsn = %d, kernel_time = %.6f, Err = %.8f\n", P, Q, N, rtsm, rtsn, kernel_time,err);
#endif
	Y->free_data2();
	dEdB->free_data2();
    if (kernel_time < 10000)
        return true;
    else
        return false;
}



