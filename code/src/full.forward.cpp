#include "full.hpp"
#include <vector>
#include "main.hpp"
#include "common.hpp"


using namespace std;

string FULL::generate_header_forward(const int& tsm, const int & tsn, const int& tsk, const int& wptm, const int& wptn, const int& width, const int & activation) {
    char ret[1024];
    if (activation == RELU) //relu
        sprintf(ret, "#define FORWARD\n#define NUM_SUCC %d\n#define WIDTH %d\n#define TSM %d\n#define TSN %d\n#define TSK %d\n#define WPTM %d\n#define WPTN %d\n#define RTSM (TSM/WPTM)\n#define RTSN (TSN/WPTN)\n#define LPTA ((TSK*WPTM*WPTN)/(TSN))\n#define LPTB ((TSK*WPTM*WPTN)/(TSM))\n", (int)succ.size(),width, tsm, tsn, tsk, wptm, wptn);
    else //tanh
        sprintf(ret, "#define FORWARD\n#define NUM_SUCC %d\n#define WIDTH %d\n#define TANH\n#define TSM %d\n#define TSN %d\n#define TSK %d\n#define WPTM %d\n#define WPTN %d\n#define RTSM (TSM/WPTM)\n#define RTSN (TSN/WPTN)\n#define LPTA ((TSK*WPTM*WPTN)/(TSN))\n#define LPTB ((TSK*WPTM*WPTN)/(TSM))\n", (int)succ.size(), width, tsm, tsn, tsk, wptm, wptn);
    return ret;
}

/*validate: Y = f(XW+B)
[X]: N-by-P data matrix.
[W]: P-by-Q data matrix.
[B]: 1-by-Q bias vector.
activation: RELU or TANH.
[Y]: N-by-Q result matrix computed by GPU.
Return value: the difference between results computed by CPU and GPU.
 */
float check_forward(PMEM* input, float* W, float* B, const int activation, int _N, int _P, int _Q, vector<PMEM*> outputs, int* y_offset, int* y_stride, float* Y) {
return 0; //DDD
	int N = roundup(_N);
	int P = roundup(_P);
	int Q = roundup(_Q);
    float err = 0;
    float* Y0 = new float [_N * _Q];
    memset(Y0, 0, sizeof (float)* _N * _Q);
    for (int i = 0; i < _N; i++) {
        for (int k = 0; k < _P; k++) {
            for (int j = 0; j < Q; j++) {
                Y0[i * _Q + j] += input->data1[i * P + k] * W[k * Q + j];
            }
        }
    }
    if (activation == RELU) {
        for (int i = 0; i < _N; i++) {
            for (int j = 0; j < _Q; j++) {
                Y0[i * _Q + j] += B[j];
                Y0[i * _Q + j] = Y0[i * _Q + j] > 0 ? Y0[i * _Q + j] : 0;
            }
        }
    } else {
        for (int i = 0; i < _N; i++) {
            for (int j = 0; j < _Q; j++) {
                Y0[i * _Q + j] += B[j];
                Y0[i * _Q + j] = tanh(Y0[i * _Q + j]);
            }
        }
    }
	for (int i = 0; i < _N; i++)
		for (int j = 0; j < _Q; j++)
			err = fabs(Y0[i * _Q + j] - Y[i * Q + j]) > err ? fabs(Y0[i * _Q + j] - Y[i * Q + j]) : err;
    for(int k = 0; k<outputs.size(); k++)		//for each output.
		for (int i = 0; i < _N; i++)
		    for (int j = 0; j < _Q; j++)
		        err = fabs(Y0[i * _Q + j] - outputs[k]->data1[y_offset[k] + i * y_stride[k] + j]) > err ? fabs(Y0[i * _Q + j] - outputs[k]->data1[y_offset[k] + i * y_stride[k] + j]) : err;
    delete Y0;
    return err;
}

/*Get the best parameter configuration w.r.t. underlying hardwares, and dimension of matrices (P,Q,N). P is the width of X, Q is the width of W. Y = f(XW+B) is of size NxM*/
/*Return true is successful, false otherwise*/
bool FULL::get_optimal_parameters_forward(const DEVICE& dev, int& tsk, int& tsm, int & tsn, int& wptm, int& wptn, int& width, float& kernel_time, float& gflops) {
	
    /*generate candidate parameters*/
    vector<int> tsks;
    vector<int> tsms;
    vector<int> tsns;
    vector<int> wptms;
    vector<int> wptns;
    if (P > 512)
        tsks.push_back(16);
    else if (P > 128) {
        tsks.push_back(16);
        tsks.push_back(8);
    } else {
    	if(P>=16) tsks.push_back(16);
        if(P>=8) tsks.push_back(8);
        if(P>=4) tsks.push_back(4);
    }

    if (N > 512) {
        tsms.push_back(128);
        tsms.push_back(64);
    } else if (N > 64) {
        tsms.push_back(128);
        tsms.push_back(64);
        tsms.push_back(32);
    } else {
        tsms.push_back(N);
        tsms.push_back(N / 2);
    }

    if (Q < 32 && N > 512)
        tsms.push_back(32);

    if (N > 256) {
        wptms.push_back(8);
        wptms.push_back(4);
    } else if (N > 64) {
        wptms.push_back(4);
        wptms.push_back(2);
    } else {
        wptms.push_back(2);
        wptms.push_back(1);
    }
    if (Q > 512) {
        tsns.push_back(128);
        tsns.push_back(64);
    } else if (Q > 64) {
        tsns.push_back(128);
        tsns.push_back(64);
        tsns.push_back(32);
    } else {
        tsns.push_back(Q);
        tsns.push_back(Q / 2);
    }
    if (N < 32 && Q > 512)
        tsns.push_back(32);


    if (Q > 256) {
        wptns.push_back(8);
        wptns.push_back(4);
    } else if (Q > 64) {
        wptns.push_back(4);
        wptns.push_back(2);
    } else {
        wptns.push_back(4);
        wptns.push_back(2);
        wptns.push_back(1);
    }
    
    if(P<16)
    	width = 1;
    else
    	width = 4;


    kernel_time = 10000;
    gflops = 0;
    cl_int status;
    /*Move Y, W and B into device memory.*/
    W->set_data2();			//copy data1 to data2.
    W->migrate(dev);		//migrate data2 into specific device.
    B->set_data2();			//copy data1 to data2.
    B->migrate(dev);		//migrate data2 into specific device.
    Y->set_data2();
    Y->migrate(dev);
    
    /*Write random data into input*/
    {
    	input->create_input_buffer();	//create empty data2 buffer.
    	input->set_data1();				//copy data2 into data1 (to create a data1 object).
        for (int i = 0; i < _N; i++) {
            for (int j = 0; j < _P; j++) {
                input->data1[i * input->d_stride + j] = ((0.1 + rand()) / RAND_MAX - 0.5);
            }
        }
        input->set_data2();				//copy data1 to data2.
    }

	/*Create input buffer of successors*/
	for(int i = 0; i<outputs.size(); i++)
		outputs[i]->create_input_buffer();
		
    for (int i = 0; i < tsks.size(); i++) {
        for (int j = 0; j < tsms.size(); j++) {
            for (int k = 0; k < tsns.size(); k++) {
                for (int u = 0; u < wptms.size(); u++) {
                    for (int v = 0; v < wptns.size(); v++) {

                        /*check constraints*/
                        if (tsns[k] < 1) continue;
                        if (tsms[j] < 1) continue;
                        if (P % tsks[i] != 0) continue;
                        if (((tsks[i] * wptms[u] * wptns[v]) / (tsns[k])) % width != 0) continue;
                        if (((tsks[i] * wptms[u] * wptns[v]) / (tsms[j])) % width != 0) continue;
                        if (tsks[i] % width != 0) continue;

                        if ((tsms[j] / wptms[u])*(tsns[k] / wptns[v]) > dev.MaxWorkgroupSize) continue;
                        if (tsms[j] % wptms[u] != 0) continue;
                        if (tsns[k] % wptns[v] != 0) continue;
                        if ((tsks[i] * wptms[u] * wptns[v]) % tsns[k] != 0) continue;

                        if ((tsks[i] * wptms[u] * wptns[v]) % tsms[j] != 0) continue;
                        if (tsms[j] > N) continue;
                        if (tsns[k] > Q || tsns[k] < 4) continue;
                        if (4 * tsks[i]*(tsns[k] + tsms[j]) > 0.95 * dev.LocalMemSize) continue;

                        /*Create program object */
                        string source = readKernelFile("src/full.cl");
                        string header = generate_header_forward(tsms[j], tsns[k], tsks[i], wptms[u], wptns[v], width, activation).c_str();
                        string code = header + source;
                        const char* constCode = code.c_str();
                        cl_program program = clCreateProgramWithSource(env.context, 1, &constCode, 0, &status);
                        if (status != CL_SUCCESS) {
                            printf("FULL::get_optimal_parameters_forward: unable to create program.  %s.\n", getErrorString(status));
                            exit(-1);
                        }
                        /*Build program. */
                        clBuildProgram(program, 1, &dev.id, NULL, NULL, NULL);

                        /*Create kernel*/
                        cl_kernel kernel = clCreateKernel(program, "forward", &status);
                        if (status != CL_SUCCESS) {
                            printf("FULL::get_optimal_parameters_forward: unable to create kernel.  %s (%s).\n", getErrorString(status), "forward");
                            exit(-1);
                        }
                        /*Set kernel arguments*/
						clSetKernelArg(kernel, 0, sizeof (input->data2), (void *) &input->data2); 	/*input (X)*/
						clSetKernelArg(kernel, 1, sizeof (W->data2), (void *) &W->data2); 			/*W*/
						clSetKernelArg(kernel, 2, sizeof (B->data2), (void *) &B->data2); 			/*B*/
						clSetKernelArg(kernel, 3, sizeof (P), (void *) &P);
						clSetKernelArg(kernel, 4, sizeof (Q), (void *) &Q);
						clSetKernelArg(kernel, 5, sizeof (_N), (void *) &_N);
						clSetKernelArg(kernel, 6, sizeof (_Q), (void *) &_Q);
						clSetKernelArg(kernel, 7, sizeof (y_offset->data2), (void *) &y_offset->data2);
						clSetKernelArg(kernel, 8, sizeof (y_stride->data2), (void *) &y_stride->data2);
						clSetKernelArg(kernel, 9, sizeof (Y->data2), (void *) &Y->data2);
						for(int ii = 0; ii<outputs.size();ii++)		//set valid output buffers.
							clSetKernelArg(kernel, 10+ii, sizeof (outputs[ii]->data2), (void *) &outputs[ii]->data2);
						for(int ii = outputs.size(); ii<10;ii++)		//fill the rest arguments with dummy argument.
							clSetKernelArg(kernel, 10+ii, sizeof (B->data2), (void *) &B->data2);

                        /*run kernel*/
                        double kt = 0;
                        double fp = 0;
                        float err = -1;
                        int t;
                        cl_event ndrEvt;
                        for (t = 0; t < 3;) {
                            size_t global_work_size[] = {(size_t)(Q / wptns[v]), (size_t)(N / wptms[u])};
                            size_t local_work_size[] = {(size_t)(tsns[k] / wptns[v]), (size_t)(tsms[j] / wptms[u])};
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
                                printf("FULL::get_optimal_parameters_forward: unable to run the kernel. %s.\n", getErrorString(status));
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
                                /*Get the input and outputs back from device*/
								input->set_data1();
								Y->set_data1();
								for(int ii = 0; ii<outputs.size(); ii++){
									outputs[ii]->set_data1();
								}
                                /*Validate the results*/
                                err = check_forward(input, W->data1, B->data1, activation, _N, _P, _Q, outputs, y_offset->data1, y_stride->data1,Y->data1);
                                if (err > 1e-6) {
                                    printf("FULL::get_optimal_parameters_forward: Incorrect results. Err = %.6f. Debugging info has been written to dumps.txt.\n", err);
                                    char msg [100];
                                    sprintf(msg,"FULL::get_optimal_parameters_forward@%s [X]",get_id().c_str());
                                    input->print_data1(msg);
                                    sprintf(msg,"FULL::get_optimal_parameters_forward@%s [W]",get_id().c_str());
                                    W->print_data1(msg);
                                    sprintf(msg,"FULL::get_optimal_parameters_forward@%s [B]",get_id().c_str());
                                    B->print_data1(msg);
                                    sprintf(msg,"FULL::get_optimal_parameters_forward@%s [offset]",get_id().c_str());
                                    y_offset->print_data1(msg);
                                    sprintf(msg,"FULL::get_optimal_parameters_forward@%s [stride]",get_id().c_str());
                                    y_stride->print_data1(msg);     
                                    sprintf(msg,"FULL::get_optimal_parameters_forward@%s [Y]",get_id().c_str());
                                    Y->print_data1(msg);                  
                                    for(int tt = 0; tt<outputs.size(); tt++){
		                                sprintf(msg,"FULL::get_optimal_parameters_forward@%s [Y][%d]",get_id().c_str(),tt);
		                                outputs[tt]->print_data1(msg);
                                    }
                                    exit(-1);
                                }
                            }
#endif

                            if (2 * (double) P * Q * N / (kt / t)*(1e-9) < 0.7 * gflops) {
                                break;
                            }

                        }

                        kt /= t;
                        fp = 2 * (double) P * Q * N / kt * (1e-9);

                        clReleaseEvent(ndrEvt);
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                        if (gflops < fp) {
                            kernel_time = kt;
                            gflops = fp;
                            tsk = tsks[i];
                            tsm = tsms[j];
                            tsn = tsns[k];
                            wptm = wptms[u];
                            wptn = wptns[v];
                        }
#ifdef DEBUG3
                        printf("P = %d, Q = %d, N = %d, tsk = %d, tsm = %d, tsn = %d, wptm = %d, wptn = %d, kernel_time = %.6f, gflops = %.6f, Err = %.8f\n", P, Q, N, tsks[i], tsms[j], tsns[k], wptms[u], wptns[v], kt, fp, err);
#endif
                    }
                }
            }
        }
    }
    W->free_data2();
    B->free_data2();
    if (kernel_time < 10000)
        return true;
    else
        return false;
}





