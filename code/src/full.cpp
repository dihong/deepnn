#include "full.hpp"
#include <vector>
#include "main.hpp"
#include "common.hpp"
#include <random>

using namespace std;

/*Create kernels for all devices*/
void FULL::create_kernel(string kernel_name, string header) {
    cl_int status;
    for (int i = 0; i < env.devices.size(); i++) {
        /*Create program object*/
        string source = readKernelFile("src/full.cl");
        string code = header + source;
        const char* constCode = code.c_str();
        /*Build program*/
        cl_program program = clCreateProgramWithSource(env.context, 1, &constCode, 0, &status);
        if (status != CL_SUCCESS) {
            printf("FULL::create_kernel: unable to create program. %s.\n", getErrorString(status));
            exit(-1);
        }
        clBuildProgram(program, 1, &env.devices[i]->id, NULL, NULL, NULL);
        /*create kernel*/
        if (kernel_name == "dEdX")
            ParamsdEdX[i].kernel = clCreateKernel(program, kernel_name.c_str(), &status);
        if (kernel_name == "dEdW")
            ParamsdEdW[i].kernel = clCreateKernel(program, kernel_name.c_str(), &status);
        if (kernel_name == "dEdB")
            ParamsdEdB[i].kernel = clCreateKernel(program, kernel_name.c_str(), &status);
        if (kernel_name == "forward")
            Paramsforward[i].kernel = clCreateKernel(program, kernel_name.c_str(), &status);

        if (status != CL_SUCCESS) {
            printf("FULL::create_kernel: unable to create kernel.  %s (%s).\n", getErrorString(status), kernel_name.c_str());
#ifdef DEBUG3
            puts(constCode);
#endif
            exit(-1);
        }
    }
}


//##########################################################################################################################################################################################



void FULL::forward(DEVICE& device) {

#ifdef DEBUG3
    printf("Entering forward@%s\n", get_id().c_str());
#endif

    /*Find index of the device*/
    int idx_device = -1;
    for (int i = 0; i < env.devices.size(); i++) {
        if (env.devices[i]->id == device.id) {
            idx_device = i;
            break;
        }
    }

#ifdef DEBUG0
    if (idx_device < 0) {
        puts("FULL::forward: cannot find a matching device.");
        exit(-1);
    }
#endif

    /*B(W,B,Y)*/
    cl_int status;
    cl_event event;
    device.reserve_mem((P * Q + Q) * sizeof (float)); //Reserve global memory on target device. Block until requested amount memory has been granted.
    W->set_data2();			//copy data1 to data2.
    W->migrate(device);		//migrate data2 into specific device.
    B->set_data2();			//copy data1 to data2.
    B->migrate(device);		//migrate data2 into specific device.
    Y->set_data2();
    Y->migrate(device);
    
#ifdef DEBUG4
	// X, W and B.
	//W->print_data1("forward: W@"+get_id());
	//B->print_data1("forward: B@"+get_id());
#endif

	/*Create input buffer of successors*/
	for(int i = 0; i<outputs.size(); i++){
		outputs[i]->create_input_buffer();
		outputs[i]->migrate(device);
	}
		
    /*Set kernel arguments*/
    clSetKernelArg(Paramsforward[idx_device].kernel, 0, sizeof (input->data2), (void *) &input->data2); 	/*input (X)*/
    clSetKernelArg(Paramsforward[idx_device].kernel, 1, sizeof (W->data2), (void *) &W->data2); 			/*W*/
    clSetKernelArg(Paramsforward[idx_device].kernel, 2, sizeof (B->data2), (void *) &B->data2); 			/*B*/
    clSetKernelArg(Paramsforward[idx_device].kernel, 3, sizeof (P), (void *) &P);
    clSetKernelArg(Paramsforward[idx_device].kernel, 4, sizeof (Q), (void *) &Q);
    clSetKernelArg(Paramsforward[idx_device].kernel, 5, sizeof (_N), (void *) &_N);
    clSetKernelArg(Paramsforward[idx_device].kernel, 6, sizeof (_Q), (void *) &_Q);
    clSetKernelArg(Paramsforward[idx_device].kernel, 7, sizeof (y_offset->data2), (void *) &y_offset->data2);
    clSetKernelArg(Paramsforward[idx_device].kernel, 8, sizeof (y_stride->data2), (void *) &y_stride->data2);
    clSetKernelArg(Paramsforward[idx_device].kernel, 9, sizeof (Y->data2), (void *) &Y->data2);
    for(int i = 0; i<succ.size();i++)		//set valid output buffers.
    	clSetKernelArg(Paramsforward[idx_device].kernel, 10+i, sizeof (outputs[i]->data2), (void *) &outputs[i]->data2);
    for(int i = succ.size(); i<10;i++)		//fill the rest arguments with dummy argument.
    	clSetKernelArg(Paramsforward[idx_device].kernel, 10+i, sizeof (B->data2), (void *) &B->data2);
                        
	                        
    /*C(Y): compute Y = f(W'X+B) once the device is idle: this is guaranteed by linear order of command queues (thread-safe).*/
    status = clEnqueueNDRangeKernel(
            device.compute_q,
            Paramsforward[idx_device].kernel,
            2,
            NULL,
            Paramsforward[idx_device].global_work_size,
            Paramsforward[idx_device].local_work_size,
            forward_event_wait_list.size(),
            &forward_event_wait_list[0], 		//Wait all predecessors complete their work.
            &event);
    if (status != CL_SUCCESS) {
        printf("FULL::forward: unable to run the kernel. %s.\n", getErrorString(status));
        exit(-1);
    }
    clFlush(device.compute_q);

#ifdef DEBUG3
    printf("%s is waiting for signal.\n", get_id().c_str());
#endif
    clWaitForEvents(1, &event);					//Wait computation completes.
    
#ifdef DEBUG4
	// print input, outputs.
	input->print_data2("forward: X@"+get_id());
#endif
    
    /*Inform all the successors that output of this unit is ready.*/
    signal_forward_ready();


    /*Read X and Y back from device to host memory for future backward pass.*/
    input->set_data1();
    Y->set_data1();


#ifdef DEBUG4
	// print input, outputs.
	Y->print_data1("forward: Y@"+get_id());
	char msg[100];
	for(int i = 0; i<outputs.size(); i++){
		sprintf(msg,"forward: Y[%d]@%s", i, get_id().c_str());
		outputs[i]->print_data2(msg);
	}
#endif

   
    /*Free memory*/
    W->free_data2();
    B->free_data2();
    input->free_data2();
    Y->free_data2();
    device.release_mem((P * Q + Q) * sizeof (float));

    /*Clear the forward event wait list set by predecessors.*/
    clear_forward_event_list();


    /*clean up*/
    clReleaseEvent(event);

#ifdef DEBUG3
    printf("Leaving forward@%s\n", get_id().c_str());
#endif
}



//##########################################################################################################################################################################################




/*Compute back propagation using given device dev.*/
void FULL::backward(DEVICE& device) {

#ifdef DEBUG3
    printf("Entering backward@%s\n", get_id().c_str());
#endif

    /*Find index of the device*/
    int idx_device = -1;
    for (int i = 0; i < env.devices.size(); i++) {
        if (env.devices[i]->id == device.id) {
            idx_device = i;
            break;
        }
    }

#ifdef DEBUG0
    if (idx_device < 0) {
        puts("FULL::backward: cannot find a matching device.");
        exit(-1);
    }
#endif

    /*B(Y,dEdB): move Y back into device, and allocate a new dEdB memory object on the device.*/
    cl_int status;
    cl_event event, event_dEdB, event_dEdX, event_dEdW;
    Y->set_data2();
    Y->migrate(device);
    dEdB->set_data2();
    dEdB->migrate(device);
    
    /*C(dEdB)*/
    
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 0, sizeof (Y->data2), (void *) &Y->data2); 										/*Y*/
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 1, sizeof (dEdB->data2), (void *) &dEdB->data2); 									/*dEdB*/
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 2, sizeof (int), (void *) &Q);
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 3, sizeof (int), (void *) &_N);
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 4, sizeof (int), (void *) &_Q);
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 5, sizeof (y_offset->data2), (void *) &y_offset->data2);
    clSetKernelArg(ParamsdEdB[idx_device].kernel, 6, sizeof (y_stride->data2), (void *) &y_stride->data2);
    for(int ii = 0; ii<outputs.size();ii++)		//set valid output buffers.
    	clSetKernelArg(ParamsdEdB[idx_device].kernel, 7+ii, sizeof (outputs[ii]->data2), (void *) &outputs[ii]->data2);
    for(int ii = outputs.size(); ii<10;ii++)		//fill the rest arguments with dummy argument.
    	clSetKernelArg(ParamsdEdB[idx_device].kernel, 7+ii, sizeof (dEdB->data2), (void *) &dEdB->data2);
	
	/*Y->print_data2("Y[2]");
	outputs[0]->print_data2("dEdY");
	dEdB->print_data2("dEdB[1]");*/
	
#ifdef DEBUG4
	// print dEdY.
	char msg[100];
	for(int i = 0; i<outputs.size(); i++){
		sprintf(msg,"backward: dEdY[%d]@%s", i, get_id().c_str());
		outputs[i]->print_data2(msg);
	}
#endif
    
    status = clEnqueueNDRangeKernel(
            device.compute_q,
            ParamsdEdB[idx_device].kernel,
            2, //work dimension.
            NULL, //global work offset.
            ParamsdEdB[idx_device].global_work_size,
            ParamsdEdB[idx_device].local_work_size,
            backward_event_wait_list.size(),
            &backward_event_wait_list[0],
            &event_dEdB);
    if (status != CL_SUCCESS) {
        printf("FULL:backward(%s): unable to run the kernel for C(dEdB). %s.\n", get_id().c_str(), getErrorString(status));
        exit(-1);
    }
    
    
    /*(clFlush(device.compute_q);
    
    clWaitForEvents(1, &event_dEdB);
    
    dEdB->print_data2("dEdB[2]");
    
    exit(0);*/
    
    
    //clFlush(device.compute_q);
    
    /*B(input,dEdW)*/
    input->set_data2();
    input->migrate(device);
    dEdW->set_data2();
    dEdW->migrate(device);

    /*C(dEdW)*/
    clSetKernelArg(ParamsdEdW[idx_device].kernel, 0, sizeof (dEdB->data2), (void *) &dEdB->data2); 		/*dEdB*/
    clSetKernelArg(ParamsdEdW[idx_device].kernel, 1, sizeof (input->data2), (void *) &input->data2); 	/*input*/
    clSetKernelArg(ParamsdEdW[idx_device].kernel, 2, sizeof (dEdW->data2), (void *) &dEdW->data2); 		/*dEdW*/
    clSetKernelArg(ParamsdEdW[idx_device].kernel, 3, sizeof (int), (void *) &P);
    clSetKernelArg(ParamsdEdW[idx_device].kernel, 4, sizeof (int), (void *) &Q);
    clSetKernelArg(ParamsdEdW[idx_device].kernel, 5, sizeof (int), (void *) &N);
    status = clEnqueueNDRangeKernel(
            device.compute_q,
            ParamsdEdW[idx_device].kernel,
            2, //work dimension.
            NULL, //global work offset.
            ParamsdEdW[idx_device].global_work_size,
            ParamsdEdW[idx_device].local_work_size,
            1, //number of events in wait list.
            &event_dEdB, //event wait list.
            &event_dEdW);
    if (status != CL_SUCCESS) {
        printf("FULL:backward(%s): unable to run the kernel for C(dEdW). %s.\n", get_id().c_str(), getErrorString(status));
        exit(-1);
    }
    //clFlush(device.compute_q);
    
    
    /*B(W)*/
    W->set_data2();
    W->migrate(device);

    /*C(dEdX)*/
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 0, sizeof (dEdB->data2), (void *) &dEdB->data2); 									/*dEdB*/
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 1, sizeof (W->data2), (void *) &W->data2); 										/*W*/
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 2, sizeof (input->data2), (void *) &input->data2); 								/*dEdX: write to input.*/
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 3, sizeof (int), (void *) &P);
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 4, sizeof (int), (void *) &Q);
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 5, sizeof (int), (void *) &_N);
    clSetKernelArg(ParamsdEdX[idx_device].kernel, 6, sizeof (int), (void *) &_P);
    status = clEnqueueNDRangeKernel(
            device.compute_q,
            ParamsdEdX[idx_device].kernel,
            2,
            NULL,
            ParamsdEdX[idx_device].global_work_size,
            ParamsdEdX[idx_device].local_work_size,
            1,
            &event_dEdW,
            &event_dEdX);
    if (status != CL_SUCCESS) {
        printf("FULL:backward(%s): unable to run the kernel for C(dEdX). %s.\n", get_id().c_str(), getErrorString(status));
        exit(-1);
    }
    clFlush(device.compute_q);

#ifdef DEBUG3
    printf("%s is waiting for signal.\n", get_id().c_str());
#endif
    /*Wait for C(dEdX)*/
    clWaitForEvents(1, &event_dEdX);
#ifdef DEBUG4
	sprintf(msg,"backward: dEdX@%s", get_id().c_str());	
	input->print_data2(msg);
	sprintf(msg,"backward: dEdB@%s", get_id().c_str());	
	dEdB->print_data2(msg);
	sprintf(msg,"backward: dEdW@%s", get_id().c_str());	
	dEdW->print_data2(msg);
#endif

    /*Set reference count of input->data2, so that it will be released by predecessors in the backward pass once the reference count reaches 0.*/
    input->set_data2_ref_count(pred.size());

    /*Inform ready.*/
    signal_backward_ready();
    
    /*Read derivatives back from device*/
    dEdB->set_data1();
    dEdW->set_data1();
    
    /*Free memory*/
    Y->free_data2();
    dEdB->free_data2();
    dEdW->free_data2();
    W->free_data2();
    for(int i = 0; i<outputs.size(); i++)
    	outputs[i]->dec_data2_ref_count();
    
    /*Clear the backward event wait list set by successors.*/
    clear_backward_event_list();
    
    /*update parameters*/
    for(int i = 1; i<_N; i++)			// Here begins at 1 because it accumulates over _N directions, and accumulated results are written back into the first row.
    	for(int j = 0; j<_Q; j++)
    		dEdB->data1[j] += dEdB->data1[i*Q+j];
    //if(id =="Full_1"){
	sol->update(B->data1, dEdB->data1);
	sol->update(W->data1, dEdW->data1);
	sol->inc_iter();
	//}
    /*clean up*/
    clReleaseEvent(event_dEdB);
    clReleaseEvent(event_dEdX);
    clReleaseEvent(event_dEdW);

#ifdef DEBUG3
    printf("Leaving backward@%s\n", get_id().c_str());
#endif
}



//##########################################################################################################################################################################################




/*Initialize the unit:
        1) Get the optimal parameter settings for each device, based on size of the problem.
        2) Get the running time information for the unit on each device.
        3) Get the resource requirement for the problem.
        4) Create OpenCL program and kernel for each device.
 */
void FULL::init() {

	if(succ.size() != offsets.size() || succ.size() != outputs.size()){
		printf("Inconsistent dimensions: succ.size() = %ld, offsets.size() = %ld, outputs.size() = %ld.\n",succ.size(), offsets.size(), outputs.size());
		exit(-1);
	}

    _P = (input->h)*(input->w)*(input->d);
    P = roundup(_P);
    _Q = d;		//output width.
    Q = roundup(_Q);
    _N = input->n;
    N = roundup(_N);
    

    /*Initialize W, B, dEdW, dEdB, Y*/
    W = new MATRIX<float>(env,"W@"+get_id(),_P, _Q, P, Q); /*PxQ matrix*/
    dEdW = new MATRIX<float>(env,"dEdW@"+get_id(),_P, _Q, P, Q);
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0,0.01);		//Gaussian distribution: N(u,s)
    for (int i = 0; i < W->h; i++)
        for (int j = 0; j < W->w; j++)
            W->data1[i * W->W + j] = distribution(generator);
    B = new MATRIX<float>(env,"B@"+get_id(),1, _Q, 1, Q); //initialize B elements as 0.
   	//for(int i = 0; i< B->w; i++)
   	//	B->data1[i] = 0.01 * ((2.0 * rand()) / RAND_MAX - 1.0);
    dEdB = new MATRIX<float>(env,"dEdB@"+get_id(),_N, _Q, N, Q); //gradients w.r.t. each data point.
    Y = new MATRIX<float>(env,"Y@"+get_id(),_N, _Q, N, Q);


    /*Move offsets and strides of outputs into pinned memroy (persistent).*/
    y_offset = new MATRIX<int>(env,"y_offset@"+get_id(),1, offsets.size(), 1, offsets.size());		//offset vector.
    y_stride = new MATRIX<int>(env,"y_stride@"+get_id(),1, offsets.size(), 1, offsets.size());		//stride vector.
    for(int i = 0; i<offsets.size();i++){
    	y_offset->data1[i] = offsets[i];
    	y_stride->data1[i] = outputs[i]->d_stride;
    }
    y_offset->set_data2();		//move into pinned memory area.
    y_stride->set_data2();		//move into pinned memory area.


	string h_forward, h_dEdB, h_dEdX, h_dEdW; //headers.
	
    /*Parameter exploration dEdB*/
    ParamsdEdB = new DEV [env.devices.size()];
    for (int i = 0; i < env.devices.size(); i++) {
        if (get_optimal_parameters_dEdB(*env.devices[i], ParamsdEdB[i].rtsm, ParamsdEdB[i].rtsn, ParamsdEdB[i].kt)) {
#ifdef DEBUG2
            printf("FULL(%s)::init(dEdB): P = %d, Q = %d, N = %d, rtsm = %d, rtsn = %d, kernel_time = %.6f\n",get_id().c_str(), P, Q, N, ParamsdEdB[i].rtsm, ParamsdEdB[i].rtsn, ParamsdEdB[i].kt);
#endif
        } else {
            puts("FULL::init: Failed to get optimal parameters dEdB.");
            exit(-1);
        }
        ParamsdEdB[i].global_work_size[0] = Q;
        ParamsdEdB[i].global_work_size[1] = N;
        ParamsdEdB[i].local_work_size[0] = ParamsdEdB[i].rtsn;
        ParamsdEdB[i].local_work_size[1] = ParamsdEdB[i].rtsm;
        h_dEdB = generate_header_dEdB(*env.devices[i], ParamsdEdB[i].rtsm, ParamsdEdB[i].rtsn);
    }

    /*Parameter exploration dEdW*/
    ParamsdEdW = new DEV [env.devices.size()];
    for (int i = 0; i < env.devices.size(); i++) {
        if (get_optimal_parameters_dEdW(*env.devices[i], ParamsdEdW[i].tsk, ParamsdEdW[i].tsm, ParamsdEdW[i].tsn, ParamsdEdW[i].wptm, ParamsdEdW[i].wptn, ParamsdEdW[i].width, ParamsdEdW[i].kt, ParamsdEdW[i].gflops)) {
#ifdef DEBUG2
            printf("FULL(%s)::init(dEdW): P = %d, Q = %d, N = %d, tsk = %d, tsm = %d, tsn = %d, wptm = %d, wptn = %d, kernel_time = %.6f, gflops = %.6f\n", get_id().c_str(), P, Q, N, ParamsdEdW[i].tsk, ParamsdEdW[i].tsm, ParamsdEdW[i].tsn, ParamsdEdW[i].wptm, ParamsdEdW[i].wptn, ParamsdEdW[i].kt, ParamsdEdW[i].gflops);
#endif
        } else {
            puts("FULL::init: Failed to get optimal parameters dEdW.");
            exit(-1);
        }

        ParamsdEdW[i].global_work_size[0] = Q / ParamsdEdW[i].wptm;
        ParamsdEdW[i].global_work_size[1] = P / ParamsdEdW[i].wptn;
        ParamsdEdW[i].local_work_size[0] = ParamsdEdW[i].tsm / ParamsdEdW[i].wptm;
        ParamsdEdW[i].local_work_size[1] = ParamsdEdW[i].tsn / ParamsdEdW[i].wptn;
        h_dEdW = generate_header_dEdW(ParamsdEdW[i].tsm, ParamsdEdW[i].tsn, ParamsdEdW[i].tsk, ParamsdEdW[i].wptm, ParamsdEdW[i].wptn, ParamsdEdW[i].width);
    }

    /*Parameter exploration forward pass*/
    Paramsforward = new DEV [env.devices.size()];
    for (int i = 0; i < env.devices.size(); i++) {
        if (get_optimal_parameters_forward(*env.devices[i], Paramsforward[i].tsk, Paramsforward[i].tsm, Paramsforward[i].tsn, Paramsforward[i].wptm, Paramsforward[i].wptn, Paramsforward[i].width, Paramsforward[i].kt, Paramsforward[i].gflops)) {
#ifdef DEBUG2
            printf("FULL(%s)::init(forward): P = %d, Q = %d, N = %d, tsk = %d, tsm = %d, tsn = %d, wptm = %d, wptn = %d, kernel_time = %.6f, gflops = %.6f\n",get_id().c_str(), P, Q, N, Paramsforward[i].tsk, Paramsforward[i].tsm, Paramsforward[i].tsn, Paramsforward[i].wptm, Paramsforward[i].wptn, Paramsforward[i].kt, Paramsforward[i].gflops);
#endif
        } else {
            puts("FULL::init: Failed to get optimal parameters forward.");
            exit(-1);
        }
        Paramsforward[i].global_work_size[0] = Q / Paramsforward[i].wptn;
        Paramsforward[i].global_work_size[1] = N / Paramsforward[i].wptm;
        Paramsforward[i].local_work_size[0] = Paramsforward[i].tsn / Paramsforward[i].wptn;
        Paramsforward[i].local_work_size[1] = Paramsforward[i].tsm / Paramsforward[i].wptm;
        h_forward = generate_header_forward(Paramsforward[i].tsm, Paramsforward[i].tsn, Paramsforward[i].tsk, Paramsforward[i].wptm, Paramsforward[i].wptn, Paramsforward[i].width, activation);
    }

    /*Parameter exploration dEdX*/
    ParamsdEdX = new DEV [env.devices.size()];
    for (int i = 0; i < env.devices.size(); i++) {
        if (get_optimal_parameters_dEdX(*env.devices[i], ParamsdEdX[i].tsk, ParamsdEdX[i].tsm, ParamsdEdX[i].tsn, ParamsdEdX[i].wptm, ParamsdEdX[i].wptn, ParamsdEdX[i].width, ParamsdEdX[i].kt, ParamsdEdX[i].gflops)) {
#ifdef DEBUG2
            printf("FULL(%s)::init(dEdX): P = %d, Q = %d, N = %d, tsk = %d, tsm = %d, tsn = %d, wptm = %d, wptn = %d, kernel_time = %.6f, gflops = %.6f\n", get_id().c_str(),P, Q, N, ParamsdEdX[i].tsk, ParamsdEdX[i].tsm, ParamsdEdX[i].tsn, ParamsdEdX[i].wptm, ParamsdEdX[i].wptn, ParamsdEdX[i].kt, ParamsdEdX[i].gflops);
#endif
        } else {
            puts("FULL::init: Failed to get optimal parameters dEdX.");
            exit(-1);
        }
        ParamsdEdX[i].global_work_size[0] = P / ParamsdEdX[i].wptn;
        ParamsdEdX[i].global_work_size[1] = N / ParamsdEdX[i].wptm;
        ParamsdEdX[i].local_work_size[0] = ParamsdEdX[i].tsn / ParamsdEdX[i].wptn;
        ParamsdEdX[i].local_work_size[1] = ParamsdEdX[i].tsm / ParamsdEdX[i].wptm;
        h_dEdX = generate_header_dEdX(ParamsdEdX[i].tsm, ParamsdEdX[i].tsn, ParamsdEdX[i].tsk, ParamsdEdX[i].wptm, ParamsdEdX[i].wptn, ParamsdEdX[i].width);
    }


    /*Create OpenCL kernel for each device*/
    create_kernel("dEdB", h_dEdB);
    create_kernel("dEdW", h_dEdW);
    create_kernel("dEdX", h_dEdX);
    create_kernel("forward", h_forward);

    /*Create solver*/
    sol = new SOLVER(env.sol_param);
    sol->enroll_param(B->data1,  1, _Q, Q);
    sol->enroll_param(W->data1, _P, _Q, Q);
#ifdef DEBUG2
    printf("*%s initialization done.\tP = %d.\tQ = %d.\tN = %d.\t", id.c_str(), P, Q, N);
    for (int i = 0; i < env.devices.size(); i++) {
        printf("gflops(%s) = %.2f.\t", env.devices[i]->name, Paramsforward[i].gflops);
    }
    puts("");
#endif
}




