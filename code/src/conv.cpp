#include "conv.hpp"
#include <vector>
#include "main.hpp"
#include "common.hpp"

using namespace std;


/*Create kernels for all devices*/
void CONV::create_kernel(string kernel_name, string header="") {
    cl_int status;
    for (int i = 0; i < env.devices.size(); i++) {
        /*Create program object*/
        string source = readKernelFile("src/conv.cl");
        string code = header + source;
        const char* constCode = code.c_str();
        /*Build program*/
        cl_program program = clCreateProgramWithSource(env.context, 1, &constCode, 0, &status);
        if (status != CL_SUCCESS) {
            printf("CONV::create_kernel: unable to create program. %s.\n", getErrorString(status));
            exit(-1);
        }
        clBuildProgram(program, 1, &env.devices[i]->id, NULL, NULL, NULL);
        /*create kernel*/
        if (kernel_name == "dotplus")
            kernel_dotplus = clCreateKernel(program, "dotplus", &status);
        else if(kernel_name == "write_outputs")
        	kernel_write_outputs = clCreateKernel(program, "write_outputs", &status);

        if (status != CL_SUCCESS) {
            printf("CONV::create_kernel: unable to create kernel.  %s (%s).\n", getErrorString(status), kernel_name.c_str());
#ifdef DEBUG3
            puts(constCode);
#endif
            exit(-1);
        }
    }
}




void CONV::forward(DEVICE& device) {

	char msg[500];
#ifdef DEBUG3
    printf("Entering forward@%s\n", get_id().c_str());
#endif
	
	/*Wait for input ready*/
	clWaitForEvents(forward_event_wait_list.size(), &forward_event_wait_list[0]);

	
	cl_int status;
	cl_event event, event2;
	
	/*Apply forward FFT on filters*/
	filters->set_data2();		//move data2 into pinned memory area.
	filters->migrate(device);	//move pinned data2 into target device memory.
	status = clfftEnqueueTransform(filters->planHandle, CLFFT_FORWARD, 1, &device.compute_q, forward_event_wait_list.size(), &forward_event_wait_list[0], &event, &filters->data2, NULL, NULL);
	if (status != CL_SUCCESS){
		printf("CONV::forward: error occurs while calling function clfftEnqueueTransform for filter.\n");
		exit(0);
	}
	
	
#ifdef DEBUG4
	clWaitForEvents(1, &event);
	sprintf(msg,"forward: X [n = %d, #inputs = %d @ %dx%d]@%s",n,num_input_maps,input->h+filter_height/2, (input->w+filter_width/2 + 2 - (input->w+filter_width/2)%2),get_id().c_str());
	input->print_data2(msg);
	sprintf(msg,"forward: Filter [#filters = %d @ %dx%d]@%s",(int)filters->mapper_filter.size(),filters->fh,filters->fw,get_id().c_str());
	filters->print_data3(msg);
	sprintf(msg,"forward: FilterPad [%dx%d]@%s",filters->fH,filters->fW+filters->fft_padding,get_id().c_str());
	filters->print_data1(msg);
	sprintf(msg,"forward: FilterAfterFFT@%s",get_id().c_str());
	filters->print_data2(msg);
#endif

	
	/*Apply forward FFT on inputs*/
	input->migrate(device);
	status = clfftEnqueueTransform(planHandleInput, CLFFT_FORWARD, 1, &device.compute_q, 1, &event, &event2, &input->data2, NULL, NULL);
	if (status != CL_SUCCESS){
		printf("CONV::forward: error occurs while calling function clfftEnqueueTransform for input.\n");
		exit(0);
	}


#ifdef DEBUG4
	clWaitForEvents(1, &event2);
	sprintf(msg,"forward: InputAfterFFT@%s",get_id().c_str());
	input->print_data2(msg);
#endif

	
	/*Dot product and accumulate in frequence domain (filters->data2 .* input->data2).*/
	local_output_buffer->set_data2();
	local_output_buffer->migrate(device);
    clSetKernelArg(kernel_dotplus, 0, sizeof (filters->data2), (void *) &filters->data2);
    clSetKernelArg(kernel_dotplus, 1, sizeof (input->data2), (void *) &input->data2);
    clSetKernelArg(kernel_dotplus, 2, sizeof (local_output_buffer->data2), (void *) &local_output_buffer->data2);
    clSetKernelArg(kernel_dotplus, 3, sizeof (num_input_maps), (void *) &num_input_maps);
    clSetKernelArg(kernel_dotplus, 4, sizeof (d), (void *) &d);								//number of output maps.
    clSetKernelArg(kernel_dotplus, 5, sizeof (filters->fH), (void *) &filters->fH);			//height of complex feature maps.
    int width_complex_feature_maps = (filters->fW+filters->fft_padding)/2;					//width of complex feature maps.
    clSetKernelArg(kernel_dotplus, 6, sizeof (width_complex_feature_maps), (void *) &width_complex_feature_maps);						
    clSetKernelArg(kernel_dotplus, 7, sizeof (n), (void *) &n);								//size batch.
    clSetKernelArg(kernel_dotplus, 8, sizeof (indexPairs->data2), (void *) &indexPairs->data2);
    clSetKernelArg(kernel_dotplus, 9, sizeof (rangePairs->data2), (void *) &rangePairs->data2);
	size_t global_work_size[2];
	size_t local_work_size[2];
	global_work_size[0] = roundup(d*width_complex_feature_maps,16);
	global_work_size[1] = roundup(n*filters->fH,16);
	local_work_size[0] = 16;
	local_work_size[1] = 16;
    status = clEnqueueNDRangeKernel(
            device.compute_q,
            kernel_dotplus,
            2,		//work dimension.
            NULL,
            global_work_size,
            local_work_size,
            1,
            &event2,
            &event);
    if (status != CL_SUCCESS) {
        printf("CONV::forward: unable to run the kernel kernel_dotplus. %s.\n", getErrorString(status));
        exit(-1);
    }
    clFlush(device.compute_q);
#ifdef DEBUG4
	clWaitForEvents(1, &event);
	sprintf(msg,"forward: LocalOutputBufferBeforeFFT (n=%d*fH=%d)x(#outs=%d*wMaps=%d)@%s",n,filters->fH,d,width_complex_feature_maps,get_id().c_str());
	local_output_buffer->print_data2(msg);
#endif

	
	/*Apply inverse FFT on local output buffer*/
	status = clfftEnqueueTransform(planHandleOutput, CLFFT_BACKWARD, 1, &device.compute_q, 1, &event, &event2, &local_output_buffer->data2, NULL, NULL);
	if (status != CL_SUCCESS){
		printf("CONV::forward: error occurs while calling function clfftEnqueueTransform (reverse) for local_output_buffer.\n");
		exit(0);
	}

#ifdef DEBUG4
	clWaitForEvents(1, &event2);
	sprintf(msg,"forward: LocalOutputBufferAfterFFT@%s",get_id().c_str());
	local_output_buffer->print_data2(msg);
#endif


	
	/*Create input buffer of successors*/
	for(int i = 0; i<outputs.size(); i++){
		outputs[i]->create_input_buffer();
		outputs[i]->migrate(device);
	}
	
	/*Add biases to local output buffer and write into successors' input buffer*/
	biases->set_data2();			//biases need to move into pinned memory every time since it has been updating.
	biases->migrate(device);
    clSetKernelArg(kernel_write_outputs, 0, sizeof (local_output_buffer->data2), (void *) &local_output_buffer->data2);
    clSetKernelArg(kernel_write_outputs, 1, sizeof (h), (void *) &h);
    clSetKernelArg(kernel_write_outputs, 2, sizeof (w), (void *) &w);
    clSetKernelArg(kernel_write_outputs, 3, sizeof (d), (void *) &d);
    clSetKernelArg(kernel_write_outputs, 4, sizeof (n), (void *) &n);								//number of output maps.
    clSetKernelArg(kernel_write_outputs, 5, sizeof (filters->fH), (void *) &filters->fH);			//height of complex feature maps.
    int w2Maps = (filters->fW+filters->fft_padding);												//width of real ifft feature maps.
    clSetKernelArg(kernel_write_outputs, 6, sizeof (w2Maps), (void *) &w2Maps);		
    clSetKernelArg(kernel_write_outputs, 7, sizeof (y_offset->data2), (void *) &y_offset->data2);
    clSetKernelArg(kernel_write_outputs, 8, sizeof (y_d_stride->data2), (void *) &y_d_stride->data2);
    clSetKernelArg(kernel_write_outputs, 9, sizeof (y_f_stride->data2), (void *) &y_f_stride->data2);
    clSetKernelArg(kernel_write_outputs, 10, sizeof (y_w_stride->data2), (void *) &y_w_stride->data2);
    clSetKernelArg(kernel_write_outputs, 11, sizeof (biases->data2), (void *) &biases->data2);
    for(int i = 0; i<succ.size();i++)		//set valid output buffers.
    	clSetKernelArg(kernel_write_outputs, 12+i, sizeof (outputs[i]->data2), (void *) &outputs[i]->data2);
    for(int i = succ.size(); i<10;i++)		//fill the rest arguments with dummy argument.
    	clSetKernelArg(kernel_write_outputs, 12+i, sizeof (y_offset->data2), (void *) &y_offset->data2);
	global_work_size[0] = roundup(w2Maps*d,16);
	global_work_size[1] = roundup(filters->fH*n,16);
	local_work_size[0] = 16;
	local_work_size[1] = 16;
	
    status = clEnqueueNDRangeKernel(
            device.compute_q,
            kernel_write_outputs,
            2,		//work dimension.
            NULL,
            global_work_size,
            local_work_size,
            1,
            &event2,
            &event);
    if (status != CL_SUCCESS) {
        printf("CONV::forward: unable to run the kernel kernel_write_outputs. %s.\n", getErrorString(status));
        exit(-1);
    }
    clFlush(device.compute_q);
    
    clWaitForEvents(1, &event);		//wait write output buffer completes.
    
    
    
	/*Signal forward ready*/
	signal_forward_ready();

    /*Free memory*/
    filters->free_data2();
    input->free_data2();
    local_output_buffer->free_data2();
    
    
    /*Clear the forward event wait list set by predecessors.*/
    clear_forward_event_list();


    /*clean up*/
    clReleaseEvent(event);
    clReleaseEvent(event2);
    
        
#ifdef DEBUG3
    printf("Leaving forward@%s\n", get_id().c_str());
#endif

}

void CONV::backward(DEVICE& device) {

#ifdef DEBUG3
    printf("Entering backward@%s\n", get_id().c_str());
#endif

    /*Inform ready.*/
    signal_backward_ready();
}

void CONV::init() {

	

    /* Setup clFFT. */
    cl_int err;
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);
    if (err != CL_SUCCESS) {
    	printf("CONV::init: unable to initialize clFFT library. Error code is %d.\n",err);
    	exit(1);
    }
	/*Initialize filter_mask*/
	std::vector<int> rand_permutation;
	for(int i = 0; i<num_input_maps; i++){
		rand_permutation.push_back(i);
	}
	for(int i = 0; i<d; i++){
		std::vector<int> tmp;
		for(int j = 0; j<num_input_maps; j++){
			tmp.push_back(0);
		}
		std::random_shuffle ( rand_permutation.begin(), rand_permutation.end() );
		int num_positive_inputs = 1+floor(rand()/(0.5+RAND_MAX)*num_input_maps);
		for(int j = 0; j<num_positive_inputs; j++){
			tmp[rand_permutation[j]] = 1;
		}
		filter_mask.push_back(tmp);
	}

	/*Initialize the FILTER*/
	int h_padding = floor(filter_height/2);
	int w_padding = floor(filter_width/2);
	filters = new FILTER(env,"filters@"+id, filter_height, filter_width, h+h_padding, w+w_padding, num_input_maps, d);
	filters->init(filter_mask);
	
	/*Setup FFT plans*/
	int h_maps = h+h_padding;
	int w_maps = w+w_padding;
	size_t clLengths[2] = {(size_t)h_maps, (size_t)w_maps};					// Size of the feature maps after convolution padding (not including fft padding).
	//input
	err = clfftCreateDefaultPlan(&planHandleInput, env.context, CLFFT_2D, clLengths);
	err = clfftSetPlanPrecision(planHandleInput, CLFFT_SINGLE);
	err = clfftSetLayout(planHandleInput, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);				//Input and Output layout.
	err = clfftSetResultLocation(planHandleInput, CLFFT_INPLACE);
	err = clfftSetPlanBatchSize(planHandleInput, n*num_input_maps); 								//Number of input feature maps to transform.
	int fft_padding = 2 - (w_maps % 2);
	size_t inStride[2] = {1,(size_t)(w_maps+fft_padding)};									//inStride[1] includes fft padding.
	size_t outStride[2] = {1,(size_t)(w_maps+fft_padding)/2};								//outStride[1] is measured in complex.
	err = clfftSetPlanInStride(planHandleInput, CLFFT_2D, inStride);
	err = clfftSetPlanOutStride(planHandleInput, CLFFT_2D, outStride);
																							//Set input and output distances between each filter (measured in real and complex, respectively).
	err = clfftSetPlanDistance(planHandleInput,h_maps*(w_maps+fft_padding), h_maps*(w_maps+fft_padding)/2);		
	err = clfftBakePlan(planHandleInput, 1, &env.devices[0]->buffer_q, NULL, NULL);
	//output
	err = clfftCreateDefaultPlan(&planHandleOutput, env.context, CLFFT_2D, clLengths);
	err = clfftSetPlanPrecision(planHandleOutput, CLFFT_SINGLE);
	err = clfftSetLayout(planHandleOutput, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);		//Input and Output layout.
	err = clfftSetResultLocation(planHandleOutput, CLFFT_INPLACE);
	err = clfftSetPlanBatchSize(planHandleOutput, n*d); 									//Number of output feature maps to transform (batch_size*num_outputs).
	err = clfftSetPlanInStride(planHandleOutput, CLFFT_2D, outStride);						// inStride of output is equal to outStride of input.
	err = clfftSetPlanOutStride(planHandleOutput, CLFFT_2D, inStride);						// outStride of output is equal to inStride of output.
																							//Set input and output distances between each output map (measured in complex and real, respectively).
	err = clfftSetPlanDistance(planHandleOutput, h_maps*(w_maps+fft_padding)/2, h_maps*(w_maps+fft_padding));		
	err = clfftBakePlan(planHandleOutput, 1, &env.devices[0]->buffer_q, NULL, NULL);
	
	
	/*Initialize biases as zeros*/
	biases = new MATRIX<float> (env,"biases@"+get_id(),1, d, 1, d);
    
    /*Initialize output buffer*/
    local_output_buffer = new MATRIX<float> (env,"outputs@"+get_id(),n*filters->fH,d*(filters->fW+filters->fft_padding),n*filters->fH,d*(filters->fW+filters->fft_padding));

	/*Initialize indexPairs and rangePairs*/
	int begin = 0, end = 0;
	std::vector<int> _indexPairs, _rangePairs;
	for(int k = 0; k<filter_mask.size(); k++){										//for each output.
		for(int t = 0; t<filter_mask[k].size(); t++){								//for each input.
			if(filter_mask[k][t]==1){												//input[t] is linked to output[k] by filter (t,k).
				_indexPairs.push_back(filters->mapper_filter[pair<int,int>(k,t)]);	//push index of flatten filters.
				_indexPairs.push_back(t);											//push index of input feature maps.
				end++;
			}
		}
		if(end>begin){
			_rangePairs.push_back(begin);
			_rangePairs.push_back(end);
			begin = end;
		}
		else{
			printf("CONV::init@%s: The output feature maps [%d] is not associated to any input feature maps.",get_id().c_str(),k);
			exit(-1);
		}
	}
	indexPairs = new MATRIX<int> (env,"indexPairs@"+get_id(),1,_indexPairs.size(),1,_indexPairs.size(),&_indexPairs[0]);
	rangePairs = new MATRIX<int> (env,"rangePairs@"+get_id(),1,_rangePairs.size(),1,_rangePairs.size(),&_rangePairs[0]);
	indexPairs->set_data2();		// in pinned memory persistently.
	rangePairs->set_data2();		// in pinned memory persistently.
	
    /*Create solver*/
    sol = new SOLVER(env.sol_param);
    sol->enroll_param(filters->data3,  filters->H, filters->W, filters->W);
    
    /*Create OpenCL kernels*/
    create_kernel("dotplus");
    char header[200];
    if(activation==TANH)
    	sprintf(header,"#define NUM_SUCC %d\n#define TANH\n",(int)succ.size());
    else
    	sprintf(header,"#define NUM_SUCC %d\n",(int)succ.size());
    create_kernel("write_outputs",header);
 

	/*Create stride buffers (permant resident in pinned memory)*/
    y_offset = new MATRIX<int>(env,"y_offset@"+get_id(),1, offsets.size(), 1, offsets.size());			//offset vector.
    y_d_stride = new MATRIX<int>(env,"y_d_stride@"+get_id(),1, offsets.size(), 1, offsets.size());		//stride vector.
    y_f_stride = new MATRIX<int>(env,"y_f_stride@"+get_id(),1, offsets.size(), 1, offsets.size());		//stride vector.
    y_w_stride = new MATRIX<int>(env,"y_w_stride@"+get_id(),1, offsets.size(), 1, offsets.size());		//stride vector.
    for(int i = 0; i<offsets.size();i++){
    	y_offset->data1[i] = offsets[i];
    	y_d_stride->data1[i] = outputs[i]->d_stride;
    	y_f_stride->data1[i] = outputs[i]->f_stride;
    	y_w_stride->data1[i] = outputs[i]->w_stride;
    }
    y_offset->set_data2();			//move into pinned memory area.
    y_d_stride->set_data2();		//move into pinned memory area.
    y_f_stride->set_data2();		//move into pinned memory area.
    y_w_stride->set_data2();		//move into pinned memory area.
	
    printf("*%s initialization done. n_stride = %d, d_stride = %d, n = %d, h = %d, w = %d, num_input_maps = %d, num_output_maps = %d, num_active_filters = %d.\n",get_id().c_str(), input->n_stride, input->d_stride, n, input->h, input->w,num_input_maps, d, (int)filters->mapper_filter.size());
}



