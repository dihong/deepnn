#include "input.hpp"
#include <fstream>
#include "SOIL.h"

using namespace std;

bool is_file_exist(const char *fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    size_t n = fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    n = fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);
    for(i = 0; i < size; i += 3)
    {
            unsigned char tmp = data[i];
            data[i] = data[i+2];
            data[i+2] = tmp;
    }
    return data;
}


void INPUT::forward(DEVICE& device) {
	int ww,hh,c;
	int DD = h*w*d;
	float* batch_data = new float [n*DD];
	memset(batch_data,0,sizeof(float)*n*DD);
	int* batch_label = new int [n];
	cl_int status;
	cl_event event;
	while(1){
		/*Fetch a batch of training samples from the hard drive*/
		for(int i = 0;i<n;i++){
			unsigned char* I = SOIL_load_image(sample_paths[(pos+i)%num_samples].c_str(),&ww,&hh,&c,SOIL_LOAD_AUTO);
			batch_label[i] = sample_labels[(pos+i)%num_samples];
			//printf("[%ld] %s\n",(pos+i)%num_samples,sample_paths[(pos+i)%num_samples].c_str());
			if(I == 0){
				printf("INPUT::forward: Error loading image %s.\n",sample_paths[(pos+i)%num_samples].c_str());
				exit(-1);
			}
			if(hh != h || ww != w){
				printf("INPUT::forward: Inconsistent dimensions. h = %d, hh = %d, w = %d, ww = %d.\n", h, hh, w, ww);
				exit(-1);
			}
			if(d == 3 && c == 3){
				for(int l = 0; l<c;l++){
					for(int j = 0;j<h;j++){
						for(int k = 0;k<w;k++){
							batch_data[i*DD+l*w+j*w*d+k] = I[j*w*c+k*c+l];
						}
					}
				}
			}
			else if(d == 3 && c == 1){
				//printf("INPUT::forward: Inconsistent dimensions. d = 3, c = 1.\n");
				//exit(0);
				for(int l = 0; l<c;l++){
					for(int j = 0;j<h;j++){
						for(int k = 0;k<w;k++){
							batch_data[i*DD+l*w+j*w*d+k] = I[j*w+k];
						}
					}
				}
			}
			else if(d==1 && c == 3){
				//printf("INPUT::forward: Inconsistent dimensions. d = 1, c = 3.\n");
				//exit(0);
				for(int j = 0;j<h;j++){
					for(int k = 0;k<w;k++){
						batch_data[i*DD+j*w+k] = ((float)I[j*w*3+k*3+0]+(float)I[j*w*3+k*3+1]+(float)I[j*w*3+k*3+2])/3.0f;
					}
				}
			}
			else if(d == 1 && c == 1){
				for(int j = 0;j<h;j++){
					for(int k = 0;k<w;k++){
						batch_data[i*DD+j*w+k] = I[j*w+k];
					}
				}
			}
			else{
				printf("INPUT::forward: Error dimension (%s): d = %d, c = %d. They must be either 1 (gray images) or 3 (rgb color images).\n",sample_paths[(pos+i)%num_samples].c_str(),d,c);
				exit(-1);
			}
			free(I);
		}

		/*Write into inputs of successor*/
		#ifdef DEBUG3
		printf("%s is waiting for signal.\n", get_id().c_str());
		#endif
		/*Normalize the data*/
		for(int i = 0; i<n*DD;i++)
			batch_data[i] /= 255.0f;
		/*Wait for the previous batch completes*/
		clWaitForEvents(backward_event_wait_list.size(),&backward_event_wait_list[0]);
		
		//DDD
		/*for(int i = 0; i<n; i++){
			for(int j = 0; j<h; j++){
				for(int k = 0; k<w; k++){
					batch_data[i*DD+j*w+k] = j;
				}
			}
		}*/
		
		
		/*Write all successors' input: Map sample-by-sample to avoid overlapping map for writing which can cause consistence problem. Write row-by-row of an image for potential padding (e.g. CONV)*/
		for(int k = 0; k<outputs.size(); k++){		/*For each registered successor*/
			/*Create input buffer if not exist (thread-safe)*/
			outputs[k]->create_input_buffer();
			for(int i = 0; i< n; i++){				/*For each sample*/
				if(outputs[k]->f_stride>0){		//successor is a 2D unit.
					for(int r = 0; r < h; r++){		//for each row of all [d] feature maps.
						/*Map a slice of one row (across all feature maps)*/
						float* y = (float*) clEnqueueMapBuffer( device.buffer_q,
																outputs[k]->data2,
																CL_TRUE,
																CL_MAP_WRITE,
																(offsets[k] + i*outputs[k]->d_stride+r*(outputs[k]->f_stride*d))* sizeof (float),
																(outputs[k]->f_stride*d)*sizeof(float),					//size of one slice.
																0,
																NULL,
																NULL,								//returned event.
																&status);
						if (status != CL_SUCCESS) {
							printf("INPUT::forward@%s: unable to map buffer for writting. %s.\n", get_id().c_str(), getErrorString(status));
							exit(-1);
						}
						
						/*Write multiple feature maps of [r] slice*/
						for(int j = 0; j < d; j++){
							memcpy(y + j*outputs[k]->f_stride,batch_data+i*DD+r*w*d+j*w,sizeof(float)*w);
						}
						
						/*Unmap buffer*/
						clEnqueueUnmapMemObject(device.buffer_q,
												outputs[k]->data2,
												y,
												0,
												0,
												&event);
						clFlush(device.buffer_q);
						clWaitForEvents(1, &event);
					}

				}
				else{		// 1D units: compactly write all feature maps (from one sample) directly. Note: All 1D units must store feature maps compactly.
					/*Map a region corresponding to a single sample (d_stride for each sample)*/
					float* y = (float*) clEnqueueMapBuffer( device.buffer_q,
															outputs[k]->data2,
															CL_TRUE,
															CL_MAP_WRITE,
															(offsets[k] + i*outputs[k]->d_stride)* sizeof (float),
															w*h*d*sizeof(float),					//write compactly one sample.
															0,
															NULL,
															NULL,									//returned event.
															&status);
					if (status != CL_SUCCESS) {
						printf("INPUT::forward@%s: unable to map buffer for writting. %s.\n", get_id().c_str(), getErrorString(status));
						exit(-1);
					}
					memcpy(y,batch_data+i*DD,sizeof(float)*w*h*d);  /*Write a valid row (no padding) of an image*/
					
					/*Unmap buffer*/
					clEnqueueUnmapMemObject(device.buffer_q,
											outputs[k]->data2,
											y,
											0,
											0,
											&event);
					clFlush(device.buffer_q);
					clWaitForEvents(1, &event);
				}
				
			}
		}
		
		/*Update current labels (global var)*/
		memcpy(labels_exposed_to_softmax,batch_label,sizeof(int)*n);
		/*Signal its successors that Y is ready*/
		signal_forward_ready();
		/*Clear the forward event wait list set by predecessors.*/
		clear_backward_event_list();
		
		#ifdef DEBUG3
		printf("%s done. Reading %d images (%d-%d).\n", get_id().c_str(), (int)n, int(pos+1), (int)(pos+n));
		#endif
		
		/*Move pointer forward*/		//DDD
		//pos = (pos + n) % num_samples;
	}
	delete batch_data;
}

void INPUT::init() {
    for(size_t i = 0; i < sample_paths.size(); i++){
    	if(!is_file_exist(sample_paths[i].c_str())){
    		printf("INPUT::init: The training data file '%s' cannot be opened for reading.\n",sample_paths[i].c_str());
    		exit(-1);
    	}
    }
    num_samples = sample_paths.size();
    labels_exposed_to_softmax = new int [n];
    printf("*%s initialization done.\tn = %d.\th = %d.\tw = %d. d = %d. num_samples = %ld\n", id.c_str(), n,h,w,d,num_samples);
	//forward(*env.devices[0]);
}



