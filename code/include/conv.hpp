#ifndef _CONV_HPP_
#define _CONV_HPP_
#include "main.hpp"
#include "common.hpp"
#include <string>
#include <random>	//random gaussian numbers.
#include <algorithm>
#include <clFFT.h>


/*Class for a set of filters in the convolution unit*/
class FILTER:public MATRIX<float>{
public:
	//DDD
	int flag_data2_type;			// 0 if data2 is in input domain, 1 if data2 is in frequency domain (after fft).
	int fh;
	int fw;							// Actual width of filter
	int fH;							// Padded height of filter	(Equal to Padded height of feature maps, not including fft padding)
	int fW;							// Padded width of filter	(Equal to Padded width of feature maps, not including fft padding)
	int M;							// Number of input feature maps.
	int N;							// Number of output feature maps.
	int fft_padding;				// 1 for odd dimension, 2 for even dimension. Number of elements to pad in order to apply the FFT.
	std::map<std::pair<int, int>,int> mapper_filter;	//(i,j) -> k. where i is index of output filter, j is input filter, k is the index of the flatten filters (removing inactive filters).
	clfftPlanHandle planHandle;
	
	float* data3;					// filter data without padding: |mapper_filter|*(fh*fw) matrix, each row is one filter of size fh*fw.

	FILTER(const CL_ENV& _env, const std::string& _id, int _fh, int _fw, int _fH, int _fW, int _M, int _N):MATRIX<float>(_env, _id, 1, 1, 1, 1, 0){
		flag_data2_type = 0;
		fh = _fh;
		fw = _fw;
		fH = _fH;
		fft_padding = 2 - (_fW % 2);
		fW = _fW;	// The '_fW' includes only convolution-fft padding.
		M = _M;
		N = _N;
	}
	
	~FILTER(){
		clfftDestroyPlan( &planHandle );
		clfftTeardown();
	}
	
	void init(const std::vector< std:: vector<int> >& fm){
		/*Build a mapper from 2D filter index to 1D flatten filter index*/
		int cnt = 0;
		for(int i = 0; i< fm.size(); i++)
			for(int j = 0; j<fm[i].size(); j++)
				if(fm[i][j]==1)
					mapper_filter[std::pair<int,int>(i,j)] = cnt++;
		
		/*Reset actual size of data1 (after removing inactive filters)*/
		h = H = mapper_filter.size();	//height of MATRIX
		w = W = fH*(fW+fft_padding);	//width of MATRIX
		delete data1;
		data1 = new float [mapper_filter.size()*(H*W)];
		
		/*initialize data3*/
		data3 = new float [mapper_filter.size()*(fh*fw)];
		std::default_random_engine generator;
		std::normal_distribution<float> distribution(0,0.01);		//Gaussian distribution: N(u,s)
		for(int i = 0; i<mapper_filter.size()*(fh*fw); i++)
			data3[i] = distribution(generator);
			
			
		/*Pad the filters from data3 into data1*/
		pad();
		
		/*Setup FFT plan*/
		cl_int err;
		size_t clLengths[2] = {(size_t)fH, (size_t)fW};								// Size of the filter after convolution padding (not including fft padding).
		err = clfftCreateDefaultPlan(&planHandle, env.context, CLFFT_2D, clLengths);
		err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
		err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);	//Input and Output layout.
		err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
		err = clfftSetPlanBatchSize(planHandle, mapper_filter.size()); 				//Number of (active) filters to transform.
		size_t inStride[2] = {1,(size_t)fW+fft_padding};							//inStride[1] includes fft padding.
		size_t outStride[2] = {1,(size_t)(fW+fft_padding)/2};						//outStride[1] is measured in complex.
		err = clfftSetPlanInStride(planHandle, CLFFT_2D, inStride);
		err = clfftSetPlanOutStride(planHandle, CLFFT_2D, outStride);
		err = clfftSetPlanDistance(planHandle,fH*(fW+fft_padding), fH*(fW+fft_padding)/2);		//Set input and output distances between each filter (measured in real and complex, respectively).
		err = clfftBakePlan(planHandle, 1, &env.devices[0]->buffer_q, NULL, NULL);
	}
	
	/*Convert filters from data3 to data1@UNIT by padding and circulating.*/
	void pad(){
		int h_cir = floor(fh/2);		//numbler of elements to circulate in h dimension.
		int w_cir = floor(fw/2);
		memset(data1,0,sizeof(float)*mapper_filter.size()*(H*W));
		for(int i = 0; i < mapper_filter.size(); i++){
			/*write upper-left corner*/
			for(int j = 0; j<fh-h_cir;j++)
				for(int k = 0; k<fw-w_cir;k++)
					data1[i*W+j*(fW+fft_padding)+k] = data3[i*(fh*fw)+(j+h_cir)*fw+k+w_cir];
			/*write upper-right corner*/
			for(int j = 0; j<fh-h_cir;j++)
				for(int k = fW-w_cir; k<fW;k++)
					data1[i*W+j*(fW+fft_padding)+k] = data3[i*(fh*fw)+(j+h_cir)*fw+k-fW+w_cir];
			/*write bottom-left corner*/
			for(int j = fH-h_cir; j<fH;j++)
				for(int k = 0; k<fw-w_cir;k++)
					data1[i*W+j*(fW+fft_padding)+k] = data3[i*(fh*fw)+(j-fH+h_cir)*fw+k+w_cir];
			/*write bottom-right corner*/
			for(int j = fH-h_cir; j<fH;j++)
				for(int k = fW-w_cir; k<fW;k++){
					data1[i*W+j*(fW+fft_padding)+k] = data3[i*(fh*fw)+(j-fH+h_cir)*fw+k-fW+w_cir];
					//printf("data3[%d,%d,%d] = %.2f, fh = %d, fw = %d, h_cir = %d, w_cir = %d.\n",i,j-fH+h_cir,fW+w_cir,fh,fw,h_cir,w_cir);
				}
		}
	}
	
	/*Dump data3*/
	void print_data3(std::string varname){
		FILE* fp = fopen("dumps.txt", "a+");
		if (!fp) {
		    printf("FILTER::print_filter(%s): unable to open the file 'dumps.txt'.",id.c_str());
		    exit(-1);
		}
		fprintf(fp, "%s\n", varname.c_str());
		for (int i = 0; i < mapper_filter.size(); i++) {
			for (int j = 0; j < fh*fw; j++) {
				fprintf(fp, "%.4f ", data3[i * (fh*fw) + j]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
		fclose(fp);
	}
	
};


class CONV:public UNIT{
	SOLVER* sol;						/*Solver to update the model parameter*/
	std::vector< std:: vector<int> > filter_mask;			/*#outx#in matrix, filter_mask[i][j] is either 0 or 1. If it is 1, then the output[i] involves input[j], by filter (i,j); 0 if not involved.*/
	FILTER* filters;					/*Filter object contains (M*N)x(H*W) filters, where M is (#) input feature maps, and N is the (#) output feature maps. H and W are sizes of feature maps.*/
	MATRIX<float>* biases;				/*1xM bias vector, where M is the number of output feature maps.*/
	clfftPlanHandle planHandleInput, planHandleOutput;
	void create_kernel(std::string kernel_name, std::string header);
	cl_kernel kernel_dotplus, kernel_write_outputs;
	MATRIX<float>* local_output_buffer;	/*local_output_buffer buffer: It is a (sizeBatch*hMaps) x (numOutFeatMaps*wMaps) complex matrix (float2). Here wMaps = (fW + 2 - (fW%2))/2*/
	MATRIX<int>* indexPairs;			/*Pair (i,j) means filters[i] and inputFeatMaps[j] need to dot product.*/
	MATRIX<int>* rangePairs;			/*If pair = rangePairs[k], then dot products between indexPairs[pair.even] and indexPairs[pair.odd] go towards output k.*/
	MATRIX<int> *y_offset, *y_d_stride, *y_f_stride, *y_w_stride;
public:
	int num_input_maps;					/*Number of input feature maps*/
	int filter_height;					
	int filter_width;
	std::string normalization;			/*"None", "Krizhevsky"*/
	int step;							/*Step is the number of pixels between filters applied to an image. Must be 1 currently.*/
	int activation;						/*RELU or TANH*/

	CONV(const std::string& _id, const CL_ENV& _env):UNIT(_id,_env,UNIT_TYPE_CONV){
	}

	~CONV(){
		clfftDestroyPlan( &planHandleInput );
		clfftDestroyPlan( &planHandleOutput );
		clfftTeardown();
	}
	
	void init();
	
	void forward(DEVICE& device);
	
	void backward(DEVICE& device);
	
	void set_activation(int a){
		activation = a;
	}
	
};
#endif




