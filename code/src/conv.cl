#define MUL_RE(a,b) (a.even*b.even - a.odd*b.odd)
#define MUL_IM(a,b) (a.even*b.odd + a.odd*b.even)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)


float2 complex_mul(float2 a,float2 b) { float2 x; x.even = MUL_RE(a,b); x.odd = MUL_IM(a,b); return x; }

/*Dot product and accumulate in complex domain*/
__kernel void dotplus (	const __global float2* filters,				/*filters in FFT domain. It is a (numFilters*hMaps) x wMaps matrix (float2)*/
						const __global float2* inputFeatMaps,		/*input feature maps in FFT domain. It is a (sizeBatch*hMaps) x (numInputFeatMaps*wMaps) matrix (float2)*/
						__global float2* outFeatMaps,				/*output feature maps store the dot product. It is a (sizeBatch*hMaps) x (numOutFeatMaps*wMaps) matrix (float2)*/
						const int _numInputFeatMaps,				/*number of input feature maps*/
						const int numOutFeatMaps,					/*number of output feature maps*/
						const int hMaps,							/*height of feature maps, after paddings*/
						const int _wMaps,							/*width of feature maps (by float2) after paddings. wMaps = (fW + 2 - (fW%2))/2*/
						const int sizeBatch,						/*number of samples in the batch*/
						const __global int2* indexPairs,			/*Pair (i,j) means filters[i] and inputFeatMaps[j] need to dot product.*/
						const __global int2* rangePairs				/*If pair = rangePairs[k], then dot products between indexPairs[pair.even] and indexPairs[pair.odd] go towards output k.*/
						)
/*
Local work group size is [16,16].
Global work size is output-size (rounded up to aligned with 16).
Each thread first identifies it self, and then fetch the corresponding pixels and apply dot product and accumulation.
The result of each thread is corresponding a pixel in output.
*/
{
	/*Cache global variables into private memory.*/
	const int numInputFeatMaps =  _numInputFeatMaps;
	const int wMaps = _wMaps;
	
	/*Locate range pair*/
    const int x_offset = get_global_id(0);
    const int y_offset = get_global_id(1);
    
    if (x_offset >= numOutFeatMaps*wMaps || y_offset >= sizeBatch*hMaps)
    	return;			//invalid pixel.
    
	const int2 rp = rangePairs[x_offset/wMaps];		// rangePairs is 1x#out vector.
	float2 acc = 0;									//accumualated complex sum of products.
	int2 index;
	const int wh = wMaps*hMaps;
	const int offset = (y_offset%hMaps)*wMaps+(x_offset%wMaps);		//within-map offset.
	
	for(int i = rp.even; i< rp.odd; i++){	// for each pair of index that has to calculate product.
		index = indexPairs[i];				// the pair of (filter,inputFeatMap) to do product.
		acc.even += MUL_RE(filters[index.even*wh+offset],inputFeatMaps[y_offset*wMaps*numInputFeatMaps+(index.odd/numInputFeatMaps)*wMaps + (x_offset%wMaps)]);
		acc.odd += MUL_IM(filters[index.even*wh+offset],inputFeatMaps[y_offset*wMaps*numInputFeatMaps+(index.odd/numInputFeatMaps)*wMaps + (x_offset%wMaps)]);
	}
	
	
	//(index.odd/numInputFeatMaps)*wh*numInputFeatMaps+(index.odd%numInputFeatMaps)*wMaps+offset
	outFeatMaps[y_offset*(numOutFeatMaps*wMaps) + x_offset] = acc;
}

__kernel void write_outputs(const __global float* inFeatMaps,		/*source output data from local buffer*/
							const int h,							/*height of actual size of feature maps, no padding.*/		
							const int w,							/*width of actual size of feature maps, no padding.*/
							const int d,							/*number of input feature maps for a single sample*/
							const int n,							/*number of samples in the batch*/
							const int hMaps,
							const w2Maps,							/*w2Maps = 2*wMaps, width of padded feature maps, including both convolution and fft paddings.*/
							const __global int* offset,				/*offset for each successor.*/
							const __global int* d_stride,			/*1xNUM_SUCC vector*/
							const __global int* f_stride,			/*1xNUM_SUCC vector*/ 			
							const __global int* w_stride,			/*1xNUM_SUCC vector*/
							const __global float* biases,			/*1xd bias vector. One for each input.*/
							__global float* Y0,
							__global float* Y1,
							__global float* Y2,
							__global float* Y3,
							__global float* Y4,
							__global float* Y5,
							__global float* Y6,
							__global float* Y7,
							__global float* Y8,
							__global float* Y9)
/*
Local work group size is [16,16].
Global work size is rounded up to aligned with 16.
Each thread first identifies it self, and fetch the corresponding pixels. Then add bias and apply activation function.
The result of each thread is corresponding a pixel in targeted buffers.
*/
{
	
	/*Get my index*/
    const int x_offset = get_global_id(0);
    const int y_offset = get_global_id(1);
    const int inMapIdX = x_offset/w2Maps;
    const int inMapIdY = y_offset/hMaps;
    const int inIdX = x_offset % w2Maps;
    const int inIdY = y_offset % hMaps;
    
    
    if(inMapIdX>=d || inMapIdY>=n || inIdX>=w || inIdY>=h)   //check invalid pixels.
    	return;
    	
    	
    const int inputPixelIndex = y_offset*w2Maps*d + x_offset;
    
    int targetPixelIndex = 0;
    
#if NUM_SUCC > 9
	if(f_stride[9]==0)		//target buffer is 1D.
		targetPixelIndex = offset[9] + inMapIdY*d_stride[9] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[9] + f_stride[9]*inMapIdX + offset[9] + inIdX + inIdY*w_stride[9];

#ifdef TANH
	Y9[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y9[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 8
	if(f_stride[8]==0)		//target buffer is 1D.
		targetPixelIndex = offset[8] + inMapIdY*d_stride[8] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[8] + f_stride[8]*inMapIdX + offset[8] + inIdX + inIdY*w_stride[8];

#ifdef TANH
	Y8[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y8[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 7	
	if(f_stride[7]==0)		//target buffer is 1D.
		targetPixelIndex = offset[7] + inMapIdY*d_stride[7] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[7] + f_stride[7]*inMapIdX + offset[7] + inIdX + inIdY*w_stride[7];

#ifdef TANH
	Y7[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y7[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 6	
	if(f_stride[6]==0)		//target buffer is 1D.
		targetPixelIndex = offset[6] + inMapIdY*d_stride[6] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[6] + f_stride[6]*inMapIdX + offset[6] + inIdX + inIdY*w_stride[6];

#ifdef TANH
	Y6[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y6[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 5	
	if(f_stride[5]==0)		//target buffer is 1D.
		targetPixelIndex = offset[5] + inMapIdY*d_stride[5] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[5] + f_stride[5]*inMapIdX + offset[5] + inIdX + inIdY*w_stride[5];

#ifdef TANH
	Y5[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y5[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 4	
	if(f_stride[4]==0)		//target buffer is 1D.
		targetPixelIndex = offset[4] + inMapIdY*d_stride[4] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[4] + f_stride[4]*inMapIdX + offset[4] + inIdX + inIdY*w_stride[4];

#ifdef TANH
	Y4[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y4[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 3	
	if(f_stride[3]==0)		//target buffer is 1D.
		targetPixelIndex = offset[3] + inMapIdY*d_stride[3] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[3] + f_stride[3]*inMapIdX + offset[3] + inIdX + inIdY*w_stride[3];

#ifdef TANH
	Y3[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y3[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 2	
	if(f_stride[2]==0)		//target buffer is 1D.
		targetPixelIndex = offset[2] + inMapIdY*d_stride[2] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[2] + f_stride[2]*inMapIdX + offset[2] + inIdX + inIdY*w_stride[2];

#ifdef TANH
	Y2[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y2[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 1	
	if(f_stride[1]==0)		//target buffer is 1D.
		targetPixelIndex = offset[1] + inMapIdY*d_stride[1] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[1] + f_stride[1]*inMapIdX + offset[1] + inIdX + inIdY*w_stride[1];
#ifdef TANH
	Y1[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y1[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif


#if NUM_SUCC > 0	
	if(f_stride[0]==0)		//target buffer is 1D.
		targetPixelIndex = offset[0] + inMapIdY*d_stride[0] + inMapIdX*h*w + inIdY*w + inIdX;
	else					//target buffer is 2D.
		targetPixelIndex = inMapIdY*d_stride[0] + f_stride[0]*inMapIdX + offset[0] + inIdX + inIdY*w_stride[0];
#ifdef TANH
	Y0[targetPixelIndex] = tanh(inFeatMaps[inputPixelIndex] + biases[inMapIdX]);		/*Tanh activation*/
#else
	Y0[targetPixelIndex] = MAX(inFeatMaps[inputPixelIndex] + biases[inMapIdX],0);		/*ReLU activation*/
#endif
#endif

}





