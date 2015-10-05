#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

__kernel void forward( __global float* X,				/*N-by-D matrix, each row is one sample. D is the number of classes for prediction.*/
					   const __global int*  T,			/*N-by-1 matrix, T[n] is the index of true class label, e.g. 0, 1, ..., D-1.*/
                       __global float* E,				/*N-by-1 matrix, the negative log-likelihood value*/
                       const int D,
                       const int X_height,				/*height of valid data region of X*/
                       const int X_width,				/*width of valid data region of X*/
                       const int X_offset )				/*Global offset in the pinned memory area of X*/
{

    const int tidm = get_local_id(1); // Local y-dim ID
    const int tidn = get_local_id(0); // Local x-dim ID
    const int offsetY = RTSM*get_group_id(1); // Work-group offset
    const int gy = get_group_id(1);
    __local double SharedSumX[RTSM][RTSN];
	double SumX, MeanX, x; 				//sum for each row of exponential values.
    const int numTiles = D/RTSN;
    
    int i, j, row, col;
    
    
    //---------------------Calculate the Mean of each row of X------------------------
    
    /*calculate SumX*/
    SumX = 0;
	#pragma unroll
	for( i = 0;i<numTiles;i++){
		if(i*RTSN + tidn<X_width && offsetY + tidm< X_height){
			SumX += X[X_offset+(offsetY + tidm)*D + i*RTSN + tidn];
		}
	}
	
    /*Expose SumX to local memory*/
    SharedSumX[tidm][tidn] = SumX;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*Calculate the private copy of row sum*/
    SumX = 0;
    if(tidn==0){
		#pragma unroll
		for(i=0;i<RTSN;i++){
			SumX += SharedSumX[tidm][i];
		}
		SharedSumX[tidm][0] = SumX;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    MeanX = SharedSumX[tidm][0]/X_width;		//the mean of each row.
    
    
    //---------------------------------------------
    
    SumX = 0;
    /*Exp(X) and calculate SumX*/
	#pragma unroll
	for( i = 0;i<numTiles;i++){
		if(i*RTSN + tidn<X_width && offsetY + tidm< X_height){
			x = X[X_offset+(offsetY + tidm)*D + i*RTSN + tidn];
			x = exp(x-MeanX);		//normalize x to prevent overflow.
			X[X_offset+(offsetY + tidm)*D + i*RTSN + tidn] = x;
			SumX += x;
		}
	}
	
    /*Expose SumX to local memory*/
    SharedSumX[tidm][tidn] = SumX;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*Calculate the private copy of row sum*/
    SumX = 0;
    if(tidn==0){
		#pragma unroll
		for(i=0;i<RTSN;i++){
			SumX += SharedSumX[tidm][i];
		}
		SharedSumX[tidm][0] = SumX;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*P(X) and calculate dEdX*/
    #pragma unroll
    for( i = 0;i<numTiles;i++){
    	if(offsetY + tidm<X_height){
			X[X_offset+(offsetY + tidm)*D + i*RTSN + tidn]  /= SharedSumX[tidm][0];  //dEdX.
		}
    }
    
    
    if(tidn == 0)
    	E[tidm+gy*RTSM] = 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    if(tidn == 0 && offsetY + tidm < X_height){
    	i = (offsetY + tidm)*D + T[tidm+gy*RTSM];
    	E[tidm+gy*RTSM] = -log(X[X_offset+i]);
    	X[X_offset+i] -= 1.0f;
    }
}





