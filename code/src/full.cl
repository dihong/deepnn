#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
#define MAX(a,b) ((a) > (b)) ? (a) : (b)


#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif

#ifdef FORWARD
/*Forward pass: Write into multiple output Y*/
__kernel void forward(const __global floatX* X, 	/*N-by-P matrix (padded), each row is one sample*/
        const __global floatX* W, 					/*P-by-Q matrix, P is number of input, Q is number of output*/
        const __global float* B, 					/*1-by-Q vector, the bias of each output*/
        const int P, 								/*Aligned width of X (padded with zeros)*/
        const int Q, 								/*Aligned width of W (padded with zeros)*/
        const int Y_height, 						/*Write valid data region only*/
        const int Y_width, 							/*Write valid data region only*/
        const __global int* Y_offset,				/*Offset for each output buffer.*/
        const __global int* Y_stride,				/*Stride for each output buffer*/
        __global float* Y,
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
{
    // Thread identifiers
    const int tidn = get_local_id(0); // Local row ID (max: TSN/WPTN == RTSM)
    const int tidm = get_local_id(1); // Local col ID (max: TSM/WPTM == RTSN)
    const int offsetX = TSN * get_group_id(0); // Work-group offset
    const int offsetY = TSM * get_group_id(1); // Work-group offset

    // Local memory to fit a tile of X and W
    __local float Asub[TSM][TSK];
    __local float Bsub[TSK][TSN];

    // Allocate register space
    float Breg;
    float Areg[WPTM];
    float acc[WPTN][WPTM];
    int wm, wn;
    float y;

    // Initialise the accumulation registers
#pragma unroll
    for (wn = 0; wn < WPTN; wn++) {
#pragma unroll
        for (wm = 0; wm < WPTM; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }

    // Loop over all tiles
    const int numTiles = P / TSK;

    floatX vec;

    int t = 0, k, row, col, la;

    do {
#pragma unroll
        for (la = 0; la < LPTA / WIDTH; la++) {
            row = MOD2(la * RTSN * RTSM + tidm * RTSN + tidn, TSK / WIDTH);
            col = DIV2(la * RTSN * RTSM + tidm * RTSN + tidn, TSK / WIDTH);
            vec = X[(offsetY + col)*(P / WIDTH) + TSK * t / WIDTH + row];
#if WIDTH == 1
            Asub[col][row] = vec;
#elif WIDTH == 2
            Asub[col][WIDTH * row + 0] = vec.x;
            Asub[col][WIDTH * row + 1] = vec.y;
#elif WIDTH == 4
            Asub[col][WIDTH * row + 0] = vec.x;
            Asub[col][WIDTH * row + 1] = vec.y;
            Asub[col][WIDTH * row + 2] = vec.z;
            Asub[col][WIDTH * row + 3] = vec.w;
#endif
        }

#pragma unroll
        for (la = 0; la < LPTB / WIDTH; la++) {
            row = MOD2(la * RTSN * RTSM + tidm * RTSN + tidn, TSN / WIDTH);
            col = DIV2(la * RTSN * RTSM + tidm * RTSN + tidn, TSN / WIDTH);
            vec = W[(TSK * t + col)*(Q / WIDTH) + offsetX / WIDTH + row];
#if WIDTH == 1
            Bsub[col][row] = vec;
#elif WIDTH == 2
            Bsub[col][WIDTH * row + 0] = vec.x;
            Bsub[col][WIDTH * row + 1] = vec.y;
#elif WIDTH == 4
            Bsub[col][WIDTH * row + 0] = vec.x;
            Bsub[col][WIDTH * row + 1] = vec.y;
            Bsub[col][WIDTH * row + 2] = vec.z;
            Bsub[col][WIDTH * row + 3] = vec.w;
#endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        // Loop over the values of a single tile
#pragma unroll
        for (k = 0; k < TSK; k++) {
#pragma unroll
            for (wm = 0; wm < WPTM; wm++) {
                Areg[wm] = Asub[tidm + wm * RTSM][k];
            }

#pragma unroll
            for (wn = 0; wn < WPTN; wn++) {
                Breg = Bsub[k][tidn + wn * RTSN];
#pragma unroll
                for (wm = 0; wm < WPTM; wm++) {
                    acc[wn][wm] += Breg * Areg[wm];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        t++;
    } while (t < numTiles);

	
#pragma unroll
    for (int wn = 0; wn < WPTN; wn++) {
        row = offsetX + tidn + wn*RTSN;
        Breg = B[row];
#pragma unroll
        for (int wm = 0; wm < WPTM; wm++) {
            col = offsetY + tidm + wm*RTSM;
            if (row < Y_width && col < Y_height) {
#ifdef TANH     
				y = tanh(acc[wn][wm] + Breg); 		/*Tanh activation*/
#else
				y = MAX(acc[wn][wm] + Breg, 0); 	/*ReLU activation*/
#endif

				Y[col * Q + row] = y;		//output for local.
				
#if NUM_SUCC > 9
				Y9[Y_offset[9] + col * Y_stride[9] + row] = y;
#endif
#if NUM_SUCC > 8
				Y8[Y_offset[8] + col * Y_stride[8] + row] = y;
#endif
#if NUM_SUCC > 7	
				Y7[Y_offset[7] + col * Y_stride[7] + row] = y;
#endif
#if NUM_SUCC > 6				
				Y6[Y_offset[6] + col * Y_stride[6] + row] = y;
#endif
#if NUM_SUCC > 5				
				Y5[Y_offset[5] + col * Y_stride[5] + row] = y;
#endif
#if NUM_SUCC > 4				
				Y4[Y_offset[4] + col * Y_stride[4] + row] = y;
#endif
#if NUM_SUCC > 3				
				Y3[Y_offset[3] + col * Y_stride[3] + row] = y;
#endif
#if NUM_SUCC > 2				
				Y2[Y_offset[2] + col * Y_stride[2] + row] = y;
#endif
#if NUM_SUCC > 1				
				Y1[Y_offset[1] + col * Y_stride[1] + row] = y;
#endif
#if NUM_SUCC > 0				
				Y0[Y_offset[0] + col * Y_stride[0] + row] = y;
#endif

			}
		}
	}
}
#endif




#ifdef DEDB

/* dEdB = dEdY.*Y (local_work_group_size = {RTSN,RTSM})*/
__kernel void dEdB(	const __global float* Y, 					/*N-by-Q matrix*/
        			__global float* dEdB, 						/*N-by-Q matrix*/
        			const int Q,								/*Padd width of Y*/
        			const int Y_height, 						/*Write valid data region only*/
        			const int Y_width, 							/*Write valid data region only*/
        			const __global int* Y_offset,				/*Offset for each output buffer.*/
        			const __global int* Y_stride,				/*Stride for each output buffer*/
        			const __global float* dEdY0,
        			const __global float* dEdY1,
       			 	const __global float* dEdY2,
        			const __global float* dEdY3,
        			const __global float* dEdY4,
        			const __global float* dEdY5,
        			const __global float* dEdY6,
        			const __global float* dEdY7,
        			const __global float* dEdY8,
        			const __global float* dEdY9)
{

    const int tidn = get_local_id(0);
    const int tidm = get_local_id(1);
    const int x = RTSN * get_group_id(0) + get_local_id(0);
    const int y = RTSM * get_group_id(1) + get_local_id(1);
    float Yq, dedb;

    if (x < Y_width && y < Y_height)
#ifdef TANH
    {
    	dedb = 0;
    	Yq = Y[y * Q + x];
#if NUM_SUCC > 9
		dedb += (1 - Yq * Yq) * dEdY9[Y_offset[9] + y * Y_stride[9] + x];
#endif
#if NUM_SUCC > 8
		dedb += (1 - Yq * Yq) * dEdY8[Y_offset[8] + y * Y_stride[8] + x];
#endif
#if NUM_SUCC > 7	
		dedb += (1 - Yq * Yq) * dEdY7[Y_offset[7] + y * Y_stride[7] + x];
#endif
#if NUM_SUCC > 6		
		dedb += (1 - Yq * Yq) * dEdY6[Y_offset[6] + y * Y_stride[6] + x];
#endif
#if NUM_SUCC > 5		
		dedb += (1 - Yq * Yq) * dEdY5[Y_offset[5] + y * Y_stride[5] + x];
#endif
#if NUM_SUCC > 4		
		dedb += (1 - Yq * Yq) * dEdY4[Y_offset[4] + y * Y_stride[4] + x];
#endif
#if NUM_SUCC > 3		
		dedb += (1 - Yq * Yq) * dEdY3[Y_offset[3] + y * Y_stride[3] + x];
#endif
#if NUM_SUCC > 2		
		dedb += (1 - Yq * Yq) * dEdY2[Y_offset[2] + y * Y_stride[2] + x];
#endif
#if NUM_SUCC > 1		
		dedb += (1 - Yq * Yq) * dEdY1[Y_offset[1] + y * Y_stride[1] + x];
#endif
#if NUM_SUCC > 0		
		dedb += (1 - Yq * Yq) * dEdY0[Y_offset[0] + y * Y_stride[0] + x];
#endif
		dEdB[y * Q + x] = dedb;
    }
#else
	{
		dedb = 0;
		Yq = Y[y * Q + x];
		if (Yq > 0){
#if NUM_SUCC > 9
			dedb += dEdY9[Y_offset[9] + y * Y_stride[9] + x];
#endif
#if NUM_SUCC > 8
			dedb += dEdY8[Y_offset[8] + y * Y_stride[8] + x];
#endif
#if NUM_SUCC > 7	
			dedb += dEdY7[Y_offset[7] + y * Y_stride[7] + x];
#endif
#if NUM_SUCC > 6			
			dedb += dEdY6[Y_offset[6] + y * Y_stride[6] + x];
#endif
#if NUM_SUCC > 5			
			dedb += dEdY5[Y_offset[5] + y * Y_stride[5] + x];
#endif
#if NUM_SUCC > 4			
			dedb += dEdY4[Y_offset[4] + y * Y_stride[4] + x];
#endif
#if NUM_SUCC > 3			
			dedb += dEdY3[Y_offset[3] + y * Y_stride[3] + x];
#endif
#if NUM_SUCC > 2			
			dedb += dEdY2[Y_offset[2] + y * Y_stride[2] + x];
#endif
#if NUM_SUCC > 1			
			dedb += dEdY1[Y_offset[1] + y * Y_stride[1] + x];
#endif
#if NUM_SUCC > 0			
			dedb += dEdY0[Y_offset[0] + y * Y_stride[0] + x];
#endif
			dEdB[y * Q + x] = dedb;
		}
		else
			dEdB[y * Q + x] = 0;
	}
#endif
}

#endif




#ifdef DEDX

/*dEdX = dEdB*W'*/
__kernel void dEdX(const __global floatX* dEdB, /*N-by-Q, padded*/
        const __global floatX* W, /*P-by-Q, padded*/
        __global float* dEdX, /*X_heightxX_width, not padded*/
        const int P,
        const int Q,
        const int X_height, /*Write valid data region only*/
        const int X_width  /*Write valid data region only*/
) {
    // Thread identifiers
    const int tidn = get_local_id(0); // Local row ID (max: TSN/WPTN == RTSM)
    const int tidm = get_local_id(1); // Local col ID (max: TSM/WPTM == RTSN)
    const int offsetX = TSN * get_group_id(0); // Work-group offset
    const int offsetY = TSM * get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSK][TSN];


    // Allocate register space
    float Breg;
    float Areg[WPTM];
    float acc[WPTN][WPTM];
    int wm, wn;

    // Initialise the accumulation registers
#pragma unroll
    for (wn = 0; wn < WPTN; wn++) {
#pragma unroll
        for (wm = 0; wm < WPTM; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }

    // Loop over all tiles
    const int numTiles = Q / TSK;

    floatX vec;

    int t = 0, k, row, col, la;

    do {
        // Load one tile of A and B into local memory
#pragma unroll
        for (la = 0; la < LPTA / WIDTH; la++) {
            row = MOD2(la * RTSN * RTSM + tidm * RTSN + tidn, TSK / WIDTH);
            col = DIV2(la * RTSN * RTSM + tidm * RTSN + tidn, TSK / WIDTH);

            // Load the values (wide vector load)
            vec = dEdB[(offsetY + col)*(Q / WIDTH) + TSK * t / WIDTH + row];

            // Store the loaded vectors into local memory
#if WIDTH == 1
            Asub[row][col] = vec;
#elif WIDTH == 2
            Asub[WIDTH * row + 0][col] = vec.x;
            Asub[WIDTH * row + 1][col] = vec.y;
#elif WIDTH == 4
            Asub[WIDTH * row + 0][col] = vec.x;
            Asub[WIDTH * row + 1][col] = vec.y;
            Asub[WIDTH * row + 2][col] = vec.z;
            Asub[WIDTH * row + 3][col] = vec.w;
#endif
        }

#pragma unroll
        for (la = 0; la < LPTB / WIDTH; la++) {
            row = MOD2(la * RTSN * RTSM + tidm * RTSN + tidn, TSK / WIDTH);
            col = DIV2(la * RTSN * RTSM + tidm * RTSN + tidn, TSK / WIDTH);

            // Load the values (wide vector load)
            vec = W[(offsetX + col)*(Q / WIDTH) + TSK * t / WIDTH + row];

            // Store the loaded vectors into local memory
#if WIDTH == 1
            Bsub[row][col] = vec;
#elif WIDTH == 2
            Bsub[WIDTH * row + 0][col] = vec.x;
            Bsub[col][WIDTH * row + 1][col] = vec.y;
#elif WIDTH == 4
            Bsub[WIDTH * row + 0][col] = vec.x;
            Bsub[WIDTH * row + 1][col] = vec.y;
            Bsub[WIDTH * row + 2][col] = vec.z;
            Bsub[WIDTH * row + 3][col] = vec.w;
#endif

        }

        // Synchronise to make sure the tile is loaded

        barrier(CLK_LOCAL_MEM_FENCE);
        // Loop over the values of a single tile
#pragma unroll
        for (k = 0; k < TSK; k++) {
            // Cache the values of Asub in registers
#pragma unroll
            for (wm = 0; wm < WPTM; wm++) {
                Areg[wm] = Asub[k][tidm + wm * RTSM];
            }

            // Perform the computation
#pragma unroll
            for (wn = 0; wn < WPTN; wn++) {
                Breg = Bsub[k][tidn + wn * RTSN];
#pragma unroll
                for (wm = 0; wm < WPTM; wm++) {
                    acc[wn][wm] += Breg * Areg[wm];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t < numTiles);
    // Store the final results in dEdX
#pragma unroll
    for (int wn = 0; wn < WPTN; wn++) {
        row = offsetX + tidn + wn*RTSN;
#pragma unroll
        for (int wm = 0; wm < WPTM; wm++) {
            col = offsetY + tidm + wm*RTSM;
            if (row < X_width && col < X_height)
                dEdX[col * P + row] = acc[wn][wm];
        }
    }
}

#endif


#ifdef DEDW

/*dEdW = X'dEdB*/
__kernel void dEdW(const __global floatX* dEdB, 		/*N-by-Q, padded*/
        const __global floatX* X, 						/*N-by-P, padded*/
        __global float* dEdW, 							/*P-by-Q, padded*/
        const int P,
        const int Q,
        const int N) {

    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetX = TSM * get_group_id(0); // Work-group offset
    const int offsetY = TSN * get_group_id(1); // Work-group offset

    __local float Asub[TSK][TSM];
    __local float Bsub[TSK][TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];
    int wm, wn;
    // Initialise the accumulation registers
#pragma unroll
    for (wm = 0; wm < WPTM; wm++) {
#pragma unroll
        for (wn = 0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
    const int numTiles = N / TSK;
    int t = 0, k, row, col, la;
    do {
        // Load one tile of dEdB and X into local memory
#pragma unroll
        for (la = 0; la < LPTA / WIDTH; la++) {
            row = MOD2(la * RTSN * RTSM + tidn * RTSM + tidm, TSM / WIDTH);
            col = DIV2(la * RTSN * RTSM + tidn * RTSM + tidm, TSM / WIDTH);

            // Load the values (wide vector load)
            floatX vecA = dEdB[(TSK * t + col)*(Q / WIDTH) + offsetX / WIDTH + row];

            // Store the loaded vectors into local memory
#if WIDTH == 1
            Asub[col][row] = vecA;
#elif WIDTH == 2
            Asub[col][WIDTH * row + 0] = vecA.x;
            Asub[col][WIDTH * row + 1] = vecA.y;
#elif WIDTH == 4
            Asub[col][WIDTH * row + 0] = vecA.x;
            Asub[col][WIDTH * row + 1] = vecA.y;
            Asub[col][WIDTH * row + 2] = vecA.z;
            Asub[col][WIDTH * row + 3] = vecA.w;
#endif

        }

#pragma unroll
        for (la = 0; la < LPTB / WIDTH; la++) {
            row = MOD2(la * RTSN * RTSM + tidn * RTSM + tidm, TSN / WIDTH);
            col = DIV2(la * RTSN * RTSM + tidn * RTSM + tidm, TSN / WIDTH);

            // Load the values (wide vector load)
            floatX vecB = X[(TSK * t + col)*(P / WIDTH) + offsetY / WIDTH + row];

            // Store the loaded vectors into local memory
#if WIDTH == 1
            Bsub[col][row] = vecB;
#elif WIDTH == 2
            Bsub[col][WIDTH * row + 0] = vecB.x;
            Bsub[col][WIDTH * row + 1] = vecB.y;
#elif WIDTH == 4
            Bsub[col][WIDTH * row + 0] = vecB.x;
            Bsub[col][WIDTH * row + 1] = vecB.y;
            Bsub[col][WIDTH * row + 2] = vecB.z;
            Bsub[col][WIDTH * row + 3] = vecB.w;
#endif
        }
        // Synchronise to make sure the tile is loaded

        barrier(CLK_LOCAL_MEM_FENCE);
        // Loop over the values of a single tile
#pragma unroll
        for (k = 0; k < TSK; k++) {
            // Cache the values of Bsub in registers
#pragma unroll
            for (wn = 0; wn < WPTN; wn++) {
                Breg[wn] = Bsub[k][tidn + wn * RTSN];
            }

            // Perform the computation
#pragma unroll
            for (wm = 0; wm < WPTM; wm++) {
                Areg = Asub[k][tidm + wm * RTSM];
#pragma unroll
                for (wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t < numTiles);

    // Store the final results in dEdW
#pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = offsetX + tidm + wm*RTSM;
#pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            int globalCol = offsetY + tidn + wn*RTSN;
            dEdW[globalCol * Q + globalRow] = acc[wm][wn];
        }
    }
}
#endif


#ifdef DMA_FORWARD_DEPRECIATED

/*DMA-tuned forward implementation. Currently depreciated.*/
__kernel void forward( 	const __global float* X,	/*N-by-P matrix, each row is one sample*/
						const __global float4* W,	/*P-by-Q matrix, P is number of input, Q is number of output*/
	                    const __global float*  B,	/*1-by-Q vector, the bias of each output*/
	                    __global float* Y,			/*Y_height-by-Y_width matrix, Y = tanh(XW+B)*/
	                    const int P,				/*Aligned width of A (padded with zeros)*/
	                    const int Q,				/*Aligned width of B (padded with zeros)*/
						const int X_offset,			/*Global offset in the pinned memory area of X*/
						const int Y_offset,			/*Global offset in the pinned memory area of Y*/
						const int Y_height,			/*Write valid data region only*/
						const int Y_width,			/*Write valid data region only*/
						const int Y_stride )		/*Write valid data region only*/
{

    // Thread identifiers
    const int tidn = get_local_id(0); // Local row ID (max: TSN/WPTN == RTSM)
    const int tidm = get_local_id(1); // Local col ID (max: TSM/WPTM == RTSN)
    const int offsetX = TSN*get_group_id(0); // Work-group offset
    const int offsetY = TSM*get_group_id(1); // Work-group offset
    
    // Local memory to fit a tile of X and W

    __local float Bsub[TSK][TSN];

    // Allocate register space
    float Breg;
    float acc[WPTN][WPTM];
	int wm, wn;
    
    // Initialise the accumulation registers
    #pragma unroll
    for (wn=0; wn<WPTN; wn++) {
        #pragma unroll
        for (wm=0; wm<WPTM; wm++) {
            acc[wn][wm] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = P/TSK;
    
    float4 vec;
    
    int t=0, k, row, col, la;
    
    do {
        
        #pragma unroll
        for (la=0; la<LPTB/4; la++) {
            row = MOD2(la*RTSN*RTSM + tidm*RTSN + tidn,TSN/4);
            col = DIV2(la*RTSN*RTSM + tidm*RTSN + tidn,TSN/4);
            vec = W[(TSK*t + col)*(Q/4) + offsetX/4 + row];
            Bsub[col][4*row + 0] = vec.x;
            Bsub[col][4*row + 1] = vec.y;
            Bsub[col][4*row + 2] = vec.z;
            Bsub[col][4*row + 3] = vec.w;
        }
       
		barrier(CLK_LOCAL_MEM_FENCE);
        // Loop over the values of a single tile
        #pragma unroll
        for (k=0; k<TSK; k++) {
			int tmp = X_offset+(offsetY + tidm)*P + TSK*t+k;
            #pragma unroll
            for (wn=0; wn<WPTN; wn++) {
                Breg = Bsub[k][tidn + wn*RTSN];
                #pragma unroll
                for (wm=0; wm<WPTM; wm++) {
                    acc[wn][wm] += Breg * X[wm*RTSM*P + tmp];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        t++;
    } while (t<numTiles);
    
    #pragma unroll
    for (int wn=0; wn<WPTN; wn++) {
        row = offsetX + tidn + wn*RTSN;
        Breg = B[row];
        #pragma unroll
        for (int wm=0; wm<WPTM; wm++) {
            col = offsetY + tidm + wm*RTSM;
			if(row<Y_width && col<Y_height){
            #ifdef TANH
            	Y[Y_offset+col*Y_stride + row] = tanh(acc[wn][wm]+Breg);	/*Tanh activation*/
            #else
            	Y[Y_offset+col*Y_stride + row] = MAX(acc[wn][wm]+Breg,0);	/*ReLU activation*/
            #endif
			}
        }
    }
}

#endif
