#include "main.hpp"
#include "kernels.hpp"


using namespace std;

FMATRIX generate_mat(int nrow, int ncol){
	FMATRIX A;
	A.data = new float [nrow*ncol];
	A.h = nrow;
	A.stride = A.w = ncol;
	return A;
}

int main(){
	CL_ENV cl;
	if(cl.init()<0) return -1;
	/*Test MAT_MUL*/
	MAT_MUL mm;
	K_PARAM A, B, C;
	A.mat = generate_mat(128,1024);
	A.inout = 1;
	A.env = &cl;
	A.device = cl.devices[0];
	B.mat = generate_mat(1024,64);
	B.inout = 1;
	C.mat = generate_mat(128,64);
	C.inout = 2;
	mm.run(&A,&B,&C);
	return 0;
}
