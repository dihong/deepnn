#include "main.hpp"
#include "kernels.hpp"


using namespace std;

int main(){
	CL_ENV cl;
	if(cl.init()<0) return -1;
	
	return 0;
}
