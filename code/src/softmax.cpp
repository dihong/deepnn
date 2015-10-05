#include "softmax.hpp"
#include <vector>
#include "main.hpp"
#include "common.hpp"

using namespace std;

/*The softmax will be computed using CPU*/
void SOFTMAX::forward(DEVICE& device) {

#ifdef DEBUG3
    printf("Entering forward@%s\n", get_id().c_str());
#endif
	
	/*Wait for input ready*/
	clWaitForEvents(forward_event_wait_list.size(), &forward_event_wait_list[0]);


#ifdef DEBUG4
	input->print_data2("input[In]@"+get_id());
#endif

	
	/*Read from pinned memory into host memory*/
	input->set_data1();

	/*Calculate the forward pass*/
	{
		int i,j,k;
		double maxval;
		double sumval;
		float* pX = 0;
		float err = 0;
		for(i=0;i<n;i++){						//for each row.
			pX = input->data1 + i*input->d_stride;
			maxval = pX[0];
			for(j=1;j<input->d;j++)				//h=1 and w=1 for SOFTMAX.
				if (pX[j]>maxval)
					maxval = pX[j];
			sumval = 0;							//sum of row of exp(X).
			for(j=0;j<input->d;j++){
				pX[j] = exp(pX[j]-maxval);		//exponential of X.
				sumval += pX[j];
			}
			for(j=0;j<input->d;j++){					//probability distribution.
				pX[j] /= sumval;
			}
			#ifdef DEBUG
			for(j = 0; j<input->d; j++)
				err -= log(pX[batch_labels[i]]);
			#endif
			pX[batch_labels[i]] -= 1.0;
			//printf("%01d ",batch_labels[i]);
		}
		printf("%.8f\n",err);
	}
	/*Write output back to data2*/
	input->set_data2();

#ifdef DEBUG4
	input->print_data2("input[Out]@"+get_id());
#endif

    /*Free data1*/
    input->free_data1();
    
    /*Set reference count of input->data2, so that it will be released by predecessors in the backward pass once the reference count reaches 0.*/
    input->set_data2_ref_count(pred.size());
    
	/*Signal inputs ready*/
	signal_backward_ready();
	
    /*Clear the forward event wait list set by predecessors.*/
    clear_forward_event_list();
    
#ifdef DEBUG3
    printf("Leaving forward@%s\n", get_id().c_str());
#endif

}

void SOFTMAX::backward(DEVICE& device) {
    /*SOFTMAX unit has not backward pass*/
}


void SOFTMAX::init() {
	/*Set output dimension: inherit from input.*/
	n = input->n;
	d = input->d;
	w = input->w;
	h = input->h;
    if (h != 1 || input->w != 1) {
        printf("SOFTMAX::init@%s: input must be a vector. Expected input.h = 1, input.w = 1, got: input.h = %d, input.w = %d.\n", get_id().c_str(),input->h, input->w);
        exit(-1);
    }
    if (input->d < 1) {
        printf("SOFTMAX::init@%s: invalid dimension d = %d.\n",get_id().c_str(),input->d);
        exit(-1);
    }
    if (input->n < 1) {
        printf("SOFTMAX::init@%s: invalid batch size n = %d.\n",get_id().c_str(),input->n);
        exit(-1);
    }
    printf("*%s initialization done.\n",get_id().c_str());
}




