#include "main.hpp"
#include "common.hpp"
#include "full.hpp"
#include "softmax.hpp"
#include "graph.hpp"
#include "schedule.hpp"
#include "input.hpp"
#include <stdio.h>
#include <ctime>
#include <algorithm>

using namespace std;

int main(){
	/*clear the content in matrices.txt*/
	FILE* fp = fopen("matrices.txt","w+");
	fprintf(fp,"%s","");
	fclose(fp);
	/*clear the content in pmem.txt*/
	fp = fopen("dumps.txt","w+");
	fprintf(fp,"%s","");
	fclose(fp);
    /*Setup OpenCL environment*/
	CL_ENV env;
	env.init();

	/*Set random number generator seeds if not in debugging mode.*/
#ifndef DEBUG4
	std::srand ( unsigned ( std::time(0) ) );
#endif
	
	/*Build graph*/
	GRAPH g;
	if( !g.init(&env,"userdata/example.net","userdata/data.train","userdata/solver.conf") ){
		puts("Error creating graph.");
		return -1;
	}
	if( !g.build() ){
		puts("Error building graph.");
		return -1;
	}
	
	/*Initialize scheduler*/
	SCHEDULER* s = new SCHEDULER(g.get_units());
	s->update_schedule();
	
	/*Start all the INPUT units*/
	s->start_input_units();
	
	/*Run the units one by one*/
	COMPUTE_NODE* cn = s->get_next();
	for(int i = 0; i<50; i++){		/*For each batch*/
		while(cn){		/*For each computation node in a batch*/
			#ifdef DEBUG3
			printf("running unit %s[%d].\n", cn->unit->get_id().c_str(),cn->is_forward);
			#endif
			cn->run();
			#ifdef DEBUG3
			printf("unit %s[%d] done.\n", cn->unit->get_id().c_str(),cn->is_forward);
			#endif
			cn = s->get_next();
		}
		s->update_schedule();		/*update the running schedule*/
		cn = s->get_next();			/*get the first computation node*/
	}
	
	/*Kill all input threads for safe exit.*/
	exit(0);
    
	return 0;
}
