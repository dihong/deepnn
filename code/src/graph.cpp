#include "graph.hpp"
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include "full.hpp"
#include "softmax.hpp"
#include "graph.hpp"
#include "input.hpp"
#include "conv.hpp"
#include <map>


using namespace std;

// trim from start
static inline std::string &ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
	return ltrim(rtrim(s));
}

bool isEqual(const vector<string> & x, const vector<string> & y){
	if(x.size() != y.size())
		return false;
	for(int i = 0;i<x.size();i++){
		if(x[i] != y[i])
			return false;
	}
	return true;
}


bool split(const string instr, string& left, string &right){
	size_t found = instr.find_first_of(":");
	if(found!=string::npos && found > 0){
		left = instr.substr(0,found);
		right = instr.substr(found+1);
		/*remove comment*/
		if(instr.find_first_of("#")!=string::npos){
			right = instr.substr(found+1,instr.find_first_of("#")-found-1);
		}
		/*remove space*/
		left = trim(left);
		right = trim(right);
		return true;
	}else
		return false;
}


vector<string> split(const string instr, const string delimiter){
	string s = instr;
	/*remove comment*/
	if(s.find_first_of("#")!=string::npos){
		s = s.substr(0,s.find_first_of("#")-1);
	}
	size_t pos = 0;
	std::string token;
	vector<string> ret;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		ret.push_back(trim(token));
		s.erase(0, pos + delimiter.length());
	}
	token = s;
	token = trim(token);
	if(token.size()>0)
		ret.push_back(token);
	return ret;
}


/*Load the paths of training samples and labels of each samples. Return true if success. False otherwise.*/
bool load_data_spec(const string& path_to_data_spec_file, vector<string>& path_to_train_samples, vector<int>& labels, int& num_classes){
	FILE* fp = fopen(path_to_data_spec_file.c_str(),"r");
	char* line = new char [10000];
	if(!fp){
		printf("unanble to open the file: %s.\n",path_to_data_spec_file.c_str());
		exit(-1);
	}
	int L;	//integer label of sample: 0,1,...
	char* P = new char [10000];		//path of the sample.
	while( fgets (line , 10000 , fp) ){
		if(sscanf(line,"%d\t%s",&L,P) != 2){
			printf("load_data_spec: '%s' is not in correct format.\n",line);
			return false;
		}
		if(L<0){
			printf("load_data_spec: '%s' is not in correct format. label value must be nonnegative.\n",line);
			return false;
		}
		path_to_train_samples.push_back(P);
		labels.push_back(L);
	}
	fclose(fp);
	delete line;
	delete P;
	/*check labels*/
	num_classes = 0;
	for(int i = 0;i<labels.size();i++){
		if(num_classes<labels[i])
			num_classes = labels[i];
	}
	num_classes++;
	if(num_classes>1e6){
		printf("Error: invalid label value %d. The maximum label value must be less than 1e6.\n",num_classes);
		return false;
	}
	
	int* mentioned = new int [num_classes];
	memset(mentioned,0,sizeof(int)*num_classes);
	
	for(int i = 0;i<labels.size();i++){
		mentioned[labels[i]] = 1;
	}
	
	for(int i = 0;i<num_classes;i++){
		if(mentioned[i]==0){
			printf("Error: the label value '%d' was not used. We require all label values between [0,%d] must be used by at least one sample.\n",i,num_classes-1);
			return false;
		}
	}
	
	delete mentioned;
	return true;
}


bool GRAPH::init(CL_ENV* _env, std::string path_to_network_file, std::string path_to_train_spec, std::string path_to_solver_param){
	FILE* fp = fopen(path_to_network_file.c_str(),"r");
	char line [1000];
	if(!fp){
		printf("unanble to open the file: %s.\n",path_to_network_file.c_str());
		return false;
	}
	env = _env;
	vector<string> path_to_train_samples;
	vector<int> labels;
	if(!load_data_spec(path_to_train_spec, path_to_train_samples, labels, num_classes))
		return false;
	#ifdef DEBUG
	printf("Number of training samples: %ld\nNumber of classes: %d\n",labels.size(),num_classes);
	#endif
	int type = -1;
	string left, right, ID;
	map<UNIT*, vector<string> > successors;
	map<UNIT*, vector<string> > predecessors;
	map<string,UNIT*> mapper;		//unit (string) id to unit pointer mapper.
	size_batch = -1;
	while(1){
	
		/*read until a valid line*/
		/*while(fgets (line , sizeof(line) , fp)){
			string t = line;
			trim(t);
			if(t.size()>0)
				break;
		}
		string t = line;
		trim(t);
		if(t.size()==0) break;		//end of stream.
		puts(line);*/
		
		/*read unit type*/
		if( !fgets (line , sizeof(line) , fp) )
			break;
		if(!split(line,left,right)){
			printf("Graph::init: error parsing line '%s'.\n",line);
			return false;
		}
		if(left=="Type"){
			/*read ID*/
			if( !fgets (line , sizeof(line) , fp) || !split(line,left,ID) || left != "ID"){
				printf("Graph::init: error parsing line '%s'.\n",line);
				return false;
			}
			#ifdef DEBUG3
			puts("---------------------------------------");
			#endif
			if(right=="INPUT"){
				#ifdef DEBUG3
				printf("INPUT unit: %s\n",ID.c_str());
				#endif
				INPUT* u = new INPUT (ID, *env, path_to_train_samples, labels);
				units.push_back(u);
				mapper[ID] = u;   
				/*read Dimension*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Dimension"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if(sscanf(right.c_str(),"%d-%d-%d-%d",&u->n,&u->h,&u->w,&u->d) != 4 || u->n<1 || u->h<1 || u->w<1 || u->d<1){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if(size_batch<0)
					size_batch = u->n;
				else if(size_batch != u->n){
					printf("Graph::init: error parsing line '%s' The size of batch for different input units must be the same.\n",line);
					exit(-1);
				}
				#ifdef DEBUG3
				printf("n = %d, h = %d, w = %d, d = %d\n",u->n,u->h,u->w,u->d);
				#endif
				/*Read successors*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Successor"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				vector<string> succ = split(right,",");
				sort(succ.begin(), succ.end());
				successors[u] = succ;
				#ifdef DEBUG3
				printf("Successor(s): ");
				for(int i = 0; i<succ.size(); i++)
					printf("'%s' ",succ[i].c_str());
				puts("");
				#endif
			}
			else if(right == "FULL"){
				#ifdef DEBUG3
				printf("FULL unit: %s\n",ID.c_str());
				#endif
				FULL* u = new FULL (ID, *env);
				units.push_back(u);
				mapper[ID] = u;
				/*read Dimension*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "InDim" || right != "inherit"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "OutDim"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if(sscanf(right.c_str(),"%d-%d-%d-%d",&u->n,&u->h,&u->w,&u->d) != 4 || u->n<1 || u->h!=1 || u->w!=1 || u->d<1){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				u->set_Q(u->d);
				u->set_N(u->n);
				#ifdef DEBUG3
				printf("n = %d, h = %d, w = %d, d = %d\n",u->n,u->h,u->w,u->d);
				#endif
				/*read activation*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Activation"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if(right == "Tanh")
					u->set_activation(TANH);
				else if (right=="ReLU")
					u->set_activation(RELU);
				else{
					printf("Undefined activation type '%s' for unit %s. Expecting either Tanh or ReLU.\n",right.c_str(),ID.c_str());
					return false;
				}
				/*read Predecessors*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Predecessor"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				vector<string> pred = split(right,",");
				sort(pred.begin(), pred.end());
				predecessors[u] = pred;
				#ifdef DEBUG3
				printf("Predecessor(s): ");
				for(int i = 0; i<pred.size(); i++)
					printf("'%s' ",pred[i].c_str());
				puts("");
				#endif
				/*read successors*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Successor"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				vector<string> succ = split(right,",");
				sort(succ.begin(), succ.end());
				successors[u] = succ;
				#ifdef DEBUG3
				printf("Successor(s): ");
				for(int i = 0; i<succ.size(); i++)
					printf("'%s' ",succ[i].c_str());
				puts("");
				#endif
			}
			else if(right=="SOFTMAX"){
				#ifdef DEBUG3
				printf("SOFTMAX unit: %s\n",ID.c_str());
				#endif
				SOFTMAX* u = new SOFTMAX (ID, *env);
				units.push_back(u);
				mapper[ID] = u;
				/*read Dimension*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Dimension" || right != "inherit"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				/*read Predecessors*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Predecessor"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				vector<string> pred = split(right,",");
				sort(pred.begin(), pred.end());
				predecessors[u] = pred;
				#ifdef DEBUG3
				printf("Predecessor(s): ");
				for(int i = 0; i<pred.size(); i++)
					printf("'%s' ",pred[i].c_str());
				puts("");
				#endif
			}
			
			else if (right=="CONV"){
				#ifdef DEBUG3
				printf("CONV unit: %s\n",ID.c_str());
				#endif
				CONV* u = new CONV (ID, *env);
				units.push_back(u);
				mapper[ID] = u;
				/*read Dimension*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "InDim" || right != "inherit"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "OutDim"){
					printf("Graph::init: error parsing line '%s'. Expecting 'OutDim'.\n",line);
					return false;
				}
				if(sscanf(right.c_str(),"%d-%d-%d-%d",&u->n,&u->h,&u->w,&u->d) != 4 || u->n<1 || u->h<1 || u->w<1 || u->d<1){
					printf("Graph::init: error parsing line '%s'. Cannot read (n,h,w,d).\n",line);
					return false;
				}
				#ifdef DEBUG3
				printf("n = %d, h = %d, w = %d, d = %d\n",u->n,u->h,u->w,u->d);
				#endif
				/*read FilterHeight*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "FilterHeight"){
					printf("Graph::init: error parsing line '%s'. Expecting 'FilterHeight'.\n",line);
					return false;
				}
				if(sscanf(right.c_str(),"%d",&u->filter_height) != 1 || u->filter_height < 1){
					printf("Graph::init: error parsing line '%s'. Invalid FilterHeight = %d.\n",line, u->filter_height);
					return false;
				}
				/*read FilterWidth*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "FilterWidth"){
					printf("Graph::init: error parsing line '%s'. Expecting 'FilterWidth'.\n",line);
					return false;
				}
				if(sscanf(right.c_str(),"%d",&u->filter_width) != 1 || u->filter_width < 1){
					printf("Graph::init: error parsing line '%s'. Invalid FilterWidth = %d.\n",line, u->filter_width);
					return false;
				}
				/*read Step*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Step"){
					printf("Graph::init: error parsing line '%s'. Expecting 'Step'.\n",line);
					return false;
				}
				if(sscanf(right.c_str(),"%d",&u->step) != 1 || u->step != 1){
					printf("Graph::init: error parsing line '%s'. Invalid Step = %d. Step has to be 1.\n",line, u->step);
					return false;
				}
				
				/*read Normalization*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Normalization"){
					printf("Graph::init: error parsing line '%s'. Expecting 'Normalization'.\n",line);
					return false;
				}
				if (right == "None"){
					u->normalization = "None";
				}
				else if (right == "Krizhevsky"){
					u->normalization = "Krizhevsky";
				}
				else{
					printf("Graph::init: error parsing line '%s'. Undefined Normalization = %s.\n",line, right.c_str());
					return false;
				}
				
				/*read activation*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Activation"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				if(right == "Tanh")
					u->set_activation(TANH);
				else if (right=="ReLU")
					u->set_activation(RELU);
				else{
					printf("Undefined activation type '%s' for unit %s. Expecting either Tanh or ReLU.\n",right.c_str(),ID.c_str());
					return false;
				}
				
				/*read Predecessors*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Predecessor"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				vector<string> pred = split(right,",");
				sort(pred.begin(), pred.end());
				predecessors[u] = pred;
				#ifdef DEBUG3
				printf("Predecessor(s): ");
				for(int i = 0; i<pred.size(); i++)
					printf("'%s' ",pred[i].c_str());
				puts("");
				#endif
				/*read successors*/
				if( !fgets (line , sizeof(line) , fp) || !split(line,left,right) || left != "Successor"){
					printf("Graph::init: error parsing line '%s'.\n",line);
					return false;
				}
				vector<string> succ = split(right,",");
				sort(succ.begin(), succ.end());
				successors[u] = succ;
				#ifdef DEBUG3
				printf("Successor(s): ");
				for(int i = 0; i<succ.size(); i++)
					printf("'%s' ",succ[i].c_str());
				puts("");
				#endif
			}
			else{
				printf("Undefined unit type: %s\n",right.c_str());
				return false;
			}
		}else{
			printf("Expecting attribute: Type, but I got: %s.\n",left.c_str());
			return false;
		}
		if(!fgets (line , sizeof(line) , fp))  //remove one empty line.
			break;	
	}
	fclose(fp);
	/*Link predecessors and successors*/
	#ifdef DEBUG3
	puts("---------------------------------------");
	#endif
	map<UNIT*, vector<string> >:: iterator it;
	for(it = predecessors.begin(); it!= predecessors.end(); it++){
		UNIT* u = it->first;
		#ifdef DEBUG3
		printf("Predcessors of %s: ",u->get_id().c_str());
		#endif
		for(int i = 0;i<it->second.size();i++){
			if(mapper.find(it->second[i]) != mapper.end()){
				//Find the rank of this unit in its predecessors' output list.
				vector<string> output_list_of_pred = successors[mapper[it->second[i]]];
				int j = 0;
				for(j = 0;j<output_list_of_pred.size();j++){
					if(output_list_of_pred[j] == u->get_id()){
						break;
					}
				}
				if(j==output_list_of_pred.size()){
					printf("** GRAPH::init: Error in '%s'. The '%s' is not a valid predecessor. **\n", u->get_id().c_str(), it->second[i].c_str());
					printf("The successors of %s are: ",it->second[i].c_str());
					for(j = 0;j<output_list_of_pred.size();j++)
						printf("'%s' ",output_list_of_pred[j].c_str());
					puts("");
					return false;
				}
			
				u->add_pred(mapper[it->second[i]],j);
				
				#ifdef DEBUG3
				printf("'%s' ",mapper[it->second[i]]->get_id().c_str());
				#endif
			}
			else{
				printf("\n** undefined unit '%s' which is the predecessor of '%s' **\n",it->second[i].c_str(),u->get_id().c_str());
				return false;
			}
		}
		#ifdef DEBUG3
		puts("");
		#endif
	}
	for(it = successors.begin(); it!= successors.end(); it++){
		UNIT* u = it->first;
		#ifdef DEBUG3
		printf("Successors of %s: ",u->get_id().c_str());
		#endif
		for(int i = 0;i<it->second.size();i++){
			if(mapper.find(it->second[i]) != mapper.end()){
				//Find the rank of this unit in its successors' input list.
				vector<string> input_list_of_succ = predecessors[mapper[it->second[i]]];
				int j = 0;
				for(j = 0;j<input_list_of_succ.size();j++){
					if(input_list_of_succ[j] == u->get_id()){
						break;
					}
				}
				if(j==input_list_of_succ.size()){
					printf("** GRAPH::init: Error in '%s'. The '%s' is not a valid successor. **\n", u->get_id().c_str(), it->second[i].c_str());
					printf("The predecessors of %s are: ", it->second[i].c_str());
					for(j = 0;j<input_list_of_succ.size();j++)
						printf("%s\t",input_list_of_succ[j].c_str());
					puts("");
					return false;
				}
				u->add_succ(mapper[it->second[i]],j);
				#ifdef DEBUG3
				printf("'%s' ",mapper[it->second[i]]->get_id().c_str());
				#endif
			}
			else{
				printf("** undefined unit '%s' which is the successor of '%s' **\n",it->second[i].c_str(),u->get_id().c_str());
				return false;
			}
		}
		#ifdef DEBUG3
		puts("");
		#endif
	}
	#ifdef DEBUG3
	puts("---------------------------------------");
	#endif
	
	/*load solver parameters*/
	fp = fopen(path_to_solver_param.c_str(),"r");
	char* line2 = new char [10000];
	if(!fp){
		printf("unanble to open the file: %s.\n",path_to_solver_param.c_str());
		exit(-1);
	}
	SOLVER_PARAM* sp = new SOLVER_PARAM;
	while( fgets (line2 , 10000 , fp) ){
		
		if(!split(line2, left, right))
			continue;					//skip lines that are not containing ':'.
		
		if(left == "algorithm"){
			sscanf(right.c_str(),"%s",line2);
			if(string(line2) == "SGD")
				sp->algorithm = SOLVER_SGD;
			else if(string(line2) == "ADAGRAD")
				sp->algorithm = SOLVER_ADAGRAD;
			else if(string(line2) == "NESTEROV")
				sp->algorithm = SOLVER_NESTEROV;
			else{
				printf("Graph::init: unknown solver algorithm '%s'. The algorithm must be one of the 'SGD', 'ADAGRAD' and 'NESTEROV'.\n",line2);
				exit(-1);
			}
		}
		else if (left == "base_lr")
			sscanf(right.c_str(),"%f",&sp->base_lr);
		else if (left == "gamma")
			sscanf(right.c_str(),"%f",&sp->gamma);
		else if (left == "stepsize")
			sscanf(right.c_str(),"%d",&sp->stepsize);
		else if (left == "max_iter")
			sscanf(right.c_str(),"%d",&sp->max_iter);
		else if (left == "momentum")
			sscanf(right.c_str(),"%f",&sp->momentum);
		else{
			printf("Graph::init: unknown attribute '%s'.\n", left.c_str());
			exit(-1);
		}
	}
	fclose(fp);
	delete line2;

	//check parameters.
	if(sp->algorithm == SOLVER_NESTEROV){
		printf("Graph::init: sorry, but the NESTEROV solver algorithm is currently not supported.\n");
		exit(-1);
	}
	if (sp->algorithm == SOLVER_SGD || sp->algorithm == SOLVER_ADAGRAD || sp->algorithm == SOLVER_NESTEROV){
		if(sp->base_lr<=0 || sp->base_lr>=1.0){
			printf("Graph::init: You must specify a valid base_lr within (0,1) in %s.\n",path_to_solver_param.c_str());
			exit(0);
		}
		if(sp->gamma<=0 || sp->gamma>1.0){
			printf("Graph::init: You must specify a valid gamma within (0,1] in %s.\n",path_to_solver_param.c_str());
			exit(0);
		}
		if(sp->stepsize<=0 || sp->stepsize>1e9){
			printf("Graph::init: You must specify a valid stepsize within [1,1e9] in %s.\n",path_to_solver_param.c_str());
			exit(0);
		}
		if(sp->max_iter<=0 || sp->max_iter>1e9){
			printf("Graph::init: You must specify a valid max_iter within [1,1e9] in %s.\n",path_to_solver_param.c_str());
			exit(0);
		}
		if(sp->algorithm != SOLVER_ADAGRAD && (sp->momentum<=0 || sp->momentum>1)){
			printf("Graph::init: You must specify a valid momentum within [0,1] in %s.\n",path_to_solver_param.c_str());
			exit(0);
		}
	}
	else{
		printf("Graph::init: You must specify a valid algorithm in %s.\n",path_to_solver_param.c_str());
		exit(0);
	}
	
	env->sol_param = *sp;
	
	#ifdef DEBUG
	if (sp->algorithm == SOLVER_SGD)
		printf("algorithm = SGD\n");
	else if (sp->algorithm == SOLVER_ADAGRAD)
		printf("algorithm = ADAGRAD\n");
	else if (sp->algorithm == SOLVER_NESTEROV)
		printf("algorithm = NESTEROV\n");
	printf("base_lr = %.4f\n",sp->base_lr);
	printf("gamma = %.4f\n",sp->gamma);
	printf("stepsize = %d\n",sp->stepsize);
	printf("max_iter = %d\n",sp->max_iter);
	if(sp->algorithm != SOLVER_ADAGRAD)
		printf("momentum = %.4f\n",sp->momentum);
	puts("---------------------------------------");
	#endif
	
	delete sp;
	
	return true;
}



bool GRAPH::validate(){
	/*Check DAG*/
	
	/*Check parameter settings of each unit (including id uniqueness)*/
	
	/*Check connection consistency*/
	
	return true;
}


bool GRAPH::build(){
	/*Create PMEM 'input' for each non-input unit, and register each non-input unit to the output list (outputs,offsets) of predecessors*/
	for(int i = 0; i<units.size(); i++){
		
		if(units[i]->get_type()==UNIT_TYPE_INPUT){
			/*No action for INPUT unit*/
		}else if(units[i]->get_type()==UNIT_TYPE_FULL){
			units[i]->input = new PMEM(*env, units[i]->get_id());									/*input buffer of the unit. written by predecessors. actual memory is allocated/freed on the fly.*/
			units[i]->input->n = size_batch;
			units[i]->input->h = 1;																	/*FULL only accepts vector input. All non-vector input will be vectorized.*/
			units[i]->input->w = 1;																	/*FULL only accepts vector input. All non-vector input will be vectorized.*/
			units[i]->input->d = 0;																	/*Total number of elements of all predecessor elements. All non-vector input will be vectorized.*/
			const std::vector<UNIT*> pred = units[i]->get_pred();									/*A list of predecessors*/
			/*register itself to each of the predecessors.*/
			for(int j = 0; j<pred.size(); j++){
				pred[j]->outputs.push_back(units[i]->input);
				pred[j]->offsets.push_back(units[i]->input->d);
				units[i]->input->d += pred[j]->h*pred[j]->w*pred[j]->d;								/*Number of vectorized elements (actual size, no padding)*/
			}
			units[i]->input->w_stride = 0;															/*Vectorized unit has not w_stride*/
			units[i]->input->f_stride = 0;															/*FULL is 1D unit, so its f_stride is invalid.*/
			units[i]->input->d_stride = roundup(units[i]->input->d);								/*Round up for efficient matrix multiplication*/
			units[i]->input->n_stride = roundup(units[i]->input->n);								/*Round up for efficient matrix multiplication*/
			
		}else if(units[i]->get_type()==UNIT_TYPE_SOFTMAX){
			units[i]->input = new PMEM(*env, units[i]->get_id());									/*input buffer of the unit. written by predecessors. actual memory is allocated/freed on the fly.*/
			units[i]->input->n = size_batch;
			units[i]->input->h = 1;																	/*SOFTMAX only accepts vector input. All non-vector input will be vectorized.*/
			units[i]->input->w = 1;																	/*SOFTMAX only accepts vector input. All non-vector input will be vectorized.*/
			units[i]->input->d = 0;																	/*Total number of elements of all predecessor elements. All non-vector input will be vectorized.*/
			const std::vector<UNIT*> pred = units[i]->get_pred();
			/*register itself to each of the predecessors*/
			for(int j = 0; j<pred.size(); j++){
				pred[j]->outputs.push_back(units[i]->input);
				pred[j]->offsets.push_back(units[i]->input->d);
				units[i]->input->d += pred[j]->h*pred[j]->w*pred[j]->d;								/*Number of vectorized elements (actual size, no padding)*/
			}
			units[i]->input->w_stride = 0;															/*Vectorized unit has not w_stride*/
			units[i]->input->f_stride = 0;															/*SOFTMAX is 1D unit, so its f_stride is invalid.*/
			units[i]->input->d_stride = units[i]->input->d;
			units[i]->input->n_stride = units[i]->input->n;
			if(num_classes != units[i]->input->d){
				printf("GRAPH::build: Invalid network. The total input dimension of SOFTMAX must be equal to number of training classes. #InDim = %d, #Classes = %d.\n",units[i]->input->d,num_classes);
				return false;
			}
		}else if(units[i]->get_type()==UNIT_TYPE_CONV){
			units[i]->input = new PMEM(*env, units[i]->get_id());									/*input buffer of the unit. written by predecessors. actual memory is allocated/freed on the fly.*/
			units[i]->input->n = size_batch;
			units[i]->input->h = units[i]->h;														/*Actual height of output feature maps (input.height=output.height).*/
			units[i]->input->w = units[i]->w;														/*Actual width of output feature maps (input.width=output.width).*/
			units[i]->input->d = units[i]->d;														/*Number of output feature maps.*/
			const std::vector<UNIT*> pred = units[i]->get_pred();									/*A list of predecessors*/
			/*Calculate the zero paddings for each feature in order to apply FFT*/
			CONV* u = (CONV*)units[i];
			int h_padding = floor(u->filter_height/2);
			int w_padding = floor(u->filter_width/2);												/*Padding size for convolution, not including fft.*/
			int d_stride = 0;
			u->num_input_maps = 0;
			int pred_w_padded = 0;
			int offset = 0;
			units[i]->input->f_stride = units[i]->input->w + w_padding;								/*f_stride: number of elements in one row of a feature map (with padding)*/
			units[i]->input->f_stride += 2 - (units[i]->input->f_stride%2);							/*Extend f_stride by 1 or 2 element for fft padding. All predecessors having the same (h,w) dimensions.*/
			/*register itself to each of the predecessors.*/
			for(int j = 0; j<pred.size(); j++){
				pred[j]->outputs.push_back(units[i]->input);
				pred[j]->offsets.push_back(offset);
				u->num_input_maps += pred[j]->d;													/*Number of input feature maps is the total number of output feature maps of all predecessors.*/
				pred_w_padded = pred[j]->w+w_padding+2-(pred[j]->w+w_padding)%2;					/*width after padding: both convolution and fft.*/
				d_stride += pred_w_padded*(pred[j]->h+h_padding)*pred[j]->d;						/*d_stride: number of elements between each sample in a batch.*/
				offset += (units[i]->input->f_stride)*pred[j]->d;									/*offset: f_stride*num_maps*/
			}
			units[i]->input->w_stride = units[i]->input->f_stride*u->num_input_maps;				/*CONV is 2D unit, so its f_stride is valid.*/
			units[i]->input->d_stride = d_stride;													/*d_stride: number of elements between each sample in a batch*/
			units[i]->input->n_stride = size_batch;													/*No padding between or after feature maps*/
		}
	}
	
	//env->host_pinned_mem_object = clCreateBuffer(env->context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, global_offset*sizeof(float) , 0, &status);				/*Pinned memory*/
	
	//env->host_pinned_mem_object = clCreateBuffer(env->context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, global_offset*sizeof(float), aligned_alloc(4096,global_offset*sizeof(float)), &status);		/*DMA*/
	
	//env->host_pinned_mem_object = clCreateBuffer(env->context, CL_MEM_READ_WRITE, global_offset*sizeof(float), 0, &status);									/*Device memory*/
	
	/*if(status != CL_SUCCESS){
		printf("GRAPH::build: clCreateBuffer failed. Unable to allocate pinned cl_mem of size %ld bytes.\n",global_offset*sizeof(float));
		return false;
	}*/

	
	
	/*DMA
::size_t size = N * sizeof(float); // size must be a multiple of 64
::size_t alignment = 4096;
// Allocate aligned memory on the host. The particular function is OS-dependent.
//  memalign(size_t alignment, size_t size) on Linux, release memory with free(...)
//  _aligned_malloc(size_t size, size_t alignment) on Windows, release with _aligned_free(...)
//  posix_memalign(void ** ptr, size_t alignment, size_t size) on OSX, release with free(...)
float * h_results = (float*) memalign(alignment, size);
 
// Create an OpenCL buffer using the host pointer
Buffer d_results = cl::Buffer(context, CL_MEM_USE_HOST_PTR, size, t_results);
	*/
	
	
	int* label = 0;
	for(int i = 0; i<units.size(); i++){
		if (units[i]->outputs.size()>10){
			printf("GRAPH::build: The maximum number of successors is 10 (%s has %d successors).\n",units[i]->get_id().c_str(),(int)units[i]->outputs.size());
			exit(-1);
		}
		if (units[i]->get_type()==UNIT_TYPE_INPUT){
			units[i]->init();
			label = ((INPUT*)units[i])->labels_exposed_to_softmax;
			break;
		}
	}
	
	if (label==0){
		printf("GRAPH::build: no input unit was found.");
		exit(-1);
	}
	
	
	/*Check the graph*/  //DDD
	/*
	1) 2D->1D: The input can be of any dimension (e.g. [Input,Conv,Full]->[Full]).
	2) 2D->2D: The input must be of the same dimension (e.g. [Conv1,Conv2]->[Full], then require that dim(Conv1)=dim(Conv2))
	3) 1D->2D: Undefined (vectorized feature cannot be recovered to 2D feature map).
	4) At least one output unit, one input unit.
	5) |succ|<=10.
	5) 1D units cannot be followed by 2D units (e.g. FULL cannot have successor CONV).
	6) Kernel dimension must be no more than feature map dimension.
	7) Feature maps (2D) connecting to the same unit must have the same dimension (w,h).
	*/
	
	/*Initialize all the units*/
	for(int i = 0; i<units.size(); i++){
		printf("Graph: initializing %s.\n", units[i]->get_id().c_str());
		if (units[i]->get_type()!=UNIT_TYPE_INPUT)
			units[i]->init();
		if(units[i]->get_type()==UNIT_TYPE_SOFTMAX)
			((SOFTMAX*)units[i])->set_label(label);		//register label for softmax units.
	}
	
	return true;
}













