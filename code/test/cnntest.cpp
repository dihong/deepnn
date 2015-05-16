#include <iostream>
#include <random>
#include <cmath>
#include <cstdio>
#include <string.h>

//#include "cnntest.h"
using namespace std;


//dummy implementation
void get_dimension (int& Xd, int& Pd)
{
	Xd = 100;
	Pd = 3;   
}

bool check_unit (const float* X, const float* P, float& E, float* dEdX, float* dEdP)
{
	return true;
}
//dummey implementation


//check dEdX and dEdP
bool test(void)
{
	bool passFlag = true;
	
	int Xd = 0;
	int Pd = 0;

	get_dimension(Xd, Pd);  //get the dimensions.
	float* X = new float [Xd]; //the X is input to the unit to be checked.
	float* P = new float [Pd]; //the P is the parameters to the unit (e.g. filter weights).
	float* dEdX = new float[Xd];
	float* dEdP = new float[Pd];
	
	float* dEdX_nouse = new float[Xd];
	float* dEdP_nouse = new float[Pd];

	float E = 0.0f;
	float E_low = 0.0f;
	float E_high = 0.0f; 

	float* X_low = new float[Xd];
	float* X_high = new float[Xd];

	float* P_low = new float[Pd];
	float* P_high = new float[Pd];

	//randomly initialize the input X as numbers between 0 and 1
	for(int i = 0; i < Xd; i++)  
	{
		X[i] = float(rand() % 10) / 10.0f;  //RAND_MAX;
	}

	//randomly initialize the P as numbers between 0 and 1
	for(int i = 0; i < Pd; i++)  
	{
		P[i] = float(rand() % 10) / 10.0f;  //RAND_MAX;
	}

	//get dEdXs from all points in matrix X
	bool flag = check_unit (X, P, E, dEdX, dEdP);

	if(flag == true)
	{		
		float delta = 0.001f;  //set delta
		for (int s = 0; s < Xd; s++)
		{
			memcpy(X_low, X, sizeof(float)* Xd);
			memcpy(X_high, X, sizeof(float)* Xd);
			
		    X_low[s] = X[s] - delta;
			X_high[s]  = X[s] + delta;

			//keep P the same
			//get E_low when increase X[s]
			//get E_high when increase X[s]
			check_unit(X_low, P, E_low, dEdX_nouse, dEdP_nouse);
			check_unit(X_high, P, E_high, dEdX_nouse, dEdP_nouse);
			
			if (abs(E_high - E_low) / (2*delta) != dEdX[s])
				{
					cout << "check dEdX not match" << endl;
					cout << "delta:" << delta << endl;
				}
		}

		delta = 0.001f;  //set delta
		for (int s = 0; s < Pd; s++)
		{
			memcpy(P_low, P, sizeof(float) * Pd);
			memcpy(P_high, P, sizeof(float) * Pd);
			
			P_low[s] = P[s] - delta;
			P_high[s] = P[s] + delta;


			//keep X the same
			//get E_low when increase P[s]
			//get E_high when increase P[s]
			check_unit(X, P_low, E_low, dEdX_nouse, dEdP_nouse);
			check_unit(X, P_high, E_high, dEdX_nouse, dEdP_nouse);

			if (abs(E_high - E_low ) / (2*delta) != dEdP[s])
			{
				cout << "check dEdP not match" << endl;
				cout << "delta:" << delta << endl;
			}
		}

	//Here you have got valid outputs, and you need to check (dEdX, dEdP).
	/*How the check dEdX:
	  (1) Randomly select an integer s, where s>=0 and s<Xd.
	  (2) X1 = X; X1[s] = X1[s] - delta
	  (3) X2 = X; X2[s] = X2[s] + delta
	  (4) check_unit (X1, P, E1, dEdX, dEdP)
	  (5) check_unit (X2, P, E2, dEdX, dEdP)
	  (6) If (E2-E1)/(X2-X1) is approximately equal to dEdX[s], then the unit is correct. Otherwise it is incorrect.
	  (7) Repeat (1)-(6) for several times.
	  The checking of dEdP is similar to dEdX.
	  Reference: http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking
	*/
	}
	else{
		cout << "check_unit failed" << endl;
	}
}


int main (void)
{
	test();

	return 0;
}

