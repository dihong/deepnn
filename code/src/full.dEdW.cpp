#include "full.hpp"
#include <vector>
#include "main.hpp"
#include "common.hpp"


using namespace std;

std::string FULL::generate_header_dEdW(const int& tsm, const int & tsn, const int& tsk, const int& wptm, const int& wptn, const int& width) {
    char ret[1024];
    sprintf(ret, "#define DEDW\n#define WIDTH %d\n#define TSM %d\n#define TSN %d\n\
    #define TSK %d\n#define WPTM %d\n#define WPTN %d\n#define RTSM (TSM/WPTM)\n\
    #define RTSN (TSN/WPTN)\n#define LPTA ((TSK*WPTM*WPTN)/(TSN))\n#define LPTB ((TSK*WPTM*WPTN)/(TSM))\n", width, tsm, tsn, tsk, wptm, wptn);
    return ret;
}

/*Check: dEdW = X'*dEdB*/
float check_dEdW(float* dEdB, float* X, float* dEdW, int _N, int _P, int _Q) {

    float err = 0;
    int N = roundup(_N);
    int P = roundup(_P);
    int Q = roundup(_Q);

    /*mxshow(dEdB, N, Q);
    mxshow(X, N, P);
    mxshow(dEdW,P,Q);*/

    float* dEdW0 = new float [P * Q];
    memset(dEdW0, 0, sizeof (float)*P * Q);
    float sum = 0;
    for (int k = 0; k < _N; k++) {
        for (int i = 0; i < _P; i++) {
            for (int j = 0; j < _Q; j++) {
                dEdW0[i * Q + j] += dEdB[k * Q + j] * X[k * P + i];
            }
        }
    }
    for (int i = 0; i < P * Q; i++)
        err = err > fabs(dEdW[i] - dEdW0[i]) ? err : fabs(dEdW[i] - dEdW0[i]);
    return err;
}


/*Get the best parameter configuration w.r.t. underlying hardwares*/

/*Return true is successful, false otherwise*/
bool FULL::get_optimal_parameters_dEdW(const DEVICE& dev, int& tsk, int& tsm, int & tsn, int& wptm, int& wptn, int& width, float& kernel_time, float& gflops) {

    /*generate candidate parameters*/
    vector<int> tsks;
    vector<int> tsms;
    vector<int> tsns;
    vector<int> wptms;
    vector<int> wptns;

    if (N > 512)
        tsks.push_back(16);
    else if (N > 128) {
        tsks.push_back(16);
        tsks.push_back(8);
    } else {
        tsks.push_back(16);
        tsks.push_back(8);
        tsks.push_back(4);
    }

    if (Q > 512) {
        tsms.push_back(128);
        tsms.push_back(64);
    } else if (Q > 64) {
        tsms.push_back(128);
        tsms.push_back(64);
        tsms.push_back(32);
    } else {
        tsms.push_back(Q);
        tsms.push_back(Q / 2);
    }

    if (N < 32 || (P < 32 && Q > 512))
        tsms.push_back(N);

    if (Q > 256) {
        wptms.push_back(8);
        wptms.push_back(4);
    } else if (Q > 64) {
        wptms.push_back(4);
        wptms.push_back(2);
    } else {
        wptms.push_back(2);
        wptms.push_back(1);
    }
    if (P > 512) {
        tsns.push_back(128);
        tsns.push_back(64);
    } else if (P > 64) {
        tsns.push_back(128);
        tsns.push_back(64);
        tsns.push_back(32);
    } else {
        tsns.push_back(P);
        tsns.push_back(P / 2);
    }

    if (N < 32 || (Q < 32 && P > 512))
        tsns.push_back(N);


    if (P > 256) {
        wptns.push_back(8);
        wptns.push_back(4);
    } else if (P > 64) {
        wptns.push_back(4);
        wptns.push_back(2);
    } else {
        wptns.push_back(4);
        wptns.push_back(2);
        wptns.push_back(1);
    }


    width = 4; //vector loading width.
    if (N < 16)
        width = 1;		///DDD


    kernel_time = 10000;
    gflops = 0;
    cl_int status;
    /*Move dEdW, X and dEdB into device memory.*/
    for (int i = 0; i < _P; i++)
        for (int j = 0; j < _Q; j++)
            dEdW->data1[i * Q + j] = (0.1 + rand()) / RAND_MAX;
    dEdW->set_data2();
    
    input->create_input_buffer();
    input->set_data1();				//copy data2 into data1 (to create a data1 object).
    for (int i = 0; i < _N; i++)
        for (int j = 0; j < _P; j++)
            input->data1[i * P + j] = (0.1 + rand()) / RAND_MAX;
    input->set_data2();
    for (int i = 0; i < _N; i++)
        for (int j = 0; j < _Q; j++)
            dEdB->data1[i * Q + j] = (0.1 + rand()) / RAND_MAX;
    dEdB->set_data2();
    /*Calculation*/
    for (int i = 0; i < tsks.size(); i++) {
        for (int j = 0; j < tsms.size(); j++) {
            for (int k = 0; k < tsns.size(); k++) {
                for (int u = 0; u < wptms.size(); u++) {
                    for (int v = 0; v < wptns.size(); v++) {

                        /*check constraints*/
                        if (tsns[k] < 4) continue;
                        if (tsms[j] < 4) continue;
                        if (N % tsks[i] != 0) continue;
                        if ((tsks[i] * wptms[u] * wptns[v]) % tsns[k] != 0) continue;
                        if ((tsks[i] * wptms[u] * wptns[v]) % tsms[j] != 0) continue;
                        if (((tsks[i] * wptms[u] * wptns[v]) / (tsns[k])) % width != 0) continue;
                        if (((tsks[i] * wptms[u] * wptns[v]) / (tsms[j])) % width != 0) continue;
                        if (tsks[i] % width != 0) continue;
                        if (tsms[j] % wptms[u] != 0) continue;
                        if (tsns[k] % wptns[v] != 0) continue;
                        if (tsms[j] > Q) continue;
                        if (tsns[k] > P) continue;
                        if ((tsms[j] / wptms[u])*(tsns[k] / wptns[v]) > dev.MaxWorkgroupSize) continue;
                        if (sizeof (float) * tsks[i]*(tsns[k] + tsms[j]) > 0.95 * dev.LocalMemSize) continue;

                        /*Create program object */
                        string source = readKernelFile("src/full.cl");
                        string header = generate_header_dEdW(tsms[j], tsns[k], tsks[i], wptms[u], wptns[v], width).c_str();
                        string code = header + source;
                        const char* constCode = code.c_str();
                        cl_program program = clCreateProgramWithSource(env.context, 1, &constCode, 0, &status);
                        if (status != CL_SUCCESS) {
                            printf("FULL::get_optimal_parameters_dEdW: unable to create program.  %s.\n", getErrorString(status));
                            puts(constCode);
                            exit(-1);
                        }

                        /*Build program. */
                        clBuildProgram(program, 1, &dev.id, NULL, NULL, NULL);

                        /*Creae kernel*/
                        cl_kernel kernel = clCreateKernel(program, "dEdW", &status);
                        if (status != CL_SUCCESS) {
                            printf("FULL::get_optimal_parameters_dEdW: unable to create kernel.  %s (%s).\n", getErrorString(status), "dEdW");
                            puts(constCode);
                            exit(-1);
                        }
                        /*Set kernel arguments*/
                        clSetKernelArg(kernel, 0, sizeof (dEdB->data2), (void *) &dEdB->data2); 	/*dEdB*/
                        clSetKernelArg(kernel, 1, sizeof (input->data2), (void *) &input->data2); 	/*X*/
                        clSetKernelArg(kernel, 2, sizeof (dEdW->data2), (void *) &dEdW->data2); 	/*dEdW*/
                        clSetKernelArg(kernel, 3, sizeof (int), (void *) &P);
                        clSetKernelArg(kernel, 4, sizeof (int), (void *) &Q);
                        clSetKernelArg(kernel, 5, sizeof (int), (void *) &N);

                        /*run kernel*/
                        double kt = 0;
                        double fp = 0;
                        float err = -1;
                        int t;
                        cl_event ndrEvt;
                        for (t = 0; t < 3;) {
                            size_t global_work_size[] = {(size_t)(Q / wptms[u]), (size_t)(P / wptns[v])};
                            size_t local_work_size[] = {(size_t)(tsms[j] / wptms[u]), (size_t)(tsns[k] / wptns[v])};
                            status = clEnqueueNDRangeKernel(
                                    dev.compute_q,
                                    kernel,
                                    2, //work dimension.
                                    NULL, //global work offset.
                                    global_work_size,
                                    local_work_size,
                                    0, //number of events in wait list.
                                    NULL, //event wait list.
                                    &ndrEvt);
                            if (status != CL_SUCCESS) {
                                printf("FULL::get_optimal_parameters_dEdW: unable to run the kernel. %s.\n", getErrorString(status));
                                exit(-1);
                            }
                            clFlush(dev.compute_q);
                            clWaitForEvents(1, &ndrEvt);
                            /*Get profiling info.*/
                            cl_ulong start = 0, end = 0;
                            clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_START, sizeof (cl_ulong), &start, NULL);
                            clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_END, sizeof (cl_ulong), &end, NULL);
                            t++;
                            kt += (cl_double) (end - start)*(cl_double) (1e-09);
#ifdef DEBUG0
                            if (t == 1) {
                                /*Get the dEdW back from device*/
                                dEdW->set_data1();

                                /*Validate the results*/
                                err = check_dEdW(dEdB->data1, input->data1, dEdW->data1, _N, _P, _Q);
                                if (err > 1e-6) {
                                    printf("P = %d, Q = %d, N = %d, tsk = %d, tsm = %d, tsn = %d, wptm = %d, wptn = %d\t", P, Q, N, tsks[i], tsms[j], tsns[k], wptms[u], wptns[v]);
                                    printf("FULL::get_optimal_parameters_dEdW: Incorrect results. Err = %.6f.\n", err);
                                    exit(-1);
                                }
                                //
                            }
#endif

                            if (2 * (double) P * Q * N / (kt / t)*(1e-9) < 0.7 * gflops) {
                                break;
                            }

                        }

                        kt /= t;
                        fp = 2 * (double) P * Q * N / kt * (1e-9);

                        clReleaseEvent(ndrEvt);
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                        if (gflops < fp) {
                            kernel_time = kt;
                            gflops = fp;
                            tsk = tsks[i];
                            tsm = tsms[j];
                            tsn = tsns[k];
                            wptm = wptms[u];
                            wptn = wptns[v];
                        }
#ifdef DEBUG3
                        printf("P = %d, Q = %d, N = %d, tsk = %d, tsm = %d, tsn = %d, wptm = %d, wptn = %d, kernel_time = %.6f, gflops = %.6f, Err = %.8f\n", P, Q, N, tsks[i], tsms[j], tsns[k], wptms[u], wptns[v], kt, fp, err);
#endif
                    }
                }
            }
        }
    }
    input->free_data2();
    dEdB->free_data2();
    dEdW->free_data2();
    if (kernel_time < 10000)
        return true;
    else
        return false;
}





