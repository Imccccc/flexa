#include <mpi.h>
#include "mri.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "mkl.h"
#include <complex>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <time.h>
#include <stddef.h>
#include <memory.h>
#include <iomanip>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_rng.h>

MRI::MRI(int rank, int size, std::string data_directory, double lambda, bool verbose, int problem_solver){
	
	// Object variables initialization
	this -> rank = rank;
	this -> size = size;
	this -> lambda = lambda;
	this -> verbose = verbose;
	this -> problem_solver = problem_solver;
	time_flag = 1; elapsed_time = 0; iterations = 0;
	communication_time = 0; initialization_time = 0;  
	
	// Synchronizing processes
	MPI_Barrier(MPI_COMM_WORLD);

	// Read data
		// Useful variable creation
		char *data_path = new char[100];
		std::ifstream file;
		int p; double bin;
		// Reading problem dimensions
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/problem_dimensions.dat");
		if (rank == 0) printf("[ALL] reading problem dimensions: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		file >> Ky; file >> Kx; file >> t;
		if (rank == 0) printf("DONE \n");
		file.close();
		n = 2*Ky*Kx*t;
		Ky_local = Ky/size;  
		n_local = n/size;
		// Read random sampling mask
		mask = new int[n_local/2];
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/mask.dat");
		if (rank == 0) printf("[ALL] reading random sampling mask: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		nnz_local = 0; p = 0; int temp; for (int i = 0; i < n/2; i++) { if (i >= rank*(n_local/2) && i < (rank+1)*(n_local/2)) { file >> mask[p]; p += 1; if(mask[p-1] == 1) nnz_local += 2; } else file >> bin; }
		if (rank==0) printf("DONE \n");
		file.close();	
		// Read Kx-Ky-t domain data
		image = new double[n_local];
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/data_kx_ky_t.dat");
		if (verbose && rank == 0) printf("[ALL] reading kx-ky-t domain image: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		p = 0; for (int i = 0; i < n/2; i++) { if (i >= rank*(n_local/2) && i < (rank+1)*(n_local/2)){ file >> image[p]; file >> image[p+1]; p += 2; } else{ file >> bin; file >> bin; } } 
		if (verbose && rank == 0) printf("DONE \n");
		file.close();
		// Read x-y-f domain data
		reference_image = new double[n_local];
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/data_x_y_f.dat");
		if (verbose && rank == 0) printf("[ALL] reading x-y-f domain reference image: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		p = 0; for (int i = 0; i < n/2; i++) { if (i >= rank*(n_local/2) && i < (rank+1)*(n_local/2)){ file >> reference_image[p]; file >> reference_image[p+1]; p += 2; } else{ file >> bin; file >> bin; } }
		if (verbose && rank == 0) printf("DONE \n");
		file.close();		

	// Initialize Fast Fourier Transforms
		// FFTs dimensions setting
		MKL_LONG stride_Kx[2]; stride_Kx[0] = 0; stride_Kx[1] = t;
		MKL_LONG stride_Ky[2]; stride_Ky[0] = 0; stride_Ky[1] = Kx*t;
		MKL_LONG stride_t[2]; stride_t[0] = 0; stride_t[1] = 1;	
		int number_of_transforms_Kx = t;
		int number_of_transforms_Ky = t*Kx;
		int number_of_transforms_t = Kx*Ky_local;
		int input_distance_Kx = 1;
		int input_distance_Ky = 1;
		int input_distance_t = t;
		// FFts descriptors initialization
		status_Kx = DftiCreateDescriptor(&dft_descriptor_handle_Kx, DFTI_DOUBLE, DFTI_COMPLEX, 1, Kx);
		status_Ky = DftiCreateDescriptor(&dft_descriptor_handle_Ky, DFTI_DOUBLE, DFTI_COMPLEX, 1, Ky);
		status_t = DftiCreateDescriptor(&dft_descriptor_handle_t, DFTI_DOUBLE, DFTI_COMPLEX, 1, t);
		// FFTs parameters setting
		status_Kx = DftiSetValue(dft_descriptor_handle_Kx, DFTI_PLACEMENT, DFTI_INPLACE);
		status_Ky = DftiSetValue(dft_descriptor_handle_Ky, DFTI_PLACEMENT, DFTI_INPLACE);
		status_t = DftiSetValue(dft_descriptor_handle_t, DFTI_PLACEMENT, DFTI_INPLACE);
		status_Kx = DftiSetValue(dft_descriptor_handle_Kx, DFTI_BACKWARD_SCALE, 1.0/std::sqrt(Kx));
		status_Ky = DftiSetValue(dft_descriptor_handle_Ky, DFTI_BACKWARD_SCALE, 1.0/std::sqrt(Ky));
		status_t = DftiSetValue(dft_descriptor_handle_t, DFTI_BACKWARD_SCALE, 1.0/std::sqrt(t));
		status_Kx = DftiSetValue(dft_descriptor_handle_Kx, DFTI_FORWARD_SCALE, 1.0/std::sqrt(Kx));
		status_Ky = DftiSetValue(dft_descriptor_handle_Ky, DFTI_FORWARD_SCALE, 1.0/std::sqrt(Ky));
		status_t = DftiSetValue(dft_descriptor_handle_t, DFTI_FORWARD_SCALE, 1.0/std::sqrt(t));
		status_Kx = DftiSetValue(dft_descriptor_handle_Kx, DFTI_NUMBER_OF_TRANSFORMS, number_of_transforms_Kx);
		status_Ky = DftiSetValue(dft_descriptor_handle_Ky, DFTI_NUMBER_OF_TRANSFORMS, number_of_transforms_Ky);
		status_t = DftiSetValue(dft_descriptor_handle_t, DFTI_NUMBER_OF_TRANSFORMS, number_of_transforms_t);
		status_Kx = DftiSetValue(dft_descriptor_handle_Kx, DFTI_INPUT_DISTANCE, input_distance_Kx);
		status_Ky = DftiSetValue(dft_descriptor_handle_Ky, DFTI_INPUT_DISTANCE, input_distance_Ky);
		status_t = DftiSetValue(dft_descriptor_handle_t, DFTI_INPUT_DISTANCE, input_distance_t);
		status_Kx = DftiSetValue(dft_descriptor_handle_Kx, DFTI_INPUT_STRIDES, stride_Kx);
		status_Ky = DftiSetValue(dft_descriptor_handle_Ky, DFTI_INPUT_STRIDES, stride_Ky);
		status_t = DftiSetValue(dft_descriptor_handle_t, DFTI_INPUT_STRIDES, stride_t);
		// FFTs descriptors committing
		status_Kx = DftiCommitDescriptor(dft_descriptor_handle_Kx);
		status_Ky = DftiCommitDescriptor(dft_descriptor_handle_Ky);
		status_t = DftiCommitDescriptor(dft_descriptor_handle_t);

	// Compute total number of samples in the Kx-Ky-t domain
	MPI_Barrier(MPI_COMM_WORLD);
	double communication_begin = MPI_Wtime();
	MPI_Allreduce(&nnz_local, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	
	// Compute MSE normalizing factor
	double normalizing_factor_local = 0;
	for (int i = 0; i < n_local; i++) normalizing_factor_local = normalizing_factor_local + reference_image[i]*reference_image[i]; 
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allreduce(&normalizing_factor_local, &normalizing_factor, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
		
}
	
MRI::~MRI(){
	
	if(rank == 0) std::cout << "[ALL] freeing memory: ";  
	delete [] x; delete [] b; delete [] F; delete [] nablaF;
	delete [] mask; delete [] image; delete [] reference_image;
	delete [] y; delete [] y_zf; delete [] y_big;
	status_Kx = DftiFreeDescriptor(&dft_descriptor_handle_Kx);
	status_Ky = DftiFreeDescriptor(&dft_descriptor_handle_Ky); 
	status_t = DftiFreeDescriptor(&dft_descriptor_handle_t); 	 
	if(rank == 0) { delete [] times; delete [] values; delete [] merits;  delete [] mse; }
	if(rank == 0) std::cout << "DONE" << std::endl;
	
}

void MRI::apply_mask(double *input, double *output){
	
	int p = 0;
	for (int i = 0; i < n_local/2; i++){
		if (mask[i]==1) {
			output[p] = input[2*i];
			output[p+1] = input[2*i+1];
			p += 2;
			}
		}
		
}

void MRI::zero_filling(double *input, double *output){

	int p = 0;
	for (int i = 0; i < n_local/2; i++){
		if (mask[i]) {
			output[2*i] = input[p];
			output[2*i+1] = input[p+1];
			p += 2;
			} 
		else {
			output[2*i] = 0;
			output[2*i+1] = 0;
			}
		}
		
}

void MRI::compute_F(double *x){	
	double *x_temp = new double [n_local];
	x_temp[0:n_local:1] = x[0:n_local:1];  
	status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)x_temp);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeForward(dft_descriptor_handle_Kx, (std::complex<double>*)(x_temp+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	double communication_begin = MPI_Wtime();
	MPI_Allgather(x_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	y_zf[0:n_local:1] = y_big[rank*n_local:n_local:1];
	apply_mask(y_zf, y);
	F[0:nnz_local:1] = y[0:nnz_local:1] - b[0:nnz_local:1]; 
	delete [] x_temp;
}

void MRI::compute_function_gradient(){
	double *y_temp = new double [n_local]; zero_filling(F,y_temp);
	status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)y_temp);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_temp+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	double communication_begin = MPI_Wtime();
	MPI_Allgather(y_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	nablaF[0:n_local:1] = y_big[rank*n_local:n_local:1]; 
	delete [] y_temp;		
}

double MRI::compute_function_value(double *x){	
	double regularization_term_local = 0; double value_local = 0; double value;
	for (int i = 0; i < nnz_local; i++) value_local = value_local + F[i]*F[i]; value_local *= 0.5;
	for (int i = 0; i < n_local; i++) regularization_term_local = regularization_term_local + std::abs(x[i]);
		regularization_term_local *= lambda;
	value_local = value_local + regularization_term_local; 
	MPI_Barrier(MPI_COMM_WORLD);
	double communication_begin = MPI_Wtime();
	MPI_Allreduce(&value_local, &value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
	communication_time += (MPI_Wtime() - communication_begin);
	return value;
}

double MRI::compute_merit_value(int merit_function){
	
	double merit_local = 0; double merit;
	if (merit_function == 0){
		for (int i = 0; i < n_local; i++) merit_local = merit_local + ((t/std::sqrt(Kx*Ky*t))*x[i] - reference_image[i])*((t/std::sqrt(Kx*Ky*t))*x[i] - reference_image[i]); 
		double communication_begin = MPI_Wtime();
		MPI_Allreduce(&merit_local, &merit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
		communication_time += (MPI_Wtime() - communication_begin);
		merit /= normalizing_factor;
		}
	else{
		for (int i = 0; i < n_local; i++) {
			double Fi = nablaF[i] - x[i];	
			if (Fi >= lambda) Fi = lambda;
			if (Fi <= -lambda) Fi = -lambda;
			double distance = std::abs(nablaF[i]-Fi);
			merit_local = std::max(merit_local, distance);
			}
		MPI_Barrier(MPI_COMM_WORLD);
		double communication_begin = MPI_Wtime();
		MPI_Allreduce(&merit_local, &merit, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
		communication_time += (MPI_Wtime() - communication_begin);
		}
	return merit;	
		
}

void MRI::solve_subproblem_ST_J(double rho, double M_global, double gamma, double *delta, int update_strategy, double *local_function_value){
	
	// Variables definition
	double direction, xnew, Fnew;
	
	// Loop begin
	for (int i = 0; i < n_local; i++) { 
		if (std::abs(delta[i]) >= rho*M_global){ 
			// Direction evaluation
			direction = -delta[i];
			// New point calculation (one block update strategy)
			/*if (update_strategy && size > 1){ 
				xnew = x[i] - gamma*direction;
				for (int j = 0; j < m; j++) {
					Fnew = F[j] + A[j+i*m]*(xnew - x[i]); 
					F_value = F_value + 0.5*(Fnew*Fnew - F[j]*F[j]);
					F[j] = Fnew;
					}
				*local_function_value = *local_function_value - std::abs(x[i]) + std::abs(xnew);
				x[i] = xnew;
				}*/
			// New point calculation (all blocks update strategy)				
			/*else*/ x[i] = x[i] - gamma*direction;	
			}
		}
		
}

void MRI::solve_subproblem_ST_GS(double rho, double M_global, double tau, double gamma, double *delta, int update_strategy, double *local_function_value){
	
	// Variables definition
	int counter = 0; 
	double direction, xopt, xnew, Fnew;
	
	// Random update pattern generation
	int *indices = new int[n_local]; 
	for (int i = 0; i < n_local; i++) indices[i] = i;
	for (int i = 0; i < n_local; i++) {
		int one = std::rand() % n_local;
		int two = std::rand() % n_local;
		int tmp = indices[one];
		indices[one] = indices[two];
		indices[two] = tmp;
		}
	
	// Active set selection
	int *active_set = new int[n_local+1];
	for (int q = 0; q < n_local; q++) {
		int i = indices[q];
		if (std::abs(delta[i]) >= rho*M_global) {active_set[counter] = i; counter ++; }
		} active_set[counter] = -1;
		
	// Loop begin
	for (int q = 0; q < counter; q++) {	
		// Current index and next index selection
		int i = active_set[q];
		int next_index = active_set[q+1];
		// Scalar sub-problem optimal solution		
		xopt = x[i] - nablaF[i]/(1.0*nnz/n + tau);				
		double parameter = lambda/(1.0*nnz/n + tau);	
		if (xopt >= parameter) xopt -= parameter;
		else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }	
		// Direction evaluation
	    direction = -(xopt - x[i]);
		// New point calculation
		xnew = x[i] - gamma*direction; 
		if (next_index >= 0){
			x[i] = xnew;	
			compute_F(x);
			compute_function_gradient();
			/*nablaF[next_index] = 0;
			for (int j = 0; j < m; j++){
				if (update_strategy && size > 1){
					Fnew = F[j] + A[j+i*m]*(xnew - x[i]); 
					F_value = F_value + 0.5*(Fnew*Fnew - F[j]*F[j]);
					F[j] = Fnew;
					}
				elseF[j] = F[j] + A[j+i*m]*(xnew - x[i]);
				nablaF[next_index] = nablaF[next_index] + A[next_index*m+j]*F[j];
				} */
			}			
		else{
			/*if(size > 1 && update_strategy){
				for (int j = 1; j < m; j++){
					Fnew = F[j] + A[j+i*m]*(xnew - x[i]); 
					F_value = F_value + 0.5*(Fnew*Fnew - F[j]*F[j]);
					F[j] = Fnew;
					}
				}*/
			}
		//if (update_strategy && size > 1) *local_function_value = *local_function_value - std::abs(x[i]) + std::abs(xnew);
		//x[i] = xnew;	
		}

	// Freeing memory
	delete [] indices; delete [] active_set;
		
}	

void MRI::solve_flexa(double merit_tolerance, int max_iterations, double max_time, int solver, int update_strategy, double tau, double gamma, double rho){	
	
	// Tau update rule - parameter settings (proximal term weight)
	double tau_step = 0.5;
	double tau_decrease_l1 = 1e-1;
	int tau_decrease_l2 = 10; 
	int max_tau_updates = 0;
	
	// Gamma update rule - parameters setting (stepsize)
	double epsilon = 1e-7;
	double gamma_factor = 1e-1;  
	
	// Variables definition memory allocation and initialization
	double communication_begin, initialization_begin;
	int tau_increments = 0;	int tau_decrements = 0; int function_decrements = 0; 
	if (rank == 0) { times = new double[max_iterations+1]; values = new double[max_iterations+1]; merits = new double[max_iterations+1]; mse = new double[max_iterations+1];}
	y = new double[nnz_local]; y_zf = new double [n_local](); y_big = new double [n];
	x = new double[n_local](); b = new double[nnz_local]; F = new double[nnz_local](); nablaF = new double[n_local](); 
	double *x_old = new double[n_local]; double *delta = new double[n_local]; double function_value_old;
	initialization_begin = MPI_Wtime();
	apply_mask(image, b);	
	F[0:nnz_local:1] = -b[0:nnz_local:1];
	compute_function_gradient();
	function_value = compute_function_value(x);
	merit_value = compute_merit_value(1);
    mse_value = compute_merit_value(0);
    function_value_old = function_value;
    initialization_time += (MPI_Wtime() - initialization_begin);
    if (solver == 1) srand(time(NULL));
	
    // Number of processes initialization time optimal function value printing out
    if (rank == 0) std::cout << "Number of parallel processes: " << size << std::endl;
	if (rank == 0) std::cout << "Initialization time: " << std::scientific << std::setprecision(6) << initialization_time << std::endl;
	
	// First iteration function value and merit value printing out 
	if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
	
	// First iteration time and value saving step 
	if (rank == 0) { times[iterations] = elapsed_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;} 
	
	/*************************ALGORITHM BEGIN**************************/
	// Timing start
	double start_time = MPI_Wtime();
		
	while (merit_value >= merit_tolerance && iterations < max_iterations && time_flag){
		
		// Memorizing actual point 
		x_old[0:n_local:1] = x[0:n_local:1];  
		
		// Solving scalars sub-problems with soft-tresholding for greedy variables selection	
		double M_local = 0; double M_global; double xopt;		
		for (int i = 0; i < n_local; i++) {
			xopt = x[i] - nablaF[i]/(1.0*nnz/n + tau);	
			double parameter = lambda/(1.0*nnz/n + tau);
			if (xopt >= parameter) xopt -= parameter;
			else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }
			delta[i] = xopt - x[i];
			double distance = std::abs(xopt - x[i]);
			if (distance > M_local) M_local = distance;
			}	
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allreduce(&M_local, &M_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		communication_time += (MPI_Wtime() - communication_begin);
		
		// Useful variables initialization for Single-Block-Update schemes
		double local_function_value = 0; double min_function_value; 
		if(update_strategy && size > 1) local_function_value = (function_value - F_value)/lambda;
		
		// Solving scalars sub-problems for direction evaluation
		if (solver == 0) solve_subproblem_ST_J(rho, M_global, gamma, delta, update_strategy, &local_function_value);
		if (solver == 1) solve_subproblem_ST_GS(rho, M_global, tau, gamma, delta, update_strategy, &local_function_value);
		
		// Variables update
		if (update_strategy && size > 1){
			local_function_value *= lambda; local_function_value += F_value;		
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Allreduce(&local_function_value, &min_function_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			if(local_function_value != min_function_value){	x[0:n_local:1] = x_old[0:n_local:1]; }
			}
		compute_F(x);
		function_value = compute_function_value(x);		
		iterations ++;

		// Tau update and next loop useful computations
		if (function_value >= function_value_old) {
			function_decrements = 0; tau_increments++; tau /= tau_step;
			x[0:n_local:1] = x_old[0:n_local:1];
			compute_F(x);
			compute_function_gradient();
			} 
		else {
			compute_function_gradient();
			function_decrements++; function_value_old = function_value;
			if ((merit_value < tau_decrease_l1 && function_decrements >= tau_decrease_l2) && (tau_decrements < max_tau_updates)) {
				tau *= tau_step; tau_decrements++; function_decrements = 0;
				}
			}
		mse_value = compute_merit_value(0);		
		merit_value = compute_merit_value(1);

		// Gamma update
		gamma *= 1-std::min(1.0,gamma_factor/merit_value)*epsilon*gamma;
			
		// Times and values saving step
		if (rank == 0) { times[iterations] = MPI_Wtime() - start_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;}  
		
		// Iteration, function value and merit value printing out
		if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
		
		// Time flag setting
		if (rank == 0 && (MPI_Wtime() - start_time) >= max_time) { time_flag = 0; }
		MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		
		}
	
	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	/**************************ALGORITHM END***************************/
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
		
	// Freeing memory
	delete [] delta; delete [] x_old;
	
}

void MRI::solve_fista(double merit_tolerance, int max_iterations, double max_time, double power_method_tolerance, int power_method_max_iterations){	
	
	// Variables definition memory allocation and initialization
	double communication_begin, initialization_begin;
	if (rank == 0) { times = new double[max_iterations+1]; values = new double[max_iterations+1]; merits = new double[max_iterations+1]; mse = new double[max_iterations+1];}
	y = new double[nnz_local]; y_zf = new double [n_local](); y_big = new double [n];
	x = new double[n_local](); b = new double[nnz_local]; F = new double[nnz_local](); nablaF = new double[n_local](); 
	double *x_old = new double[n_local](); double *x_new = new double[n_local](); double t_old; double t_new = 1;
	initialization_begin = MPI_Wtime();
    apply_mask(image, b);	
	F[0:nnz_local:1] = -b[0:nnz_local:1];
	compute_function_gradient();
	function_value = compute_function_value(x);
	merit_value = compute_merit_value(1);
    mse_value = compute_merit_value(0);
	double local_error, global_error, reference_error, local_norm, global_norm;
	double *x_local = new double [n_local]; double *w_local = new double [n_local];
	std::fill(x_local, x_local+n_local, 1.0/std::sqrt(n));
	double *x_temp = new double [n_local]; double *y_temp = new double [n_local]; 
	x_temp[0:n_local:1] = x_local[0:n_local:1];
	status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)x_temp);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeForward(dft_descriptor_handle_Kx, (std::complex<double>*)(x_temp+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allgather(x_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	y_zf[0:n_local:1] = y_big[rank*n_local:n_local:1];
	zero_filling(y_zf,y_temp);
	status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)y_temp);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_temp+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allgather(y_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	w_local[0:n_local:1] = y_big[rank*n_local:n_local:1]; 
	local_error = 0; local_norm = 0; 
	for (int i = 0; i < n_local; i++) { local_error += (x_local[i]*w_local[i]); local_norm += (w_local[i]*w_local[i]); } 
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allreduce(&local_error, &reference_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
	communication_time += (MPI_Wtime() - communication_begin);
	global_norm = std::sqrt(global_norm);
	w_local[0:n_local:1] = (1/global_norm)*w_local[0:n_local:1];
	x_local[0:n_local:1] = w_local[0:n_local:1];
	x_temp[0:n_local:1] = x_local[0:n_local:1];
	status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)x_temp);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeForward(dft_descriptor_handle_Kx, (std::complex<double>*)(x_temp+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allgather(x_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	y_zf[0:n_local:1] = y_big[rank*n_local:n_local:1];
	zero_filling(y_zf,y_temp);
	status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)y_temp);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_temp+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allgather(y_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	w_local[0:n_local:1] = y_big[rank*n_local:n_local:1]; 
	local_error = 0; local_norm = 0; 
	for (int i = 0; i < n_local; i++) { local_error += (x_local[i]*w_local[i]); local_norm += (w_local[i]*w_local[i]); } 
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
	communication_time += (MPI_Wtime() - communication_begin);
	global_norm = std::sqrt(global_norm);
	w_local[0:n_local:1] = (1/global_norm)*w_local[0:n_local:1];
	x_local[0:n_local:1] = w_local[0:n_local:1];
	int counter = 0;
	while (std::abs(global_error-reference_error) > power_method_tolerance*global_error && counter < power_method_max_iterations){
		reference_error = global_error;
		x_temp[0:n_local:1] = x_local[0:n_local:1];
		status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)x_temp);	
		for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeForward(dft_descriptor_handle_Kx, (std::complex<double>*)(x_temp+2*i*Kx*t));
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allgather(x_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
		communication_time += (MPI_Wtime() - communication_begin);
		status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
		y_zf[0:n_local:1] = y_big[rank*n_local:n_local:1];
		zero_filling(y_zf,y_temp);
		status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)y_temp);	
		for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_temp+2*i*Kx*t));
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allgather(y_temp, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
		communication_time += (MPI_Wtime() - communication_begin);
		status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
		w_local[0:n_local:1] = y_big[rank*n_local:n_local:1]; 
		local_error = 0; local_norm = 0; 
		for (int i = 0; i < n_local; i++) { local_error += (x_local[i]*w_local[i]); local_norm += (w_local[i]*w_local[i]); } 
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
		communication_time += (MPI_Wtime() - communication_begin);
		global_norm = std::sqrt(global_norm);
		w_local[0:n_local:1] = (1/global_norm)*w_local[0:n_local:1];
		x_local[0:n_local:1] = w_local[0:n_local:1];
		counter += 1; 
		}
	double L = global_error;
	delete [] x_temp; delete [] y_temp;
	delete [] x_local; delete [] w_local;
	initialization_time += (MPI_Wtime() - initialization_begin);

    // Number of processes initialization time optimal function value printing out
    if (rank == 0) std::cout << "Number of parallel processes: " << size << std::endl;
	if (rank == 0) std::cout << "Initialization time: " << std::scientific << std::setprecision(6) << initialization_time << std::endl;
	
	// First iteration function value and merit value printing out 
	if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
	
	// First iteration time and value saving step 
	if (rank == 0) { times[iterations] = elapsed_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;} 
	
	// Timing start
	double start_time = MPI_Wtime();
	
	while (merit_value >= merit_tolerance && iterations < max_iterations && time_flag){
		
		// Solving inner problem for direction evaluation
		double xopt;
		for (int i = 0; i < n_local; i++) {  
			xopt = x[i] - nablaF[i]/L;
			double parameter = lambda/L;
			if (xopt >= parameter) xopt -= parameter;
			else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }
			x_new[i] = xopt;
			}

		// Liepschitz constant and step-size update
		t_old = t_new;
		t_new = (1 + std::sqrt(1 + 4*std::pow(t_old,2)))/2;
		double coefficient = (t_old-1)/t_new;
		 
		// Variables update
		x[0:n_local:1] = x_new[0:n_local:1] + coefficient*(x_new[0:n_local:1] - x_old[0:n_local:1]);
		x_old[0:n_local:1] = x_new[0:n_local:1];
        compute_F(x_old); 
        function_value = compute_function_value(x_old);
		merit_value = compute_merit_value(1);
		mse_value = compute_merit_value(0);
		compute_F(x); 
		compute_function_gradient();
		iterations ++;
			
		// Times and values saving step
		if (rank == 0) { times[iterations] = MPI_Wtime() - start_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;}  
		
		// Iteration, function value and merit value printing out
		if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
		
		// Time flag setting
		if (rank == 0 && (MPI_Wtime() - start_time) >= max_time) { time_flag = 0; }
		MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		
		}

	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	
	// Freeing memory
	delete [] x_old; delete [] x_new;

} 

void MRI::solve_sparsa(double merit_tolerance, int max_iterations, double max_time, double eta, double rho, int M){	
	
	// Alpha update rule - parameters setting
	double alpha = 1.0;
	double alpha_min = 1e-30;
	double alpha_max = 1e+30;
	
	// Variables definition memory allocation and initialization
	double communication_begin, initialization_begin;
	if (rank == 0) { times = new double[max_iterations+1]; values = new double[max_iterations+1]; merits = new double[max_iterations+1]; mse = new double[max_iterations+1]; }
	y = new double[nnz_local]; y_zf = new double [n_local](); y_big = new double [n];
	x = new double[n_local](); b = new double[nnz_local]; F = new double[nnz_local](); nablaF = new double[n_local](); 
	double *x_old = new double[n_local]; double *F_old = new double[nnz_local]; double *delta = new double[n_local];
	double *values_memory = new double[M];
	initialization_begin = MPI_Wtime();
	apply_mask(image, b);	
	F[0:nnz_local:1] = -b[0:nnz_local:1];
	compute_function_gradient();	
	function_value = compute_function_value(x);
	merit_value = compute_merit_value(1);
    mse_value = compute_merit_value(0);
	values_memory[0] = function_value;
    initialization_time += (MPI_Wtime() - initialization_begin);
    
    // Number of processes initialization time optimal function value printing out
    if (rank == 0) std::cout << "Number of parallel processes: " << size << std::endl;
	if (rank == 0) std::cout << "Initialization time: " << std::scientific << std::setprecision(6) << initialization_time << std::endl;
	
	// First iteration function value and merit value printing out 
	if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
	
	// First iteration time and value saving step 
	if (rank == 0) { times[iterations] = 0; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;} 
	
	// Timing start
	double start_time = MPI_Wtime();
	
	while (merit_value >= merit_tolerance && iterations < max_iterations && time_flag){
		
		// Memorizing actual point 
		x_old[0:n_local:1] = x[0:n_local:1]; 
		F_old[0:nnz_local:1] = F[0:nnz_local:1]; 
		
		// Solving scalar sub-problems with soft-tresholding for new point evaluation
		double xopt, max_value; double parameter_1; double parameter_local = 0;
		for (int i = 0; i < n_local; i++) {  
			xopt = x[i] - alpha*nablaF[i];
			double parameter = alpha*lambda;
			if (xopt >= parameter) xopt -= parameter;
			else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }
			delta[i] = x[i] - xopt; x[i] = xopt;
			parameter_local = parameter_local + delta[i]*delta[i]; 
			}
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allreduce(&parameter_local, &parameter_1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
		communication_time += (MPI_Wtime() - communication_begin);		
		max_value = *std::max_element(values_memory, values_memory + M);
		compute_F(x); 
		function_value = compute_function_value(x);
		merit_value = compute_merit_value(1);
		mse_value = compute_merit_value(0);
		iterations ++;

		// Soft-tresholding parameter and variables update
		parameter_local = 0; double parameter_2;
		int condition = function_value <= max_value - 0.5*parameter_1*rho/alpha;
		if (condition == 1 || iterations == 1){
			for (int i = 0; i < nnz_local; i++) parameter_local = parameter_local + (F_old[i] - F[i])*(F_old[i] - F[i]); 
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Allreduce(&parameter_local, &parameter_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
			communication_time += (MPI_Wtime() - communication_begin);		
			alpha = parameter_1/parameter_2;
			for (int i = 0; i < M-1; i++) values_memory[M-i-1] = values_memory[M-i-2]; values_memory[0] = function_value;
			if (condition == 0){
				x[0:n_local:1] = x_old[0:n_local:1];
				F[0:nnz_local:1] = F_old[0:nnz_local:1]; 
				}
			else{
				compute_function_gradient();
				}
			}
		else {
			alpha /= eta;
			x[0:n_local:1] = x_old[0:n_local:1];
			F[0:nnz_local:1] = F_old[0:nnz_local:1]; 
			}
		alpha = std::max(alpha,1/alpha_max);
		alpha = std::min(alpha,1/alpha_min);
			
		// Times and values saving step
		if (rank == 0) { times[iterations] = MPI_Wtime() - start_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;}  
		
		// Iteration, function value and merit value printing out
		if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
		
		// Time flag setting
		if (rank == 0 && (MPI_Wtime() - start_time) >= max_time) { time_flag = 0; }
		MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		
		}

	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	
	// Freeing memory
	delete [] delta; delete [] x_old; delete [] F_old; delete [] values_memory;

}

void MRI::solve_grock(double merit_tolerance, int max_iterations, double max_time, int mode, int P){	
	
	// Variables definition memory allocation and initialization
	double communication_begin, initialization_begin;
	if (rank == 0) { times = new double[max_iterations+1]; values = new double[max_iterations+1]; merits = new double[max_iterations+1]; mse = new double[max_iterations+1];}
	y = new double[nnz_local]; y_zf = new double [n_local](); y_big = new double [n];
	x = new double[n_local](); b = new double[nnz_local]; F = new double[nnz_local](); nablaF = new double[n_local](); 
	double *x_old = new double[n_local]; double *delta = new double[n_local]; double *max_values = new double[size]();
	initialization_begin = MPI_Wtime();
	apply_mask(image, b);	
	F[0:nnz_local:1] = -b[0:nnz_local:1];
	compute_function_gradient();
	function_value = compute_function_value(x);
	merit_value = compute_merit_value(1);
    mse_value = compute_merit_value(0);
    initialization_time += (MPI_Wtime() - initialization_begin);
    if (mode == 0) srand (time(NULL));
    
    // Number of processes initialization time optimal function value printing out
    if (rank == 0) std::cout << "Number of parallel processes: " << size << std::endl;
	if (rank == 0) std::cout << "Initialization time: " << std::scientific << std::setprecision(6) << initialization_time << std::endl;
	
	// First iteration function value and merit value printing out 
	if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
	
	// First iteration time and value saving step 
	if (rank == 0) { times[iterations] = 0; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;} 
	
	// Timing start
	double start_time = MPI_Wtime();
	
	while (merit_value >= merit_tolerance && iterations < max_iterations && time_flag){
		
		// Memorizing actual point 
		x_old[0:n_local:1] = x[0:n_local:1];
		
		// Solving scalar sub-problems with soft-tresholding for new point evaluation
		double xopt; int max_index = 0; double max_value = 0;
		for (int i = 0; i < n_local; i++) {  
			xopt = x[i] - nablaF[i]/(1.0*nnz/n);
			double parameter = lambda/(1.0*nnz/n);
			if (xopt >= parameter) xopt -= parameter;
			else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }
			delta[i] = xopt - x[i]; 
			double temp_value = std::abs(delta[i]);
			if (temp_value > max_value) { max_value = temp_value; max_index = i; }
			}

		// Variables update		
		if (mode == 0 && P < size) {
			int *indices = new int[P]; 
			for (int i = 0; i < size; i++) indices[i] = i;
			for (int i = 0; i < size; i++) {
				int one = std::rand() % size;
				int two = std::rand() % size;
				int tmp = indices[one];
				indices[one] = indices[two];
				indices[two] = tmp;
				}
			for (int i = 0; i < P; i++) { if (rank == indices[i]) x[max_index] = x_old[max_index] + delta[max_index]; }
			delete [] indices;
			}
		if (mode == 1 && P < size){
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Gather(&max_value, 1, MPI_DOUBLE, max_values, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			if (rank == 0) std::sort(max_values, max_values+size);
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Bcast(max_values, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			for (int i = 0; i < P; i++) { if (max_values[size - i -1] == max_value) x[max_index] = x_old[max_index] + delta[max_index]; }
			}
		if (P == size) { x[max_index] = x_old[max_index] + delta[max_index]; }
		compute_F(x); 
		function_value = compute_function_value(x);
		merit_value = compute_merit_value(1);
		mse_value = compute_merit_value(0);		
		compute_function_gradient();
		iterations ++;
		
		// Times and values saving step
		if (rank == 0) { times[iterations] = MPI_Wtime() - start_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;}  
		
		// Iteration, function value and merit value printing out
		if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
		
		// Time flag setting
		if (rank == 0 && (MPI_Wtime() - start_time) >= max_time) { time_flag = 0; }
		MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		
		}

	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	
	// Freeing memory
	delete [] delta; delete [] x_old; delete [] max_values;

}

void MRI::solve_barista(double merit_tolerance, int max_iterations, double max_time){	
	
	// Alpha parameter
	double alpha = -0.173648178;
	
	// Tau update rule - parameter settings (proximal term weight)
	double tau_old; double tau = 1;
	
	// Variables definition memory allocation and initialization
	double communication_begin, initialization_begin; 
	if (rank == 0) { times = new double[max_iterations+1]; values = new double[max_iterations+1]; merits = new double[max_iterations+1]; mse = new double[max_iterations+1];}
	double *z = new double[n_local](); double *D = new double[n_local/2];
	x = new double[n_local](); y = new double[nnz_local](); 
	y_zf = new double [n_local](); y_big = new double [n];
	b = new double[nnz_local]; F = new double[nnz_local](); nablaF = new double[n_local](); 
	double *x_old = new double[n_local];
	initialization_begin = MPI_Wtime();
	std::fill(D, D+n_local/2, 1.0);
	apply_mask(image, b);	
	F[0:nnz_local:1] = -b[0:nnz_local:1];
	compute_function_gradient();
	function_value = compute_function_value(x);
	merit_value = compute_merit_value(1);
    mse_value = compute_merit_value(0);
    initialization_time += (MPI_Wtime() - initialization_begin);
	
    // Number of processes initialization time optimal function value printing out
    if (rank == 0) std::cout << "Number of parallel processes: " << size << std::endl;
	if (rank == 0) std::cout << "Initialization time: " << std::scientific << std::setprecision(6) << initialization_time << std::endl;
	
	// First iteration function value and merit value printing out 
	if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
	
	// First iteration time and value saving step 
	if (rank == 0) { times[iterations] = elapsed_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;} 
	
	/*************************ALGORITHM BEGIN**************************/
	// Timing start
	double start_time = MPI_Wtime();
		
	while (merit_value >= merit_tolerance && iterations < max_iterations && time_flag){
		
		// Memorizing actual point 
		x_old[0:n_local:1] = x[0:n_local:1];  
		
		// Tau update
		tau_old = tau;
		tau = (1 + std::sqrt(1 + 4.0*tau_old*tau_old))/2;	
		
		// Solving scalars sub-problems with soft-tresholding	
		double xopt;		
		double left_local, right_local_1, right_local_2, right_local, left, right;
		left_local = 0; right_local = 0;
		for (int i = 0; i < n_local; i++) {
			xopt = z[i] - D[i/2]*nablaF[i];	
			double parameter = lambda*D[i/2];
			if (xopt >= parameter) xopt -= parameter;
			else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }
			x[i] = xopt;
			right_local_1 = right_local_1 +  (z[i] - x[i])*(z[i] - x[i]);
			right_local_2 = right_local_2 +  (x_old[i] - x[i])*(x_old[i] - x[i]);
			left_local = left_local + (z[i] - x[i])*(x[i] - x_old[i]);	
			}	
		right_local = std::sqrt(right_local_1*right_local_2);
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allreduce(&right_local, &right, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&left_local, &left, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		communication_time += (MPI_Wtime() - communication_begin);
		if(left > alpha*right){
			z[0:n_local:1] = x[0:n_local:1];
			tau = 1;
			}
		else z[0:n_local:1] = x[0:n_local:1] + ((tau_old-1)/tau)*(x[0:n_local:1] - x_old[0:n_local:1]);
		
		// Useful computations
		compute_F(x);
		function_value = compute_function_value(x);		
		mse_value = compute_merit_value(0);		
		merit_value = compute_merit_value(1);
		compute_F(z);
		compute_function_gradient();	
		iterations ++;
	
		// Times and values saving step
		if (rank == 0) { times[iterations] = MPI_Wtime() - start_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;}  
		
		// Iteration, function value and merit value printing out
		if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
		
		// Time flag setting
		if (rank == 0 && (MPI_Wtime() - start_time) >= max_time) { time_flag = 0; }
		MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		
		}
	
	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	/**************************ALGORITHM END***************************/
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
			
	// Freeing memory
	delete [] x_old; delete [] D; delete [] z;
	
}

void MRI::solve_ktfocuss(double merit_tolerance, int max_iterations, double max_time, int inner_loop_limit, int downsampling_factor, double rho){
	
	// Variables definition memory allocation and initialization
	double communication_begin, initialization_begin;
	if (rank == 0) { times = new double[max_iterations+1]; values = new double[max_iterations+1]; merits = new double[max_iterations+1]; mse = new double[max_iterations+1];}
	y = new double[nnz_local]; y_zf = new double [n_local](); y_big = new double [n];
	x = new double[n_local](); b = new double[nnz_local]; F = new double[nnz_local](); nablaF = new double[n_local](); 
	double *weights_matrix = new double [n_local/2](); double max_element_local, max_element;
	double *hybrid_image = new double [n_local];
	double *delta = new double[nnz_local]; double *delta_zf = new double[n_local];
	y = new double[nnz_local]; y_zf = new double [n_local]();
	double *direction = new double[n_local];
	double *x_new = new double[n_local]();
	double alpha, beta;
	double norm_old, norm_new;
	double t1_real, t2_real, t3, t4;
	initialization_begin = MPI_Wtime();
	apply_mask(image, b); zero_filling(b, y_zf); zero_filling(b, hybrid_image); 
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(hybrid_image+2*i*Kx*t));
	status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)y_zf);	
	for (int i = 0; i < Ky_local; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_zf+2*i*Kx*t));
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allgather(y_zf, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	y_zf[0:n_local:1] = y_big[rank*n_local:n_local:1]; 
	for (int i = 0; i < n_local/2; i++){ weights_matrix[i] = std::sqrt(y_zf[2*i]*y_zf[2*i] + y_zf[2*i+1]*y_zf[2*i+1]); }
	vdPowx(n_local/2, weights_matrix, rho, weights_matrix);
	max_element_local = *std::max_element(weights_matrix, weights_matrix + n_local/2);
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allreduce(&max_element_local, &max_element, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	if (max_element_local != max_element) max_element_local = 0;  
	MPI_Barrier(MPI_COMM_WORLD);
	communication_begin = MPI_Wtime();
	MPI_Allreduce(&max_element_local, &max_element, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	communication_time += (MPI_Wtime() - communication_begin);
	weights_matrix[0:(n_local/2):1] = (1.0/max_element)*weights_matrix[0:(n_local/2):1];	
	F[0:nnz_local:1] = -b[0:nnz_local:1];
	compute_function_gradient();
	function_value = compute_function_value(x);
	merit_value = compute_merit_value(1);
    mse_value = compute_merit_value(0);
    initialization_time += (MPI_Wtime() - initialization_begin);
    	
    // Number of processes initialization time optimal function value printing out
    if (rank == 0) std::cout << "Number of parallel processes: " << size << std::endl;
	if (rank == 0) std::cout << "Initialization time: " << std::scientific << std::setprecision(6) << initialization_time << std::endl;
	
	// First iteration function value and merit value printing out 
	if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
	
	// First iteration time and value saving step 
	if (rank == 0) { times[iterations] = elapsed_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;} 
	
	/*************************ALGORITHM BEGIN**************************/
	// Timing start
	double start_time = MPI_Wtime();
	
	while (merit_value >= merit_tolerance && iterations < max_iterations && time_flag){
		
		for (int inner_loop_counter = 0; inner_loop_counter < inner_loop_limit; inner_loop_counter++){
			
			// Inner loop main computations
			if (inner_loop_counter == 0){
				int p = 0;
				for (int i = 0; i < n_local/2; i++){
					if (mask[i]) {
						delta[p] =  hybrid_image[2*i];
						delta[p+1] =  hybrid_image[2*i+1];								
						delta_zf[2*i] =  hybrid_image[2*i];
						delta_zf[2*i+1] =  hybrid_image[2*i+1];
						p += 2;
						} 
					else {
						delta_zf[2*i] = 0;
						delta_zf[2*i+1] = 0;
						}	
					}
				}
			else{
				status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)x);	
				MPI_Barrier(MPI_COMM_WORLD);
				communication_begin = MPI_Wtime();
				MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
				communication_time += (MPI_Wtime() - communication_begin);
				status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
				x[0:n_local:1] = y_big[rank*n_local:n_local:1]; 
				int p = 0;
				for (int i = 0; i < n_local/2; i++){
					if (mask[i]) {
						delta[p] =  hybrid_image[2*i] - x[2*i];
						delta[p+1] =  hybrid_image[2*i+1] - x[2*i+1];								
						delta_zf[2*i] =  delta[p];
						delta_zf[2*i+1] =  delta[p+1];
						p += 2;
						} 
					else {
						delta_zf[2*i] = 0;
						delta_zf[2*i+1] = 0;
						}	
					}
				}
			status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)delta_zf);	
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Allgather(delta_zf, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
			delta_zf[0:n_local:1] = y_big[rank*n_local:n_local:1];	
			for (int i = 0; i < n_local/2; i++){
				delta_zf[2*i] = delta_zf[2*i]*weights_matrix[i];
				delta_zf[2*i+1] = delta_zf[2*i+1]*weights_matrix[i];
				}
			delta_zf[0:n_local:1] = lambda*x_new[0:n_local:1] - delta_zf[0:n_local:1];
			double norm_new_local = 0;
			for (int i = 0; i < n_local; i++){ norm_new_local = norm_new_local + delta_zf[i]*delta_zf[i]; }
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Allreduce(&norm_new_local, &norm_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			if (inner_loop_counter == 0){
				direction[0:n_local:1] = -delta_zf[0:n_local:1];
				}
			else{
				beta = norm_new/norm_old;
				direction[0:n_local:1] = beta*direction[0:n_local:1] - delta_zf[0:n_local:1];
				}		
			norm_old = norm_new;
			for (int i = 0; i < n_local/2; i++){
				y_zf[2*i] = delta_zf[2*i]*weights_matrix[i];
				y_zf[2*i+1] = delta_zf[2*i+1]*weights_matrix[i]; 
				}
			status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)y_zf);	
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Allgather(y_zf, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
			y_zf[0:n_local:1] = y_big[rank*n_local:n_local:1];	
			apply_mask(y_zf, y);
			double t1_local = 0; for (int i = 0; i < nnz_local/2; i++){ t1_local = t1_local + delta[2*i]*y[2*i] + delta[2*i+1]*y[2*i+1]; }
			double t2_local = 0; for (int i = 0; i < n_local/2; i++){ t2_local = t2_local + delta_zf[2*i]*x_new[2*i] + delta_zf[2*i+1]*x_new[2*i+1];; }
			double t3_local = 0; for (int i = 0; i < nnz_local; i++){ t3_local = t3_local + y[i]*y[i]; }
			double t4_local = 0; for (int i = 0; i < n_local; i++){ t4_local = t4_local + delta_zf[i]*delta_zf[i]; }
			MPI_Barrier(MPI_COMM_WORLD);
			communication_begin = MPI_Wtime();
			MPI_Allreduce(&t1_local, &t1_real, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&t2_local, &t2_real, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&t3_local, &t3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&t4_local, &t4, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			communication_time += (MPI_Wtime() - communication_begin);
			alpha = (t1_real - lambda*t2_real)/(t3 + lambda*t4);
			x_new[0:n_local:1] = alpha*delta_zf[0:n_local:1] + x_new[0:n_local:1];
			for (int i = 0; i < n_local/2; i++){
				x[2*i] = x_new[2*i]*weights_matrix[i];
				x[2*i+1] = x_new[2*i+1]*weights_matrix[i];; 
				}
			
			// Useful quantities computation
			compute_F(x); compute_function_gradient();
			function_value = compute_function_value(x);		
			mse_value = compute_merit_value(0);		
			merit_value = compute_merit_value(1);
			iterations ++;

			// Times and values saving step
			if (rank == 0) { times[iterations] = MPI_Wtime() - start_time; values[iterations] = function_value; merits[iterations] = merit_value; mse[iterations] = mse_value;}  
		
			// Iteration, function value and merit value printing out
			if (rank == 0 && verbose) std::cout << "Iteration: " << iterations << "  Value: " << function_value << "  Merit: " << merit_value << std::endl;
		
			// Time flag setting
			if (rank == 0 && (MPI_Wtime() - start_time) >= max_time) { time_flag = 0; }
			MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
			
			} 
		
		// Window elements update
		for (int i = 0; i < n_local/2; i++){ weights_matrix[i] = std::sqrt(x[2*i]*x[2*i] + x[2*i+1]*x[2*i+1]); }
		vdPowx(n_local/2, weights_matrix, rho, weights_matrix);
		max_element_local = *std::max_element(weights_matrix, weights_matrix + n_local/2);
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allreduce(&max_element_local, &max_element, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		communication_time += (MPI_Wtime() - communication_begin);
		if (max_element_local != max_element) max_element_local = 0;  
		MPI_Barrier(MPI_COMM_WORLD);
		communication_begin = MPI_Wtime();
		MPI_Allreduce(&max_element_local, &max_element, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		communication_time += (MPI_Wtime() - communication_begin);
		weights_matrix[0:(n_local/2):1] = (1.0/max_element)*weights_matrix[0:(n_local/2):1];
		
		}
	
	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	/**************************ALGORITHM END***************************/
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	MPI_Allgather(x, n_local, MPI_DOUBLE, y_big, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
		
	// Freeing memory
	delete [] delta; delete [] delta_zf; 
	delete [] direction; delete [] x_new;
	
}

void MRI::save_results(){
	
	system("mkdir results"); 
	std::string folder_1; std::string folder_2; std::string algorithm_string;
	if(problem_solver == 0) algorithm_string = "flexa";
	if(problem_solver == 1) algorithm_string = "fista";
	if(problem_solver == 2) algorithm_string = "sparsa";
	if(problem_solver == 3) algorithm_string = "grock";
	if(problem_solver == 4) algorithm_string = "focuss";
	if(problem_solver == 5) algorithm_string = "barista";  
	folder_1 = "mkdir results/" + algorithm_string; 
	system(folder_1.c_str());
	folder_2 = "results/" + algorithm_string;
	std::string ciao_1; ciao_1 = folder_2 + "/comm.dat";
	std::string ciao_2; ciao_2 = folder_2 + "/init.dat";
	std::string ciao_3; ciao_3 = folder_2 + "/times.dat";
	std::string ciao_4; ciao_4 = folder_2 + "/values.dat";
	std::string ciao_5; ciao_5 = folder_2 + "/merits.dat";
	std::string ciao_6; ciao_6 = folder_2 + "/mse.dat";
	std::string ciao_7; ciao_7 = folder_2 + "/image.dat";
	std::ofstream tempi, valori, meriti, errore, immagine, comunicazione, inizializzazione;
	comunicazione.open(ciao_1.c_str());
	inizializzazione.open(ciao_2.c_str());
	comunicazione << std::scientific << std::setprecision(10) << communication_time;
	inizializzazione << std::scientific << std::setprecision(10) << initialization_time; 
	comunicazione.close();
	inizializzazione.close();
	tempi.open(ciao_3.c_str());
	valori.open(ciao_4.c_str());
	meriti.open(ciao_5.c_str());
	errore.open(ciao_6.c_str());
	for (int i = 0; i < iterations + 1; i++){
			tempi << times[i] << std::endl;
			valori << std::scientific << std::setprecision(10) << values[i] << std::endl;
			meriti << std::scientific << std::setprecision(10) << merits[i] << std::endl;	
			errore << std::scientific << std::setprecision(10) << mse[i] << std::endl;
			} 
    tempi.close();
	valori.close();
	meriti.close();
	errore.close();
	immagine.open(ciao_7.c_str());
	for (int i = 0; i < n; i++){
			immagine << y_big[i] << std::endl;
			}
	immagine.close();
		
}
