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
		// Read random sampling mask
		mask = new int[n/2];
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/mask.dat");
		if (rank == 0) printf("[ALL] reading random sampling mask: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		nnz = 0; p = 0; 
		int temp; 
		for (int i = 0; i < n/2; i++) { if (i >= rank*(n/2) && i < (rank+1)*(n/2)) { file >> mask[p]; p += 1; if(mask[p-1] == 1) n += 2; } else file >> bin; }
		if (rank==0) printf("DONE \n");
		file.close();	
		// Read Kx-Ky-t domain data
		image = new double[n];
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/data_kx_ky_t.dat");
		if (verbose && rank == 0) printf("[ALL] reading kx-ky-t domain image: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		p = 0; for (int i = 0; i < n/2; i++) { if (i >= rank*(n/2) && i < (rank+1)*(n/2)){ file >> image[p]; file >> image[p+1]; p += 2; } else{ file >> bin; file >> bin; } } 
		if (verbose && rank == 0) printf("DONE \n");
		file.close();
		// Read x-y-f domain data
		reference_image = new double[n];
		strcpy(data_path,data_directory.c_str()); strcat(data_path,"/data_x_y_f.dat");
		if (verbose && rank == 0) printf("[ALL] reading x-y-f domain reference image: ");
		file.open(data_path);
		if (!file && rank == 0) { std::cout << "ERROR" << std::endl; throw std::exception(); }
		p = 0; for (int i = 0; i < n/2; i++) { if (i >= rank*(n/2) && i < (rank+1)*(n/2)){ file >> reference_image[p]; file >> reference_image[p+1]; p += 2; } else{ file >> bin; file >> bin; } }
		if (verbose && rank == 0) printf("DONE \n");
		file.close();		

	// Initialize Fast Fourier Transforms
		// FFTs dimensions setting
		MKL_LONG stride_Kx[2]; stride_Kx[0] = 0; stride_Kx[1] = t;
		MKL_LONG stride_Ky[2]; stride_Ky[0] = 0; stride_Ky[1] = Kx*t;
		MKL_LONG stride_t[2]; stride_t[0] = 0; stride_t[1] = 1;	
		int number_of_transforms_Kx = t;
		int number_of_transforms_Ky = t*Kx;
		int number_of_transforms_t = Kx*Ky;
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
	// Compute MSE normalizing factor
	for (int i = 0; i < n; i++) normalizing_factor = normalizing_factor + reference_image[i]*reference_image[i]; 
		
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
	for (int i = 0; i < n/2; i++){
		if (mask[i]==1) {
			output[p] = input[2*i];
			output[p+1] = input[2*i+1];
			p += 2;
			}
		}
		
}

void MRI::zero_filling(double *input, double *output){

	int p = 0;
	for (int i = 0; i < n/2; i++){
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
	y_big[0:n:1] = x[0:n:1];  
	status_t = DftiComputeBackward(dft_descriptor_handle_t, (std::complex<double>*)y_big);	
	for (int i = 0; i < Ky; i++) status_Kx = DftiComputeForward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_big+2*i*Kx*t));
	status_Ky = DftiComputeForward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	y_zf[0:n:1] = y_big[rank*n:n:1];
	apply_mask(y_zf, y);
	F[0:nnz:1] = y[0:nnz:1] - b[0:nnz:1]; 
}

void MRI::compute_function_gradient(){
	zero_filling(F,y_big);
	status_t = DftiComputeForward(dft_descriptor_handle_t, (std::complex<double>*)y_big);	
	for (int i = 0; i < Ky; i++) status_Kx = DftiComputeBackward(dft_descriptor_handle_Kx, (std::complex<double>*)(y_big+2*i*Kx*t));
	status_Ky = DftiComputeBackward(dft_descriptor_handle_Ky, (std::complex<double>*)y_big);
	nablaF[0:n:1] = y_big[rank*n:n:1]; 
}

double MRI::compute_function_value(double *x){	
	double regularization_term = 0; double value;
	for (int i = 0; i < nnz; i++) value = value+ F[i]*F[i]; value *= 0.5;
	for (int i = 0; i < n; i++) regularization_term = regularization_term + std::abs(x[i]);
		regularization_term *= lambda;
	value = value + regularization_term; 
	return value;
}

double MRI::compute_merit_value(int merit_function){
	double merit;
	if (merit_function == 0){
		for (int i = 0; i < n; i++) merit = merit + ((t/std::sqrt(Kx*Ky*t))*x[i] - reference_image[i])*((t/std::sqrt(Kx*Ky*t))*x[i] - reference_image[i]); 
		merit /= normalizing_factor;
		}
	else{
		for (int i = 0; i < n; i++) {
			double Fi = nablaF[i] - x[i];	
			if (Fi >= lambda) Fi = lambda;
			if (Fi <= -lambda) Fi = -lambda;
			double distance = std::abs(nablaF[i]-Fi);
			merit = std::max(merit, distance);
			}
		}
	return merit;	
		
}

void MRI::solve_subproblem_ST_J(double rho, double M_global, double gamma, double *delta, int update_strategy, double *local_function_value){
	
	// Variables definition
	double direction, xnew, Fnew;
	
	// Loop begin
	for (int i = 0; i < n; i++) { 
		if (std::abs(delta[i]) >= rho*M_global){ 
			// Direction evaluation
			direction = -delta[i];
			x[i] = x[i] - gamma*direction;	
			}
		}
		
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
	y = new double[nnz]; y_zf = new double [n](); y_big = new double [n];
	x = new double[n](); b = new double[nnz]; F = new double[nnz](); nablaF = new double[n](); 
	double *x_old = new double[n]; double *delta = new double[n]; double function_value_old;
	initialization_begin = MPI_Wtime();
	apply_mask(image, b);	
	F[0:nnz:1] = -b[0:nnz:1];
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
		x_old[0:n:1] = x[0:n:1];  
		
		// Solving scalars sub-problems with soft-tresholding for greedy variables selection	
		double M_global; double xopt;		
		for (int i = 0; i < n; i++) {
			xopt = x[i] - nablaF[i]/(1.0*nnz/n + tau);	
			double parameter = lambda/(1.0*nnz/n + tau);
			if (xopt >= parameter) xopt -= parameter;
			else { if (xopt <= -parameter) xopt += parameter; else xopt = 0; }
			delta[i] = xopt - x[i];
			double distance = std::abs(xopt - x[i]);
			if (distance > M_global) M_global = distance;
			}
		
		// Useful variables initialization for Single-Block-Update schemes
		double local_function_value = 0; double min_function_value; 
		if(update_strategy && size > 1) local_function_value = (function_value - F_value)/lambda;
		
		// Solving scalars sub-problems for direction evaluation
		if (solver == 0) solve_subproblem_ST_J(rho, M_global, gamma, delta, update_strategy, &local_function_value);

		compute_F(x);
		function_value = compute_function_value(x);		
		iterations ++;

		// Tau update and next loop useful computations
		if (function_value >= function_value_old) {
			function_decrements = 0; tau_increments++; tau /= tau_step;
			x[0:n:1] = x_old[0:n:1];
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
		}
	
	// Timing end
	elapsed_time = MPI_Wtime() - start_time;
	/**************************ALGORITHM END***************************/
	
	// Communication time and elapsed time printing out
	if(rank == 0) std::cout << "Communication time: " << std::scientific << std::setprecision(6) << communication_time << std::endl; 
	if(rank == 0) std::cout << "Elapsed time: " << std::scientific << std::setprecision(6) << elapsed_time << std::endl;
	
	// Save last reconstructed image
	y_big[0:n:1] = x[0:n:1];  

	// Freeing memory
	delete [] delta; delete [] x_old;
	
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
