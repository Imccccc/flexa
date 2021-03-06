#ifndef MRI_H
#define MRI_H
	
	#include <string>
	#include "mkl.h"
	
	class MRI{
		private:
			// Variables
			int problem_solver;
			int	rank, size;
			double lambda;
			int Kx, Ky, t, n, nnz;
			double *y, *y_zf, *y_big, *x, *b, *F, *nablaF;
			int *mask; double *image, *reference_image; 
			DFTI_DESCRIPTOR_HANDLE dft_descriptor_handle_Kx; MKL_LONG status_Kx;
			DFTI_DESCRIPTOR_HANDLE dft_descriptor_handle_Ky; MKL_LONG status_Ky;
			DFTI_DESCRIPTOR_HANDLE dft_descriptor_handle_t; MKL_LONG status_t;	
			bool verbose;
			int time_flag;
			int iterations;
			double *times, *values, *merits, *mse;
			double initialization_time, communication_time, elapsed_time; 
			double F_value, function_value, merit_value, mse_value, normalizing_factor;
			// Methods
			void compute_F(double *x);
			void compute_function_gradient();
			double compute_function_value(double *x);
			double compute_merit_value(int merit_function);
			void apply_mask(double *input, double *output);
			void zero_filling(double *input, double *output);
			void solve_subproblem_ST_J(double rho, double M_global, double gamma, double *delta, int update_strategy, double *local_function_value);
		public:
			// Methods
			MRI(int rank, int size, std::string data_directory, double lambda, bool verbose, int problem_solver);
			~MRI();
			void solve_flexa(double merit_tolerance, int max_iterations, double max_time, int solver, int update_strategy, double tau, double gamma, double rho);		
			void save_results();
	
		};

#endif
