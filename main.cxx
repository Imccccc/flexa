/***********************************************************************
 Written by: Paolo Scarponi (paolosca@buffalo.edu)
***********************************************************************/

#include <mpi.h>
#include "mri.h"
#include <iostream>
#include <string>
#include <sstream>
#include <list>
#include <cmath>
#include <iomanip>

// PROBLEM SOLVER SELECTION (0 -> FLEXA, 1 -> FISTA, 2 -> SPARSA, 3 -> GROCK, 4 -> KT-FOCUSS, 5 -> BARISTA)
int problem_solver = 0;

int main(int argc, char **argv) {

  // MPI environment initialization
  int rank, size;
  MPI::Init(argc,argv);
  rank = MPI::COMM_WORLD.Get_rank();
  size = MPI::COMM_WORLD.Get_size();
  std::cout << "Rank: " << rank << "  Size: " << size << std::endl;
  /**************************Object parameters*************************/
  // Verbosity parameter setting
  bool verbose = true;
  /********************************************************************/
  
  /**************************Stopping criteria*************************/
  // Merit value to be achieved
  double merit_tolerance = 1e-6;
  // Maximum number of iterations
  int max_iterations = 40*5;
  // Maximum running time
  double max_time = 100;
  /********************************************************************/
  
  /***************************Data parameters**************************/
  // Data directory
  std::string datafolder = "../data"; 
  // Regularization parameter
  double lambda = 0.000001;
  /********************************************************************/

  // Algorithm object creation
  MRI *algorithm = new MRI(rank, size, datafolder, lambda, verbose, problem_solver);

  /////////////////////////////////FLEXA////////////////////////////////
  if (problem_solver == 0){
	
	/***********************Algorithm parameters***********************/
	// Solving method selection (0 -> Jacobi, 1 -> Gauss-Seidel)
	int solver = 0;
	// Updating strategy selection (0 -> all blocks, 1 -> one block)
	int update_strategy = 0;
	// Tau initial value
	double tau = 0.5;
	// Gamma initial value
	double gamma = 0.99;
	// Greedy selection parameter value
	double rho = 0.1;
	/******************************************************************/

	// Running algorithm
	algorithm->solve_flexa(merit_tolerance, max_iterations, max_time, solver, update_strategy, tau, gamma, rho);

    }
  //////////////////////////////////////////////////////////////////////

  /////////////////////////////////FISTA////////////////////////////////
  if (problem_solver == 1){
	
	/***********************Algorithm parameters***********************/
	// Power method tolerance parameter
	double power_method_tolerance = 1e-6;
	int power_method_max_iterations = 100;
	/******************************************************************/
		
	// Running algorithm
	algorithm->solve_fista(merit_tolerance, max_iterations, max_time, power_method_tolerance, power_method_max_iterations);
	
    }
  //////////////////////////////////////////////////////////////////////

  ////////////////////////////////SPARSA////////////////////////////////
  if (problem_solver == 2){
	
	/***********************Algorithm parameters***********************/
	// Line-search parameters
	int M = 5;
	double eta = 2.0;
	double rho = 0.01;
	/******************************************************************/
	
	// Running algorithm
	algorithm->solve_sparsa(merit_tolerance, max_iterations, max_time, eta, rho, M);
	
	}
  //////////////////////////////////////////////////////////////////////

  ////////////////////////////////GROCK/////////////////////////////////
  if (problem_solver == 3){
	
	/***********************Algorithm parameters***********************/
	// Algorithm mode selection (0 -> random update, 1 -> greedy update) 
	int mode = 1;
	// Active processors number
	int N = size;
	/******************************************************************/
		
	// Running algorithm
	algorithm->solve_grock(merit_tolerance, max_iterations, max_time, mode, N);
		
	}
  //////////////////////////////////////////////////////////////////////

  //////////////////////////////KT-FOCUSS///////////////////////////////
  if (problem_solver == 4){
	
	/***********************Algorithm parameters***********************/
	// Line-search parameters
	int inner_loop_limit = 40;
	int downsampling_factor = 8;
	double rho = 0.5;
	/******************************************************************/
	
	// Running algorithm
	algorithm->solve_ktfocuss(merit_tolerance, max_iterations, max_time, inner_loop_limit, downsampling_factor, rho);

	}
  //////////////////////////////////////////////////////////////////////

  ///////////////////////////////BARISTA////////////////////////////////
  if (problem_solver == 5){
	
	// Running algorithm
	algorithm->solve_barista(merit_tolerance, max_iterations, max_time);

    }
  //////////////////////////////////////////////////////////////////////

  // Saving results
  if (rank == 0) algorithm->save_results();

  // Alogrithm object destruction
  delete algorithm; 

  // MPI environment finalization
  MPI::Finalize();
  
  // Program finalization
  return 0;
  
}
