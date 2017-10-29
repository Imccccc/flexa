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

/*
"host": "snyder-fe01.rcac.purdue.edu",
    "user": "yang750",
    "password": "puducode7",
*/

// PROBLEM SOLVER SELECTION (0 -> FLEXA, 1 -> FISTA, 2 -> SPARSA, 3 -> GROCK, 4 -> KT-FOCUSS, 5 -> BARISTA)
int problem_solver = 0;

int main(int argc, char **argv) {

  // MPI environment initialization
  int rank = 0, size = 1;
  // MPI::Init(argc,argv);
  // rank = MPI::COMM_WORLD.Get_rank();
  // size = MPI::COMM_WORLD.Get_size();
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
}
