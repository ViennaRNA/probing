#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>
#include "fold.h"
#include "part_func.h"
#include "fold_vars.h"
#include "PS_dot.h"
#include "utils.h"
#include "read_epars.h"
#include "MEA.h"
#include "RNApbfold_cmdl.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas_types.h>
#include <gsl/gsl_blas.h>


typedef struct {
  int length;
  char* seq;
  double tau;
  double kT;
  double sigma;
} minimizer_pars_struct;



PRIVATE double calculate_f   (const gsl_vector *v, void *params);
PRIVATE void   calculate_df  (const gsl_vector *v, void *params, gsl_vector *df);
PRIVATE void   calculate_df_numerically  (const gsl_vector *v, void *params, gsl_vector *df);
PRIVATE void   calculate_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df);

PRIVATE void get_pair_prob_vector(double** matrix, double* vector, int length, int type); 
PRIVATE double calculate_norm (double* vector, int length);
PRIVATE void print_dotplot(char* seq, char* struc, int length, int iteration, double D);
PRIVATE char* print_mea_string(FILE* statsfile, char* seq, int length);

PRIVATE void test_folding(char* seq, int length);
PRIVATE void test_stochastic_backtracking(char* seq, int length);
PRIVATE void test_gradient_sampling(gsl_multimin_function_fdf minimizer_func,  minimizer_pars_struct minimizer_pars);
PRIVATE void test_gradient(gsl_multimin_function_fdf minimizer_func,  minimizer_pars_struct minimizer_pars);

PRIVATE struct plist *b2plist(const char *struc);
PRIVATE struct plist *make_plist(int length, double pmin);

PUBLIC double *epsilon;             /* Perturbation vector in kcal/Mol */
PRIVATE double *q_unpaired;         /* Vector of probs. of being unpaired in the experimental structure */
PRIVATE double **p_pp;              /* Base pair probability matrix of predicted structure              */
PRIVATE double *p_unpaired;         /* Vector of probs. of being unpaired in the predicted structure    */
PRIVATE double **p_unpaired_cond;   /* List of vectors p_unpaired with the condition that i is unpaired */
PRIVATE double **p_unpaired_cond_sampled;  
PRIVATE int count_df_evaluations;

PRIVATE int  numerically = 0;
PRIVATE double numeric_d;
PRIVATE int sample_conditionals = 0;

int debug=0;


int main(int argc, char *argv[]){

  struct        RNAfold_args_info args_info;
  char          *string, *input_string, *structure=NULL, *cstruc=NULL;
  char          fname[80], ffname[80], gfname[80], *ParamFile=NULL;
  char          *ns_bases=NULL, *c;
  int           i, j, ii, jj, mu, length, l, sym, r, pf=0, noPS=0, noconv=0;
  unsigned int  input_type;
  double        energy, min_en, kT, sfact=1.07;
  int           doMEA=0, circular = 0, N;
  char *pf_struc;
  double dist;
  plist *pl;

  FILE * filehandle;
  FILE * statsfile;
  char* line;

  double tau   = 0.01; /* Variance of energy parameters */
  double sigma = 0.01; /* Variance of experimental constraints */
  double *gradient;           /* Gradient for steepest descent search
                                 epsilon[i+1]= epsilon[i] - gradient *
                                 step_size */
  double initial_step_size = 0.5;  /* Initial step size for steepest
                                      descent search */
  double step_size;                
  double D;                  /* Discrepancy (i.e. value of objective
                                function) for the current
                                prediction */
  int iteration, max_iteration = 500; /* Current and maximum number of
                                         iterations after which
                                         algorithm stops */

  double precision = 0.1; /* cutoff used for stop conditions */
  double tolerance = 0.1;   /* Parameter used by various GSL minimizers */
  int method_id = 0;        /* Method to use for minimization, 0 and 1
                               are custom steepest descent, the rest
                               are GSL implementations (see below)*/

  int sample_N = 1000;

  double *prev_epsilon;
  double *prev_gradient;
  double DD, prev_D, sum, norm; 
  int status;
  double* gradient_numeric;
  double* gradient_numeric_gsl;


  /* Minimizer vars */
  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *minimizer;
  gsl_vector *minimizer_x;
  gsl_vector *minimizer_g;
  gsl_multimin_function_fdf minimizer_func;
  minimizer_pars_struct minimizer_pars;

  char *constraints;

  char outfile[256];

  FILE* fh;
  
  do_backtrack  = 1;
  string        = NULL;

  outfile[0]='\0';

  if(RNAfold_cmdline_parser (argc, argv, &args_info) != 0) exit(1);

  /* RNAbpfold specific options */
  
  if (args_info.tau_given) tau = args_info.tau_arg;
  if (args_info.sigma_given) sigma = args_info.sigma_arg;
  if (args_info.precision_given) precision = args_info.precision_arg;
  if (args_info.step_given) initial_step_size = args_info.step_arg;
  if (args_info.maxN_given) max_iteration = args_info.maxN_arg;
  if (args_info.method_given) method_id = args_info.method_arg;
  if (args_info.tolerance_given) tolerance = args_info.tolerance_arg;
  if (args_info.outfile_given) strcpy(outfile, args_info.outfile_arg);
  
  /* Generic RNAfold options */
  
  if (args_info.temp_given)        temperature = args_info.temp_arg;
  if (args_info.constraint_given)  fold_constrained=1;
  if (args_info.noTetra_given)     tetra_loop=0;
  if (args_info.dangles_given)     dangles = args_info.dangles_arg;
  if (args_info.noLP_given)        noLonelyPairs = 1;
  if (args_info.noGU_given)        noGU = 1;
  if (args_info.noClosingGU_given) no_closingGU = 1;
  if (args_info.noconv_given)      noconv = 1;
  if (args_info.energyModel_given) energy_set = args_info.energyModel_arg;
  if (args_info.paramFile_given)   ParamFile = strdup(args_info.paramFile_arg);
  if (args_info.nsp_given)         ns_bases = strdup(args_info.nsp_arg);
  if (args_info.pfScale_given)     sfact = args_info.pfScale_arg;
  if (args_info.noPS_given)        noPS=1;

  /*
  if(args_info.MEA_given){
    pf = doMEA = 1;
    if(args_info.MEA_arg != -1)
      MEAgamma = args_info.MEA_arg;
   }
  */

  RNAfold_cmdline_parser_free (&args_info);

  if (ParamFile != NULL) {
    read_parameter_file(ParamFile);
  }

  if (ns_bases != NULL) {
    nonstandards = space(33);
    c=ns_bases;
    i=sym=0;
    if (*c=='-') {
      sym=1; c++;
    }
    while (*c!='\0') {
      if (*c!=',') {
        nonstandards[i++]=*c++;
        nonstandards[i++]=*c;
        if ((sym)&&(*c!=*(c-1))) {
          nonstandards[i++]=*c;
          nonstandards[i++]=*(c-1);
        }
      }
      c++;
    }
  }

  fname[0] = '\0';
  while((input_type = get_input_line(&input_string, 0)) & VRNA_INPUT_FASTA_HEADER){
    (void) sscanf(input_string, "%42s", fname);
    //printf("name\t%s\n", input_string);
    free(input_string);
  }

  length = (int)    strlen(input_string);
  string = strdup(input_string);
  free(input_string);
  structure = (char *) space((unsigned) length+1);

  /* For testing purpose pass dot bracket structure of reference structure via -C */
  if (fold_constrained) {
    input_type = get_input_line(&input_string, VRNA_INPUT_NOSKIP_COMMENTS);
    if(input_type & VRNA_INPUT_QUIT){ exit(1);}
    else if((input_type & VRNA_INPUT_MISC) && (strlen(input_string) > 0)){
      cstruc = strdup(input_string);
      free(input_string);
    }
    else warn_user("-C was given but reference structure is missing");
  }
  
  if(noconv) {
    str_RNA2RNA(string);
  } else {
    str_DNA2RNA(string);
  }

  /*** Start of RNApbfold specific code ***/

  /* Allocating space */
  
  epsilon =     (double *) space(sizeof(double)*(length+1));
  prev_epsilon = (double *) space(sizeof(double)*(length+1));
  gradient =    (double *) space(sizeof(double)*(length+1));
  gradient_numeric =    (double *) space(sizeof(double)*(length+1));
  gradient_numeric_gsl =    (double *) space(sizeof(double)*(length+1));
  prev_gradient = (double *) space(sizeof(double)*(length+1));
  
  q_unpaired = (double *) space(sizeof(double)*(length+1));
  p_unpaired_cond = (double **)space(sizeof(double *)*(length+1));
  p_unpaired_cond_sampled = (double **)space(sizeof(double *)*(length+1));
  p_pp =  (double **)space(sizeof(double *)*(length+1));
  p_unpaired =  (double *) space(sizeof(double)*(length+1));
  
  for (i=0; i <= length; i++){
    epsilon[i] = gradient[i] = q_unpaired[i] = 0.0;
    p_unpaired_cond[i] = (double *) space(sizeof(double)*(length+1));
    p_unpaired_cond_sampled[i] = (double *) space(sizeof(double)*(length+1));
    p_pp[i] = (double *) space(sizeof(double)*(length+1));
    for (j=0; j <= length; j++){
      p_pp[i][j]=p_unpaired_cond[i][j] = 0.0;
      p_unpaired_cond_sampled[i][j] = 0.0;
    }
  }
  
  
  /*** Get constraints from reference structure or from file constraints.dat ***/

  if (fold_constrained){
    for (i=0; i<length; i++){
      if (cstruc[i] == '(' || cstruc[i] == ')'){
        q_unpaired[i+1] = 0.0;
      } else {
        q_unpaired[i+1] = 1.0;
      }
    }
  } else {

    filehandle = fopen ("constraints.dat","r");

    if (filehandle == NULL){
      nrerror("No constraints given as dot bracket or in constraints.dat");
    }
    
    i=1;
    while (1) {
      double t;
      line = get_line(filehandle);
      if (line == NULL) break;
      if (i>length) nrerror("Too many values in constraints.dat");
      if (sscanf(line, "%lf", &q_unpaired[i]) !=1){
        nrerror("Error while reading constraints.dat");
      }
      i++;
    }

    if (i-1 != length){
      nrerror("Too few values in constraints.dat");
    }
  }

  if (outfile[0] !='\0'){
    statsfile = fopen (outfile,"w");
  } else {
    statsfile = fopen ("stats.dat","w");
  }

  setvbuf(statsfile, NULL, _IONBF, 0);

  fprintf(statsfile, "Iteration\tDiscrepancy\tNorm\tdfCount\tMEA\n");

  if (statsfile == NULL){
    nrerror("Could not open stats.dat for writing.");
  }

  fprintf(stderr, "tau^2 = %.4f; sigma^2 = %.4f; precision = %.4f; tolerance = %.4f; step-size: %.4f\n\n", 
          tau, sigma, precision, tolerance, initial_step_size);

  st_back=1;

  dangles=0;
  
  min_en = fold(string, structure);
    
  (void) fflush(stdout);

  if (length>2000) free_arrays(); 

  pf_struc = (char *) space((unsigned) length+1);


  if (dangles==1) {
    dangles=2;   /* recompute with dangles as in pf_fold() */
    min_en = energy_of_struct(string, structure);
    dangles=1;
  }

  kT = (temperature+273.15)*1.98717/1000.; /* in Kcal */
  pf_scale = exp(-(sfact*min_en)/kT/length);



  /*
  for (i=1; i <= length; i++){
    if (i%2==0){
      epsilon[i] = +0.1*i;
    } else {
      epsilon[i] = -0.1*i;
    }
  }
  */
  
  /* Set up minimizer */

  minimizer_x = gsl_vector_alloc (length+1);
  minimizer_g = gsl_vector_alloc (length+1);

  for (i=0; i <= length; i++){
    epsilon[i] = 0.0;
    gsl_vector_set (minimizer_g, i, 0.0);
    gsl_vector_set (minimizer_x, i, epsilon[i]);
  }

  minimizer_pars.length=length;
  minimizer_pars.seq = string;
  minimizer_pars.tau=tau;
  minimizer_pars.sigma=sigma;
  minimizer_pars.kT=kT;
  
  minimizer_func.n = length+1;
  minimizer_func.f = calculate_f;
  minimizer_func.df = numerically ? calculate_df_numerically: calculate_df;
  minimizer_func.fdf = calculate_fdf;
  minimizer_func.params = &minimizer_pars;

 
  test_folding(string, length);
  //test_stochastic_backtracking(string, length);
  //test_gradient(minimizer_func, minimizer_pars);
  //test_gradient_sampling(minimizer_func, minimizer_pars);

  exit(1);

  count_df_evaluations=0;

  /* GSL minimization */

  if (method_id >=2){
    char name[100];
    // Available algorithms 
    //  2  gsl_multimin_fdfminimizer_conjugate_fr
    //  3  gsl_multimin_fdfminimizer_conjugate_pr
    //  4  gsl_multimin_fdfminimizer_vector_bfgs
    //  5  gsl_multimin_fdfminimizer_vector_bfgs2
    //  6  gsl_multimin_fdfminimizer_steepest_descent
    
    //   http://www.gnu.org/software/gsl/manual/html_node/Multimin-Algorithms-with-Derivatives.html

    switch (method_id){
    case 2: 
      minimizer = gsl_multimin_fdfminimizer_alloc (gsl_multimin_fdfminimizer_conjugate_fr, length+1); 
      strcpy(name, "Fletcher-Reeves conjugate gradient");
      break;
    case 3: 
      minimizer = gsl_multimin_fdfminimizer_alloc (gsl_multimin_fdfminimizer_conjugate_pr, length+1); 
      strcpy(name, "Polak-Ribiere conjugate gradient");
      break;
    case 4: 
      minimizer = gsl_multimin_fdfminimizer_alloc ( gsl_multimin_fdfminimizer_vector_bfgs, length+1); 
      strcpy(name, "Broyden-Fletcher-Goldfarb-Shanno");
      break;
    case 5: 
      minimizer = gsl_multimin_fdfminimizer_alloc ( gsl_multimin_fdfminimizer_vector_bfgs2, length+1); 
      strcpy(name, "Broyden-Fletcher-Goldfarb-Shanno (improved version)");
      break;
    case 6: 
      minimizer = gsl_multimin_fdfminimizer_alloc (gsl_multimin_fdfminimizer_steepest_descent, length+1); 
      strcpy(name, "Gradient descent (GSL implmementation)");
      break;
    }

    fprintf(stderr, "Starting minimization via GSL implementation of %s...\n\n", name);
    
    // The last two parmeters are step size and tolerance (with
    // different meaning for different algorithms 

    gsl_multimin_fdfminimizer_set (minimizer, &minimizer_func, minimizer_x, initial_step_size, tolerance);
    
    iteration = 1;
    prev_D = -1.0;

    do {

     
      fprintf (stderr, "\nITERATION %i:\n", iteration);
      
      D = minimizer->f;
      norm = gsl_blas_dnrm2(minimizer->gradient);
      
      fprintf(stderr, "DISCREPANCY: %.4f\n", D);
      fprintf(stderr, "NORM OF GRADIENT: %.2f\n", norm);
      
      if (prev_D > -1.0) {
        fprintf(stderr, "IMPROVEMENT: %.4f%%\n", (1-(D/prev_D))*100);
      }
      
      fprintf(statsfile, "%i\t%.4f\t%.4f\t%i\t", iteration, D, norm, count_df_evaluations);
      //print_mea_string(statsfile, string, length);
      fprintf(statsfile, "\n");

      prev_D = D;

      if (!noPS) print_dotplot(string, cstruc, length, iteration, minimizer->f);

      status = gsl_multimin_fdfminimizer_iterate (minimizer);

      if (status) {
        fprintf(stderr, "An unexpected error has occured in the iteration (status:%i)\n", status);
      }
    
      status = gsl_multimin_test_gradient (minimizer->gradient, 1e-3);
      if (status == GSL_SUCCESS) printf ("Minimum found stopping.\n");
      
      iteration++;
     
    } while (status == GSL_CONTINUE && iteration < max_iteration);

    gsl_multimin_fdfminimizer_free (minimizer);
    gsl_vector_free (minimizer_x);

    /* Custom implementation of steepest descent */
  } else {

    if (method_id == 0){
      fprintf(stderr, "Starting custom implemented steepest descent search...\n\n");
    } else {
      fprintf(stderr, "Starting custom implemented steepest descent search with Barzilai Borwein step size...\n\n");
    }

    iteration = 0;
    D = 0.0;
    prev_D = -1.0;

    while (iteration++ < max_iteration){
    
      fprintf(stderr, "\nITERATION %i\n", iteration);
    
      for (i=1; i <= length; i++){
        gsl_vector_set (minimizer_x, i, epsilon[i]);
      }
    
      D = calculate_f(minimizer_x, (void*)&minimizer_pars);

      if (prev_D > -1.0) {
        fprintf(stderr, "IMPROVEMENT: %.4f%%\n", (1-(D/prev_D))*100);
        if ((prev_D-D)/(prev_D+D) < precision){
          //break;
        }
      }

      prev_D = D;

      if (!noPS) print_dotplot(string, cstruc, length, iteration, D);

      norm = calculate_norm(gradient,length);
    
      fprintf(stderr, "DISCREPANCY %.4f\n", D);
      fprintf(stderr, "NORM %.4f\n", norm);

      if (norm<precision && iteration>1) break;
      //break;

      fprintf(statsfile, "%i\t%.4f\t%.4f\t%i\t", iteration, D, norm, count_df_evaluations);
      //print_mea_string(statsfile, string, length);
      fprintf(statsfile, "\n");

      if (numerically){
        calculate_df_numerically(minimizer_x, (void*)&minimizer_pars, minimizer_g);
      } else {
        calculate_df(minimizer_x, (void*)&minimizer_pars, minimizer_g);
      }

      for (i=1; i <= length; i++){
        gradient[i] = gsl_vector_get (minimizer_g, i);
      }
   
      // Do line search

      fprintf(stderr, "\nLine search:\n");

      // After the first iteration, use Barzilai-Borwain (1988) step size (currently turned off)
      if (iteration>1 && method_id==2){
      
        double denominator=0.0;
        double numerator=0.0; 
      
        for (i=1; i <= length; i++){
          numerator += (epsilon[i]-prev_epsilon[i]) * (gradient[i]-prev_gradient[i]);
          denominator+=(gradient[i]-prev_gradient[i]) * (gradient[i]-prev_gradient[i]);
        }
        
        step_size = numerator / denominator;
    
        norm =1.0;
      } else {
        // Use step sized given by the user (normalize it first)
        step_size = initial_step_size / calculate_norm(gradient, length);
      }
      
      for (i=1; i <= length; i++){
        prev_epsilon[i] = epsilon[i];
        prev_gradient[i] = gradient[i];
      }

      do {
      
        for (mu=1; mu <= length; mu++){
          epsilon[mu] = prev_epsilon[mu] - step_size * gradient[mu];
        }

        for (i=1; i <= length; i++){
          gsl_vector_set (minimizer_x, i, epsilon[i]);
        }
      
        DD = calculate_f(minimizer_x, (void*)&minimizer_pars);

        if (step_size > 0.0001){
          fprintf(stderr, "Old D: %.4f; New D: %.4f; Step size: %.4f\n", D, DD, step_size);
        } else {
          fprintf(stderr, "Old D: %.4f; New D: %.4f; Step size: %.4e\n", D, DD, step_size);
        }

        step_size /= 2;
      } while (step_size > 1e-12 && DD > D);
      
      if (DD > D){
        fprintf(stderr, "Line search did not improve D in iteration %i. Stop.\n", iteration);
        break;
      }
      
      fprintf(stderr, "\n");
    }
  }

  free(pf_struc);
  if (cstruc!=NULL) free(cstruc);
  (void) fflush(stdout);
  free(string);
  free(structure);
  
  return 0;
}

PRIVATE struct plist *b2plist(const char *struc) {
  /* convert bracket string to plist */
  short *pt;
  struct plist *pl;
  int i,k=0;
  pt = make_pair_table(struc);
  pl = (struct plist *)space(strlen(struc)/2*sizeof(struct plist));
  for (i=1; i<strlen(struc); i++) {
    if (pt[i]>i) {
      pl[k].i = i;
      pl[k].j = pt[i];
      pl[k++].p = 0.95*0.95;
    }
  }
  free(pt);
  pl[k].i=0;
  pl[k].j=0;
  pl[k++].p=0.;
  return pl;
}


PRIVATE struct plist *make_plist(int length, double pmin) {
  /* convert matrix of pair probs to plist */
  struct plist *pl;
  int i,j,k=0,maxl;
  maxl = 2*length;
  pl = (struct plist *)space(maxl*sizeof(struct plist));
  k=0;
  for (i=1; i<length; i++)
    for (j=i+1; j<=length; j++) {
      if (pr[iindx[i]-j]<pmin) continue;
      if (k>=maxl-1) {
        maxl *= 2;
        pl = (struct plist *)xrealloc(pl,maxl*sizeof(struct plist));
      }
      pl[k].i = i;
      pl[k].j = j;
      pl[k++].p = pr[iindx[i]-j];
    }
  pl[k].i=0;
  pl[k].j=0;
  pl[k++].p=0.;
  return pl;
}


/* Objective function */
double calculate_f(const gsl_vector *v, void *params){

  double D;
  int i,j,length;
  minimizer_pars_struct *pars = (minimizer_pars_struct *)params;

  //fprintf(stderr, "=> Evaluating objective Function...\n");
  
  length = pars->length;
  
  for (i=0; i <= length; i++){
    epsilon[i] = gsl_vector_get(v, i);
    p_unpaired[i] = 0.0;
    for (j=0; j <= length; j++){
      p_pp[i][j] = 0.0;
    }
  }
  
  init_pf_fold(length);
  pf_fold_pb(pars->seq, NULL);

  for (i = 1; i < length; i++){
    for (j = i+1; j<= length; j++) {
      p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
    }
  }

  get_pair_prob_vector(p_pp, p_unpaired, length, 1); 

  free_pf_arrays();

  D = 0.0;

  for (i = 1; i <= length; i++){
    D += 1 / pars->tau * epsilon[i] * epsilon[i];
    D += 1 / pars->sigma *
      ( p_unpaired[i] - q_unpaired[i] ) *
      ( p_unpaired[i] - q_unpaired[i] );
  }

  return D;
}

     
/* Calculate the gradient analytically */
void calculate_df (const gsl_vector *v, void *params, gsl_vector *df){

  double D, sum;
  int ii,jj,i,j,mu, length, N;
  minimizer_pars_struct *pars = (minimizer_pars_struct *)params;
  char *constraints;

  int* unpaired_count;
  int** unpaired_count_cond;

  length = pars->length;

  count_df_evaluations++;

  fprintf(stderr, "=> Evaluating gradient (analytical)...\n");

  constraints = (char *) space((unsigned) length+1);
  for (i=0; i <= length; i++){
    epsilon[i] = gsl_vector_get(v, i);
    p_unpaired[i] = 0.0;
    for (j=0; j <= length; j++){
      p_pp[i][j] = p_pp[j][i] = 0.0;
      p_unpaired_cond[i][j] = 0.0;
      p_unpaired_cond_sampled[i][j] = 0.0;
    }
  }

  init_pf_fold(length);
  pf_fold_pb(pars->seq, NULL);

  for (i=1; i<length; i++){
    for (j=i+1; j<=length; j++) {
      p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
    }
  }
  get_pair_prob_vector(p_pp, p_unpaired, length, 1); 

  free_pf_arrays();
  

  if (!sample_conditionals){
    // Calculate conditional probabilities
    
    fold_constrained=1;

    for (ii = 1; ii <= length; ii++){

      // Set constraints strings like 
      //   x............
      //   .x...........
      //   ..x..........

      memset(constraints,'.',length);
      constraints[ii-1]='x';
    
      fprintf(stderr, ".");
      
      init_pf_fold(length);
      pf_fold_pb(pars->seq, constraints);
    
      for (i=1; i<length; i++){
        for (j=i+1; j<=length; j++) {
          p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
        }
      }
      get_pair_prob_vector(p_pp, p_unpaired_cond[ii], length, 1); 
      free_pf_arrays();
    }

    fold_constrained = 0;

    // Sample gradient with stochastic backtracking
  } else {

    unpaired_count = (int *) space(sizeof(int)*(length+1));
    unpaired_count_cond = (int **)space(sizeof(int *)*(length+1));

    for (i=0; i <= length; i++){
      unpaired_count[i] = 0;
      unpaired_count_cond[i] = (int *) space(sizeof(int)*(length+1));
      for (j=0; j <= length; j++){
        unpaired_count_cond[i][j] = 0;
      }
    }

    fold_constrained = 0;

    init_pf_fold(length); 

    pf_fold_pb(pars->seq, NULL);
  
    N=10000;
  
    for (i=1; i<=N; i++) {
      char *s;
      s = pbacktrack_pb(pars->seq);

      for (ii = 1; ii <= length; ii++){
        if (s[ii-1]=='.'){
          unpaired_count[ii]++;
          for (jj = 1; jj <= length; jj++){
            if (s[jj-1]=='.'){
              unpaired_count_cond[ii][jj]++;
            }
          }
        }
      }
      free(s);
    }
  
    for (i = 1; i <= length; i++){
      for (ii = 1; ii <= length; ii++){
        if (unpaired_count_cond[i][ii] > 0){
          p_unpaired_cond_sampled[i][ii] = (double)unpaired_count_cond[i][ii]/(double)unpaired_count[i];
          p_unpaired_cond[i][ii] = (double)unpaired_count_cond[i][ii]/(double)unpaired_count[i];
        } else {
          p_unpaired_cond_sampled[i][ii]= 0.0;
          p_unpaired_cond[i][ii]= 0.0;
        }
        //fprintf(stderr, "%i\t%i\t%i\t%i\t%.5f\t%.5f\n", i, ii, unpaired_count[i], unpaired_count_cond[i][ii], p_unpaired_cond[i][ii], p_unpaired_cond_sampled[i][ii]);
      }
    }
    //fprintf(stderr, "\n");
  }

  // Calculate gradient
  for (mu=1; mu <= length; mu++){
    sum = 0.0;
    for (i=1; i <= length; i++){
      sum += p_unpaired[i] *
        ( p_unpaired[i] - q_unpaired[i] ) *
        ( p_unpaired[mu] - p_unpaired_cond[i][mu] );
    }
    gsl_vector_set(df, mu, (2 * epsilon[mu] /pars->tau ) + (2 / pars->sigma /  pars->kT * sum));
  }
}

/* Calculate the gradient numerically */
void calculate_df_numerically (const gsl_vector *v, void *params, gsl_vector *df){
  
  double D;
  int ii,i,length;
  minimizer_pars_struct *pars = (minimizer_pars_struct *)params;
  double d;
  gsl_vector *minimizer_xx;
  double D_plus, D_minus;
  double curr_epsilon;

  d = numeric_d;
  
  count_df_evaluations++;

  fprintf(stderr, "=> Evaluating gradient (numerically)...");
  length = pars->length;

  minimizer_xx = gsl_vector_alloc (length+1);
  
  for (i=1; i <= length; i++){
    
    curr_epsilon = gsl_vector_get(v,i);

    for (ii=1; ii <= length; ii++){
      gsl_vector_set(minimizer_xx, ii, gsl_vector_get(v, ii));
    }

    gsl_vector_set(minimizer_xx,i, curr_epsilon+d);

    fprintf(stderr, ".");
    D_plus = calculate_f(minimizer_xx, params);
       
    gsl_vector_set(minimizer_xx,i, curr_epsilon-d);

    fprintf(stderr, ".");
    D_minus = calculate_f(minimizer_xx, params);
    gsl_vector_set(df, i, (D_plus-D_minus)/(2*d));
    
  }

  gsl_vector_free(minimizer_xx);

}



/* Compute both f and df together. */
void calculate_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df){
  *f = calculate_f(x, params);

  if (numerically) {
    calculate_df_numerically(x, params, df);
  } else {
    calculate_df(x, params, df);
  }
}

/* Gets vector of being paired (type=0) or unpaired (type=1) from
   base-pair probability matrix 'matrix' and stores it in vector
   'vector'; length is the length of the sequence */

PRIVATE void get_pair_prob_vector(double** matrix, double* vector, int length, int type) { 

  int i, j;

  for (i=1; i<=length; i++){
    vector[i] = 0.0;
    for (j=1; j<=length; j++){
      vector[i] += matrix[i][j];
    }
    if (type == 1) {
      vector[i] = 1 - vector[i];
    }
  }
}


/* Calculate euclidian norm of vector */
double calculate_norm(double* vector, int length){

  int i; 
  double norm = 0.0;
  
  for (i=1; i <= length; i++){
    norm += vector[i]*vector[i];
  }

  return(sqrt(norm));

}

/* Print dotplot for current iteration */
void print_dotplot(char* seq, char* struc, int length, int iteration, double D){

  plist *pl1,*pl2;
  char fname[100];
  char title[100];

  init_pf_fold(length);
  pf_fold_pb(seq, NULL);

  sprintf(fname,"iteration%i.ps", iteration);
  pl1 = make_plist(length, 1e-5);
  pl2 = b2plist(struc);
  sprintf(title,"Iteration %i, D = %.4f", iteration, D);
  (void) PS_dot_plot_list_epsilon(seq, fname, pl2, pl1, epsilon, title);
  free_arrays(); 
  
}


/* Calculate MEA structures for current iteration */

char* print_mea_string(FILE* statsfile, char* seq, int length){

  double MEAgamma;
  plist *pl;
  char* ss;
  double mea, mea_en;
  char* output;
  ss = (char *) space((unsigned) length+1);
  ss = (char *) space((unsigned) length+1);
  memset(ss,'.',length);
  
  for (MEAgamma=1e-5; MEAgamma<1e+6; MEAgamma*=10 ){
    pl = make_plist(length, 1e-4/(1+MEAgamma));
    mea = MEA(pl, ss, MEAgamma);
    mea_en = energy_of_struct(seq, ss);
    fprintf(statsfile,"%s,%.2e;", ss, MEAgamma);
    free(pl);
  }

  return NULL;

}


/* Testing (conditional) folding; use with eval.pl */
void test_folding(char* seq, int length){

  int i,j;
  char* constraints;

  fold_constrained = 1;
  constraints = (char *) space((unsigned) length+1);

  memset(constraints,'.',length);
  constraints[0]='x';

  //constraints=NULL;

  init_pf_fold(length);

  pf_fold_pb(seq, constraints);
  
  for (i = 1; i < length; i++){
    for (j = i+1; j<= length; j++) {
      p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
      if (p_pp[i][j]>1e-50){
        printf("%i %i %.13e \n", i, j, p_pp[i][j]);
      }
    }
  }

  free_arrays(); 
}


/*** Testing stochastic backtracking ***/

void test_stochastic_backtracking(char* seq, int length){

  int i, j, N;
  
  init_pf_fold(length);

  pf_fold_pb(seq, NULL);
  
  for (i = 1; i < length; i++){
    for (j = i+1; j<= length; j++) {
      p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
    }
  }

  get_pair_prob_vector(p_pp, p_unpaired, length, 1);

  for (i=1; i <= length; i++){
    q_unpaired[i] = 0.0;
  }

  N=10000;
  
  for (i=1; i<=N; i++) {
    char *s;
    s = pbacktrack_pb(seq);
    for (j=1; j <= length; j++){
      if (s[j-1]=='.'){
        q_unpaired[j]+=1.0/N;
      }
    }
    free(s);
  }

  for (i=1; i <= length; i++){
    printf("%.4f\t%.4f\t%.4f\n",  p_unpaired[i], q_unpaired[i], q_unpaired[i]-p_unpaired[i]);
  }

  free_arrays(); 

}


/*********** Testing gradient ***************/

void test_gradient(gsl_multimin_function_fdf minimizer_func,  minimizer_pars_struct minimizer_pars){

  int i, j, length; 
  gsl_vector *minimizer_x;
  gsl_vector *minimizer_g;
  double* gradient_analytic;
  double* gradient_numeric;

  length = minimizer_pars.length;

  gradient_analytic = (double *) space(sizeof(double)*(length+1));
  gradient_numeric = (double *) space(sizeof(double)*(length+1));

  minimizer_x = gsl_vector_alloc (length+1);
  minimizer_g = gsl_vector_alloc (length+1);

  
  for (i=1; i <= length; i++){
    epsilon[i] = 0.0;
    gsl_vector_set (minimizer_g, i, 0.0);
    gsl_vector_set (minimizer_x, i, epsilon[i]);
  }
  
  calculate_df(minimizer_x, (void*)&minimizer_pars, minimizer_g);

  for (i=1; i <= length; i++){
    gradient_analytic[i]= gsl_vector_get (minimizer_g, i);
  }
  
  /* Numerical */

  numeric_d = 0.00001;

  for (i=0; i <= length; i++){
    epsilon[i] = 0.0;
    gsl_vector_set (minimizer_g, i, 0.0);
    gsl_vector_set (minimizer_x, i, epsilon[i]);
  }

  calculate_df_numerically(minimizer_x, (void*)&minimizer_pars, minimizer_g);
  
  for (i=1; i <= length; i++){
    gradient_numeric[i]= gsl_vector_get(minimizer_g, i);
    printf("%i\t%.4f\t%.4f\t%.6f\n", i, gradient_analytic[i], gradient_numeric[i], gradient_analytic[i]-gradient_numeric[i]);
  }

}

/*********** Testing analytic gradient vs numeric gradient ***************/

void test_gradient_sampling(gsl_multimin_function_fdf minimizer_func,  minimizer_pars_struct minimizer_pars){

  int i, j, length; 
  gsl_vector *minimizer_x;
  gsl_vector *minimizer_g;
  double* gradient_analytic;
  double* gradient_sampled;

  length = minimizer_pars.length;

  gradient_analytic = (double *) space(sizeof(double)*(length+1));
  gradient_sampled = (double *) space(sizeof(double)*(length+1));

  minimizer_x = gsl_vector_alloc (length+1);
  minimizer_g = gsl_vector_alloc (length+1);
 
  /* Analytical */

  sample_conditionals=0;

  for (i=1; i <= length; i++){
    epsilon[i] = 0.0;
    gsl_vector_set (minimizer_g, i, 0.0);
    gsl_vector_set (minimizer_x, i, epsilon[i]);
  }

  calculate_df(minimizer_x, (void*)&minimizer_pars, minimizer_g);

  for (i=1; i <= length; i++){
    gradient_analytic[i]= gsl_vector_get (minimizer_g, i);
  }

  /* Sampling */
  
  sample_conditionals=1;
   
  for (i=0; i <= length; i++){
    epsilon[i] = 0.0;
    gsl_vector_set (minimizer_g, i, 0.0);
    gsl_vector_set (minimizer_x, i, epsilon[i]);
  }

  calculate_df(minimizer_x, (void*)&minimizer_pars, minimizer_g);
  
  for (i=1; i <= length; i++){
    gradient_sampled[i]= gsl_vector_get(minimizer_g, i);
    printf("%i\t%.4f\t%.4f\t%.6f\n", i, gradient_analytic[i], gradient_sampled[i], gradient_analytic[i]-gradient_sampled[i]);
  }

}


