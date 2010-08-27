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


PRIVATE struct plist *b2plist(const char *struc);
PRIVATE struct plist *make_plist(int length, double pmin);

PRIVATE void get_pair_prob_vector(double** matrix, double* vector, int length, int type); 
PRIVATE double discrepancy(double *q_unpaired, double *p_unpaired, double sigma, double tau, int length);


PUBLIC double *epsilon; /* Perturbation vector in cal/Mol */

/*--------------------------------------------------------------------------*/

int main(int argc, char *argv[]){

  struct        RNAfold_args_info args_info;
  char          *string, *input_string, *structure=NULL, *cstruc=NULL;
  char          fname[80], ffname[80], gfname[80], *ParamFile=NULL;
  char          *ns_bases=NULL, *c;
  int           i, j, ii, jj, mu, length, l, sym, r, pf=0, noPS=0, noconv=0;
  unsigned int  input_type;
  double        energy, min_en, kT, sfact=1.07;
  int           doMEA=0, circular = 0;
  double        MEAgamma = 1.;
  char *pf_struc;
  plist *pl1,*pl2;
  double dist;
  float mea, mea_en;
  plist *pl;

  FILE * filehandle;
  char* line;

  double tau   = 1.0; /* Variance of energy parameters */
  double sigma = 1.0; /* Variance of experimental constraints */

  double **p_pp;              /* Base pair probability matrix of predicted structure              */
  double *q_unpaired;         /* Vector of probs. of being unpaired in the experimental structure */
  double *p_unpaired;         /* Vector of probs. of being unpaired in the predicted structure    */
  double **p_unpaired_cond;   /* List of vectors p_unpaired with the condition that i is unpaired */
  double *gradient;           /* Gradient for steepest descent search
                                 epsilon[i+1]= epsilon[i] - gradient * step_size */

  double initial_step_size = 0.01;  /* Step size for steepest descent search */
  double step_size;                 /* Step size for steepest descent search */

  double D;                  /* Discrepancy (i.e. value of objective function) for the current prediction */

  int iteration, max_iteration = 100; /* Current and maximum number of iterations after which algorithm stops */

  double precision = 0.001;
  double norm;

  double *prev_epsilon;
  double *prev_gradient;

  double DD, prev_D, sum; 
   
  do_backtrack  = 1;
  string        = NULL;

  if(RNAfold_cmdline_parser (argc, argv, &args_info) != 0) exit(1);

  /* RNAbpfold specific options */

  if (args_info.tau_given) tau = args_info.tau_arg;
  if (args_info.sigma_given) sigma = args_info.sigma_arg;
  if (args_info.precision_given) precision = args_info.precision_arg;
  if (args_info.step_given) initial_step_size = args_info.step_arg;
  if (args_info.maxN_given) max_iteration = args_info.maxN_arg;
  
  /* Generic RNAfold options */

  /* temperature */
  if(args_info.temp_given)        temperature = args_info.temp_arg;
  /* structure constraint */
  if(args_info.constraint_given)  fold_constrained=1;
  /* do not take special tetra loop energies into account */
  if(args_info.noTetra_given)     tetra_loop=0;
  /* set dangle model */
  if(args_info.dangles_given)     dangles = args_info.dangles_arg;
  /* do not allow weak pairs */
  if(args_info.noLP_given)        noLonelyPairs = 1;
  /* do not allow wobble pairs (GU) */
  if(args_info.noGU_given)        noGU = 1;
  /* do not allow weak closing pairs (AU,GU) */
  if(args_info.noClosingGU_given) no_closingGU = 1;
  /* do not convert DNA nucleotide "T" to appropriate RNA "U" */
  if(args_info.noconv_given)      noconv = 1;
  /* set energy model */
  if(args_info.energyModel_given) energy_set = args_info.energyModel_arg;
  /* take another energy parameter set */
  if(args_info.paramFile_given)   ParamFile = strdup(args_info.paramFile_arg);
  /* Allow other pairs in addition to the usual AU,GC,and GU pairs */
  if(args_info.nsp_given)         ns_bases = strdup(args_info.nsp_arg);
  /* set pf scaling factor */
  if(args_info.pfScale_given)     sfact = args_info.pfScale_arg;
  /* do not produce postscript output */
  if(args_info.noPS_given)        noPS=1;
  /* MEA (maximum expected accuracy) settings */
  if(args_info.MEA_given){
    pf = doMEA = 1;
    if(args_info.MEA_arg != -1)
      MEAgamma = args_info.MEA_arg;
  }

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
    printf("name\t%s\n", input_string);
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
  prev_gradient = (double *) space(sizeof(double)*(length+1));
  
  q_unpaired = (double *) space(sizeof(double)*(length+1));
  p_unpaired = (double *) space(sizeof(double)*(length+1));

  p_pp =            (double **)space(sizeof(double *)*(length+1));
  p_unpaired_cond = (double **)space(sizeof(double *)*(length+1));

  for (i=0; i <= length; i++){

    epsilon[i] = gradient[i] = q_unpaired[i] = p_unpaired[i] = 0.0;
    
    p_pp[i] = (double *) space(sizeof(double)*(length+1));
    p_unpaired_cond[i] = (double *) space(sizeof(double)*(length+1));
    
    for (j=0; j <= length; j++){
      p_pp[i][j] = p_unpaired_cond[i][j] = 0.0;
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

  fold_constrained = 0;

  for (i=1; i <= length; i++){
    //epsilon[i] = i/10.0; // Perturbation vector in kcal/Mol
    epsilon[i] = 0.0;
  }

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

  /*** Main iteration ***/

  iteration = 0;
  D = 0.0;
  prev_D = -1.0;

  fprintf(stderr, "tau^2 = %.4f; sigma^2 = %.4f; precision = %.4f; step-size: %.4f\n\n", tau, sigma, precision, initial_step_size);

  while (iteration++ < max_iteration){

    fprintf(stderr, "ITERATION %i\n", iteration);
    
    init_pf_fold(length);
    
    energy = pf_fold_pb(string, NULL);
    
    printf("%i\tlnZ\t%6.6f\n", iteration,energy);

    printf("%i pairProbs\t", iteration);
  
    for (i=1; i<length; i++){
      for (j=i+1; j<=length; j++) {
        p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
        if (pr[iindx[i]-j] > 1e-5){
          printf("%i,%i:%.6f;", i, j, pr[iindx[i]-j]);
        }
      }
    }

    printf("\n");

    printf("%i\tMEA\t", iteration);

    for (MEAgamma=1.0; MEAgamma<4.0; MEAgamma+=1.0 ){
      pl = make_plist(length, 1e-4/(1+MEAgamma));
      mea = MEA(pl, structure, MEAgamma);
      mea_en = energy_of_struct(string, structure);
      printf("%s,%6.2f,%.2f;", structure, mea_en, mea);
      free(pl);
    }
  
    printf("\n");

    get_pair_prob_vector(p_pp, p_unpaired, length, 1); 

    D = discrepancy(q_unpaired, p_unpaired, sigma, tau, length);

    if (prev_D > -1.0) {
      fprintf(stderr, "Precision: %.4f\n", (prev_D-D)/(prev_D+D));
      if ((prev_D-D)/(prev_D+D) < precision){
        break;
      }
    }

    prev_D = D;

    printf("discrepancy\t%.4f\n", D);

    if (!noPS){
      sprintf(fname,"iteration%i.ps", iteration);
      pl1 = make_plist(length, 1e-5);
      pl2 = b2plist(cstruc);
      sprintf(ffname,"Iteration %i, D = %.4f", iteration, D);
      (void) PS_dot_plot_list_epsilon(string, fname, pl2, pl1, epsilon, ffname);
    }

    fprintf(stderr, "Discrepancy %.4f\n", D);

    /*** Calculate conditional probabilities ***/
    
    fprintf(stderr, "Calculating conditional probabilities\n");

    fold_constrained=1;
    
    for (ii = 1; ii <= length; ii++){

      /* Set constraints strings like 
         x............
         .x...........
         ..x..........
      */
      memset(pf_struc,'.',length);
      pf_struc[ii-1]='x';
      
      fprintf(stderr, ".");
      
      init_pf_fold(length);
    
      pf_fold_pb(string, pf_struc);

      for (i=1; i<length; i++){
        for (j=i+1; j<=length; j++) {
          p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
        }
      }
      get_pair_prob_vector(p_pp, p_unpaired_cond[ii], length, 1); 
    }

    fprintf(stderr, "\n");
    
    fold_constrained = 0;

    /*** Calculate gradient ***/

    for (mu=1; mu <= length; mu++){
      sum = 0.0;
      for (i=1; i <= length; i++){
        sum += p_unpaired[i] *
          ( p_unpaired[i] - q_unpaired[i] ) *
          ( p_unpaired[mu] - p_unpaired_cond[i][mu] );
      }

      gradient[mu] = (2 * epsilon[mu] /tau ) + (2 / sigma /  kT * sum);
    }

    

    norm = 0.0;

    for (mu=1; mu <= length; mu++){
      norm+=gradient[mu]*gradient[mu];
    }

    norm = sqrt(norm);

    fprintf(stderr,"Norm: %.4f\n ", norm);

    /*** Do line search ***/

    fprintf(stderr, "Line search:\n");

    //fprintf(stderr, "Current epsilon:\n");

    /* After the first iteration, use Barzilai-Borwain (1988) step size */
    //if (iteration>1){
    if (0){
      
      double denominator=0.0;
      double numerator=0.0; 
      
      for (i=1; i <= length; i++){
        numerator += (epsilon[i]-prev_epsilon[i]) * (gradient[i]-prev_gradient[i]);
        denominator+=(gradient[i]-prev_gradient[i]) * (gradient[i]-prev_gradient[i]);
      }

      step_size = numerator / denominator;
    
    } else {
      step_size = initial_step_size;
    }
    
    for (i=1; i <= length; i++){
      prev_epsilon[i] = epsilon[i];
      prev_gradient[i] = gradient[i];
    }

    do {
      
      for (mu=1; mu <= length; mu++){
        epsilon[mu] = prev_epsilon[mu] - step_size * gradient[mu];
        //fprintf(stderr, "%.8f,", epsilon[mu]);
      }

      fprintf(stderr, "\n");
        
      init_pf_fold(length);

      pf_fold_pb(string, NULL);
      
      for (i=1; i<length; i++){
        for (j=i+1; j<=length; j++) {
          p_pp[i][j]=p_pp[j][i]=pr[iindx[i]-j];
        }
      }
      
      get_pair_prob_vector(p_pp, p_unpaired, length, 1);
      
      DD = discrepancy(q_unpaired, p_unpaired, sigma, tau, length);

      fprintf(stderr, "Old D: %.4f; New D: %.4f; Step size: %.4f\n", D, DD, step_size);
     
      step_size /= 2;
    } while (step_size > 0.00001 && DD > D);

    if (DD > D){
      fprintf(stderr, "Line search did not improve D in iteration %i. Stop.\n", iteration);
      break;
    }
    
    fprintf(stderr, "\n");

  }

  free(pf_struc);
  free_pf_arrays();
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


/* Gets vector of being paired (type=0) or unpaired (type=1) from
   base-pair probability matrix 'matrix' and stores it in vector
   'vector'; length is the length of the sequence*/

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



double discrepancy(double *q_unpaired, double *p_unpaired, double sigma, double tau, int length){

  int i;
  double D = 0.0;

  //fprintf(stderr, "TAu: %.4f\n", tau);

  for (i=1; i<=length; i++){

    D += 1 / tau * epsilon[i] * epsilon[i];
    D += 1 / sigma *
      ( p_unpaired[i] - q_unpaired[i] ) *
      ( p_unpaired[i] - q_unpaired[i] );
  }
  return D;
}

