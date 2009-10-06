/* functions from part_func.c */
/* calculate partition function and base pair probabilities */
#ifndef __VIENNA_RNA_PACKAGE_PART_FUNC_CO_H__
#define __VIENNA_RNA_PACKAGE_PART_FUNC_CO_H__

extern int mirnatog; /*toggles no intrabp in 2nd mol*/

typedef struct cofoldF {
  /* free energies for: */
  double F0AB; /* null model without DuplexInit */
  double FAB;  /* all states with DuplexInit corretion */
  double FcAB; /* true hybrid states only */
  double FA;   /* monomer A */
  double FB;   /* monomer B */
} cofoldF;

typedef struct ConcEnt {
  double A0;    /*start concentration A*/
  double B0;    /*start concentration B*/
  double ABc;   /*End concentration AB*/
  double AAc;
  double BBc;
  double Ac;
  double Bc;
} ConcEnt;



typedef struct pairpro{
  struct plist *AB;
  struct plist *AA;
  struct plist *A;
  struct plist *B;
  struct plist *BB;
}pairpro;

cofoldF co_pf_fold(char *sequence, char *structure); /* calculate partition function and base pair probabilities */
void    init_co_pf_fold(int length);
void    free_co_pf_arrays(void);
void    update_co_pf_params(int length); /*recalculate energy parameters */
char    co_bppm_symbol(float *x);    /* string representation of structure */
void    compute_probabilities(double FAB, double FEA, double FEB,
                              struct plist  *prAB,
                              struct plist  *prA, struct plist  *prB,
                              int Alength);
struct  ConcEnt  *get_concentrations(double FEAB, double FEAA, double FEBB, double FEA, double FEB, double * startconc);
struct  plist *get_plist(struct plist *pl, int length, double cut_off);
int     make_probsum(int length, char *name);

#endif
