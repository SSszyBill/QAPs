/*****************************************************************
 Implementation of the robust taboo search of: E. Taillard
 Modified for:
 1. Double/Float matrix support
 2. CLI arguments for filename
 3. Automatic Timing
 4. Result Logging (Instance Name, Time, Obj)
 5. Log each best solution with timestamp
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>   /* For timing */
#include <string.h> /* For filename manipulation */

const double infinite = 1.0e30; 
const int FALSE = 0;
const int TRUE = 1;

typedef int* type_vector;
typedef double** type_matrix; 

double opt;
double somme_sol = 0.0;

/*************** L'Ecuyer random number generator ***************/
static int x10 = 12345, x11 = 67890, x12 = 13579, 
           x20 = 24680, x21 = 98765, x22 = 43210;

/* Function to set random seed */
void set_random_seed(int seed1, int seed2, int seed3, 
                     int seed4, int seed5, int seed6)
{
  x10 = seed1; x11 = seed2; x12 = seed3;
  x20 = seed4; x21 = seed5; x22 = seed6;
  
  /* Ensure seeds are in valid range */
  const int m = 2147483647; const int m2 = 2145483479;
  if (x10 < 1) x10 = 1; if (x10 >= m) x10 = x10 % m;
  if (x11 < 1) x11 = 1; if (x11 >= m) x11 = x11 % m;
  if (x12 < 1) x12 = 1; if (x12 >= m) x12 = x12 % m;
  if (x20 < 1) x20 = 1; if (x20 >= m2) x20 = x20 % m2;
  if (x21 < 1) x21 = 1; if (x21 >= m2) x21 = x21 % m2;
  if (x22 < 1) x22 = 1; if (x22 >= m2) x22 = x22 % m2;
}

/* Set random seed from a single integer (simpler interface) */
void set_random_seed_simple(int seed)
{
  /* Use seed to generate 6 different initial values */
  int s1 = seed;
  int s2 = (s1 * 1103515245 + 12345) & 0x7fffffff;
  int s3 = (s2 * 1103515245 + 12345) & 0x7fffffff;
  int s4 = (s3 * 1103515245 + 12345) & 0x7fffffff;
  int s5 = (s4 * 1103515245 + 12345) & 0x7fffffff;
  int s6 = (s5 * 1103515245 + 12345) & 0x7fffffff;
  
  set_random_seed(s1 % 2147483647 + 1, 
                  s2 % 2147483647 + 1, 
                  s3 % 2147483647 + 1,
                  s4 % 2145483479 + 1,
                  s5 % 2145483479 + 1,
                  s6 % 2145483479 + 1);
}

double rando()
 {
  const int m = 2147483647; const int m2 = 2145483479; 
  const int a12= 63308; const int q12=33921; const int r12=12979; 
  const int a13=-183326; const int q13=11714; const int r13=2883; 
  const int a21= 86098; const int q21=24919; const int r21= 7417; 
  const int a23=-539608; const int q23= 3976; const int r23=2071;
  const double invm = 4.656612873077393e-10;
  int h, p12, p13, p21, p23;
  h = x10/q13; p13 = -a13*(x10-h*q13)-h*r13;
  h = x11/q12; p12 = a12*(x11-h*q12)-h*r12;
  if (p13 < 0) p13 = p13 + m; if (p12 < 0) p12 = p12 + m;
  x10 = x11; x11 = x12; x12 = p12-p13; if (x12 < 0) x12 = x12 + m;
  h = x20/q23; p23 = -a23*(x20-h*q23)-h*r23;
  h = x22/q21; p21 = a21*(x22-h*q21)-h*r21;
  if (p23 < 0) p23 = p23 + m2; if (p21 < 0) p21 = p21 + m2;
  x20 = x21; x21 = x22; x22 = p21-p23; if(x22 < 0) x22 = x22 + m2;
  if (x12 < x22) h = x12 - x22 + m; else h = x12 - x22;
  if (h == 0) return(1.0); else return(h*invm);
 }

int unif(int low, int high)
 {return low + (int)((double)(high - low + 1) * rando()) ;}

void transpose(int *a, int *b) {int temp = *a; *a = *b; *b = temp;}

double min_d(double a, double b) {if (a < b) return(a); else return(b);}

double cube(double x) {return x*x*x;}

/*--------------------------------------------------------------*/
/* compute the cost difference                                  */
/*--------------------------------------------------------------*/
double compute_delta(int n, type_matrix a, type_matrix b,
                   type_vector p, int i, int j)
 {
  double d;
  int k;
  d = (a[i][i]-a[j][j])*(b[p[j]][p[j]]-b[p[i]][p[i]]) +
      (a[i][j]-a[j][i])*(b[p[j]][p[i]]-b[p[i]][p[j]]);
  for (k = 0; k < n; k = k + 1) if (k!=i && k!=j)
    d = d + (a[k][i]-a[k][j])*(b[p[k]][p[j]]-b[p[k]][p[i]]) +
            (a[i][k]-a[j][k])*(b[p[j]][p[k]]-b[p[i]][p[k]]);
  return(d);
 }

/*--------------------------------------------------------------*/
/* Idem, but the value of delta[i][j] is known                  */
/*--------------------------------------------------------------*/
double compute_delta_part(type_matrix a, type_matrix b,
                        type_vector p, type_matrix delta, 
                        int i, int j, int r, int s)
  {return(delta[i][j]+(a[r][i]-a[r][j]+a[s][j]-a[s][i])*
     (b[p[s]][p[i]]-b[p[s]][p[j]]+b[p[r]][p[j]]-b[p[r]][p[i]])+
     (a[i][r]-a[j][r]+a[j][s]-a[i][s])*
     (b[p[i]][p[s]]-b[p[j]][p[s]]+b[p[j]][p[r]]-b[p[i]][p[r]]) );
  }

void tabu_search(int n,                  /* problem size */
                 type_matrix a,          /* flows matrix */
                 type_matrix b,          /* distance matrix */
                 type_vector best_sol,   /* best solution found */
                 double *best_cost,      /* cost */
                 int tabu_duration,      
                 int aspiration,         
                 int nr_iterations,
                 clock_t start_time,     /* start time for timing */
                 FILE* log_file,         /* file to log improvements */
                 double time_limit)      /* time limit in seconds (0 = no limit) */
           
 
 {type_vector p;                        /* current solution */
  type_matrix delta;                    /* store move costs */
  int** tabu_list;                      /* tabu status */
  int current_iteration;                /* current iteration */
  double current_cost;                  /* current sol. value */
  int i, j, k, i_retained, j_retained;  /* indices */
  double min_delta;                     /* retained move cost */
  int autorized;                        /* move not tabu? */
  int aspired;                          /* move forced? */
  int already_aspired;                  /* in case many moves forced */
  double elapsed_time;                  /* time elapsed */
  int time_limit_reached = FALSE;       /* flag for time limit */

  /***************** dynamic memory allocation *******************/
  p = (int*)calloc(n, sizeof(int));
  delta = (double**)calloc(n,sizeof(double*));
  for (i = 0; i < n; i = i+1) delta[i] = (double*)calloc(n, sizeof(double));
  tabu_list = (int**)calloc(n,sizeof(int*));
  for (i = 0; i < n; i = i+1) tabu_list[i] = (int*)calloc(n, sizeof(int));

  /************** current solution initialization ****************/
  for (i = 0; i < n; i = i + 1) p[i] = best_sol[i];

  /********** initialization of current solution value ***********/
  current_cost = 0.0;
  for (i = 0; i < n; i = i + 1) for (j = 0; j < n; j = j + 1)
   {current_cost = current_cost + a[i][j] * b[p[i]][p[j]];
    if (i < j) {delta[i][j] = compute_delta(n, a, b, p, i, j);};
   };
  *best_cost = current_cost;

  /****************** tabu list initialization *******************/
  for (i = 0; i < n; i = i + 1) for (j = 0; j < n; j = j+1)
    tabu_list[i][j] = -(n*i + j);

  /******************** main tabu search loop ********************/
  for (current_iteration = 1; current_iteration <= nr_iterations; 
       current_iteration = current_iteration + 1)
   {
    /* Check time limit */
    if (time_limit > 0.0) {
      elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
      if (elapsed_time >= time_limit) {
        time_limit_reached = TRUE;
        printf("Time limit (%.2f s) reached at iteration %d\n", time_limit, current_iteration);
        break;
      }
    }
    
    i_retained = -1;       
    j_retained = -1;
    min_delta = infinite;
    already_aspired = FALSE;
    
    for (i = 0; i < n-1; i = i + 1) 
      for (j = i+1; j < n; j = j+1)
       {autorized = (tabu_list[i][p[j]] < current_iteration) || 
                    (tabu_list[j][p[i]] < current_iteration);

        aspired =
         (tabu_list[i][p[j]] < current_iteration-aspiration)||
         (tabu_list[j][p[i]] < current_iteration-aspiration)||
         (current_cost + delta[i][j] < *best_cost - 0.000001);               

        if ((aspired && !already_aspired) || 
           (aspired && already_aspired &&    
            (delta[i][j] < min_delta)   ) || 
           (!aspired && !already_aspired &&  
            (delta[i][j] < min_delta) && autorized))
          {i_retained = i; j_retained = j;
           min_delta = delta[i][j];
           if (aspired) {already_aspired = TRUE;};
          };
       };

    if (i_retained == -1) printf("All moves are tabu! \n"); 
    else 
     {
      transpose(&p[i_retained], &p[j_retained]);
      current_cost = current_cost + delta[i_retained][j_retained];
      
      tabu_list[i_retained][p[j_retained]] = 
        current_iteration + (int)(cube(rando())*tabu_duration);
      tabu_list[j_retained][p[i_retained]] = 
         current_iteration + (int)(cube(rando())*tabu_duration);

      if (current_cost < *best_cost)
       {
        /* Calculate elapsed time */
        elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        
        /* Print to console */
        printf("New best cost: %.6f at iteration %d (time: %.6f s)\n", 
               current_cost, current_iteration, elapsed_time);
        
        /* Log to file */
        if (log_file != NULL) {
          fprintf(log_file, "%.6f %.6f\n", elapsed_time, current_cost);
          fflush(log_file);  /* Ensure immediate write */
        }
        
        *best_cost = current_cost;
        for (k = 0; k < n; k = k+1) best_sol[k] = p[k];
       };

      for (i = 0; i < n-1; i = i+1) for (j = i+1; j < n; j = j+1)
        if (i != i_retained && i != j_retained && 
            j != i_retained && j != j_retained)
         {delta[i][j] = 
            compute_delta_part(a, b, p, delta, 
                               i, j, i_retained, j_retained);}
        else
         {delta[i][j] = compute_delta(n, a, b, p, i, j);};
     };
      
   }; 
  free(p);
  for (i=0; i < n; i = i+1) free(delta[i]); free(delta);
  for (i=0; i < n; i = i+1) free(tabu_list[i]); free(tabu_list);
} 

void generate_random_solution(int n, type_vector  p)
 {int i;
  for (i = 0; i < n;   i++) p[i] = i;
  for (i = 0; i < n-1; i++) transpose(&p[i], &p[unif(i, n-1)]);
 }

/* Helper to extract instance name */
void extract_instance_name(const char* filepath, char* output) {
    const char *last_slash = strrchr(filepath, '/');
    const char *last_backslash = strrchr(filepath, '\\');
    const char *filename_start = filepath;
    char *dot;

    /* Find start of filename (handle Linux/Windows paths) */
    if (last_slash != NULL && last_slash > last_backslash) {
        filename_start = last_slash + 1;
    } else if (last_backslash != NULL) {
        filename_start = last_backslash + 1;
    }

    /* Copy to output buffer */
    strcpy(output, filename_start);

    /* Remove extension */
    dot = strrchr(output, '.');
    if (dot != NULL) {
        *dot = '\0';
    }
}

int main(int argc, char *argv[])
 {
  /* --- Configuration --- */

  /* --------------------- */

  int n;                    
  type_matrix a, b;         
  type_vector solution;     
  double cost;              
  int no_res;
  
  double global_best_cost = infinite; 
  clock_t start_time, end_time;
  double total_time_elapsed;

  FILE* data_file;
  FILE* result_file;
  FILE* improvement_log;
  char* file_path_arg;
  char instance_name[256]; /* Buffer for extracted name */
  char log_filename[300];  /* Buffer for log filename */
  int i, j;
  char bidon[1000];
  int random_seed = 0;  /* Random seed, 0 means use time */
  double time_limit = 0.0;  /* Time limit in seconds, 0 = no limit */

  if (argc < 2) {
      printf("Usage: %s <data_file_path> [random_seed] [time_limit]\n", argv[0]);
      printf("  random_seed: optional integer seed (default: use current time)\n");
      printf("  time_limit: optional time limit in seconds (default: no limit)\n");
      return 1;
  }
  file_path_arg = argv[1];
  
  /* Parse random seed if provided */
  if (argc >= 3) {
      random_seed = atoi(argv[2]);
      printf("Using random seed: %d\n", random_seed);
  } else {
      /* Use current time as seed if not provided */
      random_seed = (int)time(NULL);
      printf("Using time-based random seed: %d\n", random_seed);
  }
  
  /* Parse time limit if provided */
  if (argc >= 4) {
      time_limit = atof(argv[3]);
      if (time_limit > 0.0) {
          printf("Time limit set to: %.2f seconds\n", time_limit);
      }
  }
  
  /* Initialize random number generator */
  set_random_seed_simple(random_seed);

  /* 1. Extract instance name */
  extract_instance_name(file_path_arg, instance_name);

  printf("Running Robust Tabu Search...\n");
  printf("Full Path: %s\n", file_path_arg);
  printf("Instance Name: %s\n", instance_name);

  data_file = fopen(file_path_arg,"r");
  if (data_file == NULL) {
      printf("Error: Could not open file %s\n", file_path_arg);
      return 1;
  }

  // fscanf(data_file,"%d%lf", &n, &opt);
  fscanf(data_file,"%d", &n);
  fscanf(data_file,"%[^\n]", bidon);

  /****************** Memory Allocation ******************/
  solution = (int*)calloc(n, sizeof(int));
  a = (double**)calloc(n, sizeof(double*));
  for (i = 0; i < n; i = i+1) a[i] = (double*)calloc(n, sizeof(double));
  b = (double**)calloc(n,sizeof(double*));
  for (i = 0; i < n; i = i+1) b[i] = (double*)calloc(n, sizeof(double));

  for (i = 0; i < n; i++) for (j = 0; j < n; j = j+1)
    fscanf(data_file,"%lf", &a[i][j]);

  for (i = 0; i < n; i = i+1) for (j = 0; j < n; j = j+1)
    fscanf(data_file,"%lf", &b[i][j]);
    
  fclose(data_file);

  /* Open improvement log file */
  sprintf(log_filename, "results/baseline/%s_rots_improvements.txt", instance_name);
  improvement_log = fopen(log_filename, "w");
  if (improvement_log == NULL) {
      printf("Warning: Could not open improvement log file %s\n", log_filename);
  } else {
      fprintf(improvement_log, "# Time(s) ObjectiveValue\n");
      fflush(improvement_log);
  }

  int nr_iterations = 100000 * n;
  int nr_resolutions = 1;
  
  /* --- START TIMING --- */
  start_time = clock();

  for(no_res = 1; no_res <= nr_resolutions; no_res++)
   {
    generate_random_solution(n, solution);

    tabu_search(n, a, b,                     
               solution, &cost,              
               8*n, n*n*5,                   
               nr_iterations,
               start_time,
               improvement_log,
               time_limit);               

    if (cost < global_best_cost) {
        global_best_cost = cost;
    }

    printf("Trial %d/%d Cost: %f\n", no_res, nr_resolutions, cost);
    somme_sol += cost;
   }
   
  /* --- END TIMING --- */
  end_time = clock();
  total_time_elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("\n--- Final Results ---\n");
  printf("Instance: %s\n", instance_name);
  printf("Global Best Cost: %f\n", global_best_cost);
  printf("Total Time: %f seconds\n", total_time_elapsed);

  /* Write final best solution and end time to improvement log */
  if (improvement_log != NULL) {
      fprintf(improvement_log, "%.6f %.6f\n", total_time_elapsed, global_best_cost);
      fflush(improvement_log);
      fclose(improvement_log);
      printf("Improvement log saved to %s\n", log_filename);
  }

  /* Write result to result.txt (Append Mode) */
  result_file = fopen("results/baseline/tabou_qap2_result.txt", "a");
  if (result_file == NULL) {
      printf("Error opening result.txt for writing.\n");
  } else {
      /* Format: instance_name, time, best_obj */
      fprintf(result_file, "%s %.6f %.6f\n", instance_name, total_time_elapsed, global_best_cost);
      fclose(result_file);
      printf("Result saved to tabou_qap2_result.txt\n");
  }

  free(solution);
  for (i = 0; i < n; i = i+1) free(b[i]);
  free(b);
  for (i = 0; i < n; i = i+1) free(a[i]);
  free(a);

  return EXIT_SUCCESS;
 }
