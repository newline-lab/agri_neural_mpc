#include <l4casadi.hpp>

L4CasADi l4casadi("/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/_l4c_generated", "car", 20, 3, 20, 1, "cuda", true, true, true, false, true, false);

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

// Function car

static const casadi_int car_s_in0[3] = { 20, 3, 1};
static const casadi_int car_s_out0[3] = { 20, 1, 1};

// Only single input, single output is supported at the moment
CASADI_SYMBOL_EXPORT casadi_int car_n_in(void) { return 1;}
CASADI_SYMBOL_EXPORT casadi_int car_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const casadi_int* car_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return car_s_in0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* car_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return car_s_out0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int car(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.forward(arg[0], res[0]);
  return 0;
}

// Jacobian car

CASADI_SYMBOL_EXPORT casadi_int jac_car_n_in(void) { return 2;}
CASADI_SYMBOL_EXPORT casadi_int jac_car_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT int jac_car(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.jac(arg[0], res[0]);
  return 0;
}

// Sparse output if batched.
static const casadi_int jac_car_s_out0[123] = { 20, 60, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

CASADI_SYMBOL_EXPORT const casadi_int* jac_car_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return jac_car_s_out0;
    default: return 0;
  }
}


// adj1 car

CASADI_SYMBOL_EXPORT casadi_int adj1_car_n_in(void) { return 3;}
CASADI_SYMBOL_EXPORT casadi_int adj1_car_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT int adj1_car(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  // adj1 [i0, out_o0, adj_o0] -> [out_adj_i0]
  l4casadi.adj1(arg[0], arg[2], res[0]);
  return 0;
}


// jac_adj1 car

CASADI_SYMBOL_EXPORT casadi_int jac_adj1_car_n_in(void) { return 4;}
CASADI_SYMBOL_EXPORT casadi_int jac_adj1_car_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT int jac_adj1_car(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  // jac_adj1 [i0, out_o0, adj_o0, out_adj_i0] -> [jac_adj_i0_i0, jac_adj_i0_out_o0, jac_adj_i0_adj_o0]
  if (res[1] != NULL) {
    l4casadi.invalid_argument("jac_adj_i0_out_o0 is not provided by L4CasADi. If you need this feature, please contact the L4CasADi developer.");
  }
  if (res[2] != NULL) {
    l4casadi.invalid_argument("jac_adj_i0_adj_o0 is not provided by L4CasADi. If you need this feature, please contact the L4CasADi developer.");
  }
  if (res[0] == NULL) {
    l4casadi.invalid_argument("L4CasADi can only provide jac_adj_i0_i0 for jac_adj1_car function. If you need this feature, please contact the L4CasADi developer.");
  }
  l4casadi.jac_adj1(arg[0], arg[2], res[0]);
  return 0;
}

// Sparse output if batched.
static const casadi_int jac_adj1_car_s_out0[243] = { 60, 60, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 0, 20, 40, 1, 21, 41, 2, 22, 42, 3, 23, 43, 4, 24, 44, 5, 25, 45, 6, 26, 46, 7, 27, 47, 8, 28, 48, 9, 29, 49, 10, 30, 50, 11, 31, 51, 12, 32, 52, 13, 33, 53, 14, 34, 54, 15, 35, 55, 16, 36, 56, 17, 37, 57, 18, 38, 58, 19, 39, 59, 0, 20, 40, 1, 21, 41, 2, 22, 42, 3, 23, 43, 4, 24, 44, 5, 25, 45, 6, 26, 46, 7, 27, 47, 8, 28, 48, 9, 29, 49, 10, 30, 50, 11, 31, 51, 12, 32, 52, 13, 33, 53, 14, 34, 54, 15, 35, 55, 16, 36, 56, 17, 37, 57, 18, 38, 58, 19, 39, 59, 0, 20, 40, 1, 21, 41, 2, 22, 42, 3, 23, 43, 4, 24, 44, 5, 25, 45, 6, 26, 46, 7, 27, 47, 8, 28, 48, 9, 29, 49, 10, 30, 50, 11, 31, 51, 12, 32, 52, 13, 33, 53, 14, 34, 54, 15, 35, 55, 16, 36, 56, 17, 37, 57, 18, 38, 58, 19, 39, 59};
static const casadi_int jac_adj1_car_s_out23[3] = { 20 * 3, 20 * 1, 1};

CASADI_SYMBOL_EXPORT const casadi_int* jac_adj1_car_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return jac_adj1_car_s_out0;
    case 1: return jac_adj1_car_s_out23;
    case 2: return jac_adj1_car_s_out23;
    default: return 0;
  }
}



#ifdef __cplusplus
} /* extern "C" */
#endif