#ifndef AOP_PLH
#define AOP_PLH

#ifdef __cplusplus
extern "C" {
#endif

#include <R_ext/RS.h>  /* for F77_NAME */

  /* ----- Setup / teardown ----- */

  /* Set K, p, T */
  void read_dimensions_pl_arp(const int *num_categ,
                              const int *num_regr,
                              const int *num_time);

  /* Set AR order p (>=1) */
  void read_pl_ar_order_pl_arp(const int *p);

  /* Set max pairwise lag L (>=1) */
  void read_order_pair_lik_pl_arp(const int *L);

  /* Allocate / free internal buffers */
  void allocate_pl_arp(void);
  void deallocate_pl_arp(void);

  /* Data and covariates */
  void read_data_pl_arp(const int *data_input);
  void read_covariates_pl_arp(const double *design_matrix_input);

  /* ----- Pairwise log-likelihood + gradient + per-time score ----- */

  /* Average log pairwise probability (mean over included pairs) */
  void logPair_aop_ARp_Rcall_pl_arp(const double *param, double *out);

  /* Gradient of the average log pairwise probability */
  void gradient_logPair_aop_ARp_pl_arp(const double *param, double *gradient);

  /* Per-time score contributions (T x num_param, column-major) */
  void score_by_t_logPair_aop_ARp_pl_arp(const double *param, double *scores);

#ifdef __cplusplus
}
#endif

#endif /* AOP_PL_ARP_H */
