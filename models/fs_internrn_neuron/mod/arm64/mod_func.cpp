#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _Ih_reg(void);
extern void _kd_reg(void);
extern void _kdr_reg(void);
extern void _nas_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"Ih.mod\"");
    fprintf(stderr, " \"kd.mod\"");
    fprintf(stderr, " \"kdr.mod\"");
    fprintf(stderr, " \"nas.mod\"");
    fprintf(stderr, "\n");
  }
  _Ih_reg();
  _kd_reg();
  _kdr_reg();
  _nas_reg();
}

#if defined(__cplusplus)
}
#endif
