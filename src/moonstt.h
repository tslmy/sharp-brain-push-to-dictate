#ifndef MOONSTT_H
#define MOONSTT_H
#include <stddef.h>
#include <stdint.h>

typedef struct moonstt moonstt;

/* dir holds model.bin and tok.bin (see tools/convert.py) */
moonstt *moonstt_load(const char *dir);
void moonstt_free(moonstt *m);
/* 16 kHz mono int16 in; UTF-8 text out.  Returns number of tokens. */
int moonstt_transcribe(moonstt *m, const int16_t *pcm, int n, char *text, size_t cap, int verbose);

#endif
