#ifndef __G4C_H__
#define __G4C_H__

typedef struct {
	int stream;
} g4c_async_t;

#ifdef __cplusplus
extern "C" {
#endif

	int g4c_init(void);
	void g4c_exit(void);

	void* g4c_malloc(size_t sz);
	void g4c_free(void *p);

	int g4c_do_stuff_sync(void *in, void *out, int n);
	int g4c_do_stuff_async(void *in, void *out, int n, g4c_async_t *asyncdata);

	int g4c_check_async_done(g4c_async_t *asyncdata);

#ifdef __cplusplus
}
#endif	

#endif