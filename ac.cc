#ifdef _G4C_AC_TEST_
#include <stdio.h>
#include <stdlib.h>
#endif

#include <stddef.h>
#include <string.h>

#include <errno.h>
#include "g4c.h"
#include "g4c_ac.h"

#include <cstdio>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>

using namespace std;

class ACState;

class ACState {
public:
    int id;
    ACState *prev;
    map<char, int> go;
    set<int> output;
    int failure;

    int transition[AC_ALPHABET_SIZE];

    ACState():id(0), prev(0), failure(-1) {}
    ACState(int sid):id(sid), prev(0), failure(-1) {}
    ACState(int sid, ACState *sprev):id(sid), prev(sprev), failure(-1) {}
    ~ACState() {}
};

class ACMachine {
public:
    vector<ACState*> states;
    char **patterns;
    int npatterns;

    ACMachine() {}
    ~ACMachine() {
	for (int i=0; i<states.size(); i++)
	    delete states[i];
	states.clear();
    }
};

class KMPMachine {
public:
    vector<int*> lpss;
    char **patterns;
    int npatterns;

    KMPMachine() {}
    ~KMPMachine() {
        for (int i = 0; i < lpss.size(); ++i) {
            delete lpss[i];
            lpss.clear();
        }
    }
};

void computeLPSArray(char *pat, int M, int *lps) {
    int len = 0;  // length of the previous longest prefix suffix
    int i;
    lps[0] = 0; // lps[0] is always 0
    i = 1;
    // the loop calculates lps[i] for i = 1 to M-1
    while (i < M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        }
        else {
            if (len != 0) {
                len = lps[len-1];
            }
            else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

static void
ac_build_lps(char **ptns, int nptns, KMPMachine *cppkmpm) {
    int i;
    vector<int*> &lpss = cppkmpm->lpss;
    for (i=0; i<nptns; i++) {
        char* pattern = ptns[i];
        //int M = strlen(pattern);
        int *lps = (int *)malloc(sizeof(int)*PATTERN_LENGTH);
        cppkmpm->lpss.push_back(lps);
        computeLPSArray(pattern, PATTERN_LENGTH, lps);
    }
}

static void
ac_build_goto(char *kws[], int n, ACMachine *acm)
{
    int sid = 1;
    int i;
    vector<ACState*> &states = acm->states;
	
    states.push_back(new ACState(0,0));	
    for (i=0; i<n; i++) {
	char *kw = kws[i];
	int nsid = 0, j = 0;
	ACState *s = states[0];

	while (kw[j] && s->go.find(kw[j]) != s->go.end()) {
	    nsid = s->go[kw[j]];
	    s = states[nsid];
	    j++;
	}

	while (kw[j]) {
	    ACState *ns = new ACState(sid++, s);
	    states.push_back(ns);
	    s->go[kw[j]] = ns->id;
	    s = ns;
	    j++;
	}

	s->output.insert(i);
    }

    // setup initial state goto function
    for (i=0; i<AC_ALPHABET_SIZE; i++) {
	if (states[0]->go.find(i) == states[0]->go.end())
	    states[0]->go[i] = 0;
    }
}

static void
ac_build_failure(ACMachine *acm)
{
    queue<int> q;
    ACState *s = acm->states[0];

    for (map<char, int>::iterator ite = s->go.begin();
	 ite != s->go.end(); ++ite) {
	if (ite->second != s->id)
	    q.push(ite->second);
	acm->states[ite->second]->failure = s->id; //0
    }

    while (!q.empty()) {
	int rid = q.front();
	ACState *r = acm->states[rid];
		
	q.pop();
	for (map<char, int>::iterator ite = r->go.begin();
	     ite != r->go.end(); ++ite) {
	    q.push(ite->second);

	    int fid = r->failure;
	    ACState *f = acm->states[fid];
	    while(f->go.find(ite->first) == f->go.end()) {
		fid = f->failure;
		f = acm->states[fid];
	    }
	    acm->states[ite->second]->failure = f->go[ite->first];
	    f = acm->states[f->go[ite->first]];
	    acm->states[ite->second]->output.insert(
		f->output.begin(),
		f->output.end());
	}
    }				
}

static void
ac_build_transition(ACMachine *acm)
{
    queue<int> q;
    ACState *s = acm->states[0];

    for (int i=0; i<AC_ALPHABET_SIZE; i++) {
	s->transition[i] = s->go[i];
	if (s->go[i] != s->id)
	    q.push(s->go[i]);
    }

    while (!q.empty()) {
	int rid = q.front();
	ACState *r = acm->states[rid];
	q.pop();

	for (int i=0; i<AC_ALPHABET_SIZE; i++) {
	    if (r->go.find(i) != r->go.end()) {
		q.push(r->go[i]);
		r->transition[i] = r->go[i];
	    } else
		r->transition[i] = acm->states[r->failure]->transition[i];
	}
    }		
}

extern "C" {
#include <stdlib.h>
#include <string.h>
}

extern "C" int
g4c_cpu_acm_match(g4c_kmp_t *dacm, uint8_t *data, int len, int patternNum)
{
   //printf("in cpu match\n");
//    int ret = 0, nid;
//    uint8_t c;
//
//    int cid = 0;
//    for (int i=0; i<len; i++) {
//	c = data[i];
//	nid = g4c_acm_htransitions(acm, cid)[c];
//	if (*g4c_acm_houtput(acm, cid)) {
//	    if (!ret || ret > *g4c_acm_houtput(acm, cid)) {
//		ret = *g4c_acm_houtput(acm, cid);
//	    }
//	}
//	cid = nid;
//    }

//    return ret;
    int outidx = 0;
    int *lps = g4c_kmp_hlpss(dacm, patternNum);
    char* pattern = g4c_kmp_hpatterns(dacm, patternNum);
    int patternLength = dacm->patternLengths[patternNum];
    int i = 0, j = 0;// index for txt[]
    while (i < len) {
        if (pattern[j] == data[i]) {
            j++;
            i++;
        }
        if (j == patternLength) {
//            printf("Found pattern at index %d \n", i-j);
            outidx = i-j;
            break;
//            j = lps[j-1];
        } else if (i < len && pattern[j] != data[i]) { // mismatch after j matches
            // Do not match lps[0..lps[j-1]] characters,
            // they will match anyway
            if (j != 0)
                j = lps[j-1];
            else
                i = i+1;
        }
    }
    //printf("done with cpu match\n");
    return outidx;
}

//extern "C" g4c_acm_t*
//g c_create_matcher(char **ptns, int nptns, int withdev, int stream)
//{
//    ACMachine *cppacm = new ACMachine();
//    if (!cppacm) {
//	fprintf(stderr, "Out of memory for C++ ACM\n");
//	return 0;
//    }
//
//    ac_build_goto(ptns, nptns, cppacm);
//    ac_build_failure(cppacm);
//    ac_build_transition(cppacm);
//
//    size_t trsz = cppacm->states.size()*AC_ALPHABET_SIZE*sizeof(int);
//    trsz = g4c_round_up(trsz, G4C_PAGE_SIZE);
//
//    size_t outsz = cppacm->states.size()*sizeof(int);
//    outsz = g4c_round_up(outsz, G4C_PAGE_SIZE);
//
//    size_t totalsz = G4C_PAGE_SIZE + trsz + outsz;
//    g4c_acm_t *acm = (g4c_acm_t*)g4c_alloc_page_lock_mem(totalsz);
//    void *dmem = 0;
//    if (withdev) {
//	dmem = g4c_alloc_dev_mem(totalsz);
//    }
//
//    if (!acm || (withdev && !dmem)) {
//	fprintf(stderr, "Out of mem for acm GPU memory or device mem "
//		"%p, %p, %lu\n", acm, dmem, totalsz);
//	return 0;
//    }
//
//    acm->mem = (void*)acm;
//    acm->devmem = dmem;
//    acm->memsz = totalsz;
//    acm->nstates = (int)cppacm->states.size();
//    acm->transitions = (int*)g4c_ptr_add(acm->mem, G4C_PAGE_SIZE);
//    acm->outputs = (int*)g4c_ptr_add(acm->transitions, trsz);
//    if (withdev) {
//	acm->dtransitions = (int*)g4c_ptr_add(acm->devmem, G4C_PAGE_SIZE);
//	acm->doutputs = (int*)g4c_ptr_add(acm->dtransitions, trsz);
//    }
//
//    for (int i=0; i<acm->nstates; i++) {
//	ACState *cpps = cppacm->states[i];
//	memcpy(g4c_acm_htransitions(acm, i),
//	       cpps->transition,
//	       sizeof(int)*AC_ALPHABET_SIZE);
//	if (cpps->output.size()) {
//	    set<int>::iterator minout =
//		min_element(cpps->output.begin(), cpps->output.end());
//	    *g4c_acm_houtput(acm, i) = (*minout) + 1;
//	} else {
//	    *g4c_acm_houtput(acm, i) = 0;
//	}
//    }
//
//    if (withdev) {
//	g4c_h2d_async(acm->mem, acm->devmem, acm->memsz, stream);
//	g4c_stream_sync(stream);
//    }
//
//    return acm;
//}

extern "C" g4c_kmp_t*
g4c_create_matcher(char **ptns, int nptns, int withdev, int stream)
{
    fprintf(stdout, "in g4c create matcher\n");
    KMPMachine *cppkmpm = new KMPMachine();
    if (!cppkmpm) {
        fprintf(stderr, "Out of memory for C++ KMPM\n");
        return 0;
    }
    cppkmpm->patterns = ptns;
    cppkmpm->npatterns = nptns;

    ac_build_lps(ptns, nptns, cppkmpm);

//    ac_build_goto(ptns, nptns, cppacm);
//    ac_build_failure(cppacm);
//    ac_build_transition(cppacm);

    size_t lpssz = cppkmpm->lpss.size()*PATTERN_LENGTH* sizeof(int);
//    size_t trsz = cppacm->states.size()*AC_ALPHABET_SIZE*sizeof(int);
    lpssz = g4c_round_up(lpssz, G4C_PAGE_SIZE);

//    size_t outsz = cppkmpm->lpss.size()*sizeof(int);
//    outsz = g4c_round_up(outsz, G4C_PAGE_SIZE);
    size_t patternssz = cppkmpm->npatterns*(PATTERN_LENGTH+1)*sizeof(char);
    patternssz == g4c_round_up(patternssz, G4C_PAGE_SIZE);

    size_t patternLengthsz = cppkmpm->npatterns*sizeof(int);
    patternLengthsz == g4c_round_up(patternLengthsz, G4C_PAGE_SIZE);

    size_t totalsz = G4C_PAGE_SIZE + lpssz + patternssz + patternLengthsz;
    g4c_kmp_t *acm = (g4c_kmp_t*)g4c_alloc_page_lock_mem(totalsz);
    void *dmem = 0;
    if (withdev) {
        dmem = g4c_alloc_dev_mem(totalsz);
    }

    if (!acm || (withdev && !dmem)) {
        fprintf(stderr, "Out of mem for acm GPU memory or device mem "
                "%p, %p, %lu\n", acm, dmem, totalsz);
        return 0;
    }
   printf("allocated memories\n");
    acm->mem = (void*)acm;
    acm->devmem = dmem;
    acm->memsz = totalsz;
    acm->nlpss = (int)cppkmpm->lpss.size();
//    acm->transitions = (int*)g4c_ptr_add(acm->mem, G4C_PAGE_SIZE);
//    acm->outputs = (int*)g4c_ptr_add(acm->transitions, trsz);
    acm->lspss = (int*)g4c_ptr_add(acm->mem, G4C_PAGE_SIZE);
    acm->patterns = (char*)g4c_ptr_add(acm->lspss, lpssz);
    acm->patternLengths = (int*)g4c_ptr_add(acm->patterns, patternssz);
    if (withdev) {
        acm->dlspss = (int*)g4c_ptr_add(acm->devmem, G4C_PAGE_SIZE);
	//printf("in if dev\n");
        acm->dpatterns = (char*)g4c_ptr_add(acm->dlspss, lpssz);
        acm->dPatternLengths = (int*)g4c_ptr_add(acm->dpatterns, patternssz);
    }
    //printf("out of the if\n");
    for (int i=0; i<acm->nlpss; i++) {
//        ACState *cpps = cppacm->states[i];
        //printf("in for loop: %d\n", i);
        int *lps = cppkmpm->lpss[i];
        //printf("lps: %d\n", lps[1]);
        char* pattern = cppkmpm->patterns[i];
	//printf("pattern: %s\n", pattern);
        memcpy(g4c_kmp_hlpss(acm, i), lps, sizeof(int)*PATTERN_LENGTH);
        //printf("copied: %d\n", g4c_kmp_hlpss(acm, i)[1]);
        //printf("memcpy of lps: %d\n", i);
//        if (cpps->output.size()) {
//            set<int>::iterator minout =
//                    min_element(cpps->output.begin(), cpps->output.end());
//            *g4c_acm_houtput(acm, i) = (*minout) + 1;
        memcpy(g4c_kmp_hpatterns(acm, i), pattern, sizeof(char)*(strlen(pattern)));
        acm->patternLengths[i] = strlen(pattern);
        //printf("memcpy of patterns: %d\n", i);
//        } else {
//            *g4c_acm_houtput(acm, i) = 0;
//        }
    }

    if (withdev) {
        g4c_h2d_async(acm->mem, acm->devmem, acm->memsz, stream);
        g4c_stream_sync(stream);
    }

    return acm;
}

extern "C" int
ac_build_machine(ac_machine_t *acm, char **patterns,
		 int npatterns, unsigned int memflags)
{
    size_t
	psz   = 0,  // total size of all pattern strings, including '\0'
	ppsz  = 0,  // total size of all pointers of pattern strings
	stsz  = 0,  // total size of all states
	trsz  = 0,  // total size of all transition table
	outsz = 0;  // total size of all output function table
    int i;		
    ACMachine cppacm;

    // Build C++ ACMachine
    ac_build_goto(patterns, npatterns, &cppacm);
    ac_build_failure(&cppacm);
    ac_build_transition(&cppacm);

    memset(acm, 0, sizeof(ac_machine_t));

    // easy settings
    acm->nstates = static_cast<int>(cppacm.states.size());
    acm->npatterns = npatterns;
    acm->memflags = memflags;

    // calculate all sizes
    for (i=0; i<npatterns; i++)
	psz += strlen(patterns[i])+1;
    ppsz = sizeof(char*)*npatterns;

    stsz = acm->nstates * sizeof(ac_state_t);
    stsz = g4c_round_up(stsz, G4C_PAGE_SIZE);
	
    trsz = acm->nstates * AC_ALPHABET_SIZE * sizeof(int);
    trsz = g4c_round_up(trsz, G4C_PAGE_SIZE);

    for (i=0; i<acm->nstates; i++) {
	acm->noutputs += cppacm.states[i]->output.size();
    }
    outsz = acm->noutputs * sizeof(int);
    outsz = g4c_round_up(outsz, G4C_PAGE_SIZE);
	
    acm->memsz = psz + ppsz + stsz + trsz + outsz;
    acm->memsz = g4c_round_up(acm->memsz, G4C_PAGE_SIZE);

    // memory allocation and assignment
    acm->mem = malloc(acm->memsz);
    if (acm->mem) {
	int *tmpoutput;
	char *ptn;
	int output_offset = 0;

	// default layout:
	//  --------------------
	//  | states ...       |
	//  +------------------+
	//  | transitions ...  |
	//  +------------------+
	//  | outputs ...      |
	//  +------------------+
	//  | pattern ptrs ... |
	//  +------------------+
	//  | patterns ...     |
	//  --------------------
	//
	acm->states = (ac_state_t*)acm->mem;
	acm->transitions = (int*)g4c_ptr_add(acm->mem, stsz);
	acm->outputs = (int*)g4c_ptr_add(acm->transitions, trsz);
	acm->patterns = (char**)g4c_ptr_add(acm->outputs, outsz);

	// copy each state's information
	tmpoutput = acm->outputs;
	for (i=0; i<acm->nstates; i++) {
	    ACState *cpps = cppacm.states[i];
	    ac_state_t *acs = acm->states + i;

	    acs->id = cpps->id;
	    acs->prev = (cpps->prev?cpps->prev->id:-1);
	    acs->noutput = (int)(cpps->output.size());
	    acs->output = (acs->noutput?output_offset:-1);

	    // copy output table
	    for (set<int>::iterator ite = cpps->output.begin();
		 ite != cpps->output.end();
		 ++ite)
	    {
		*tmpoutput = *ite;
		++tmpoutput;
		++output_offset;
	    }

	    // copy transition table
	    memcpy(acm_state_transitions(acm, i),
		   cpps->transition,
		   sizeof(int)*AC_ALPHABET_SIZE);
	}

	ptn = (char*)(acm->patterns + npatterns);
	for (i=0; i<npatterns; i++) {		       
	    strcpy(ptn, patterns[i]);
	    acm->patterns[i] = ptn;
	    ptn += strlen(patterns[i])+1;
	}

	// OK
	return 0;			
    }
	
    return -ENOMEM;
}

extern "C" void
ac_release_machine(ac_machine_t *acm)
{
    free(acm->mem);
}


extern "C" int
ac_match(char *str, int len, unsigned int *res, int once, ac_machine_t *acm)
{
    int nm = 0;

    char c;
    ac_state_t *st = acm_state(acm, 0);	
    for (int i=0; i<len; i++) {	       
	c = str[i];
	if (c < 0)
	    return -1;
	int nid = acm_state_transitions(acm, st->id)[c];
	st = acm_state(acm, nid);
	if (st->noutput > 0) {
	    if (res) {
		for (int j=0; j<st->noutput; j++) {
		    int ot = acm_state_output(acm, st->output)[j];
		    res[ot] = 0x80000000 |
			(i+1-strlen(acm_pattern(acm, ot)));
		}
	    }
	    if (!nm && once) {
		return st->noutput;
	    }
	    nm += st->noutput;
	}
    }
    return nm;
}


#ifdef _G4C_AC_TEST_

static void
dump_state(ACState *s, char* kws[])
{
    map<char, int>::iterator ite;
    set<int>::iterator oite;
	
    printf("S %3d, previous: %3d, failure: %3d\n",
	   s->id, (s->prev?s->prev->id:-1), s->failure);

    printf("\t%4lu Gotos: ", s->go.size());
    for (ite = s->go.begin(); ite != s->go.end(); ++ite) {
	printf("(%4d --%c--> %-4d) ", s->id, ite->first, ite->second);
    }
    printf("\n");

    printf("\t%4lu Outputs: \n", s->output.size());
    for (oite = s->output.begin(); oite != s->output.end(); ++oite) {
	printf("\t\t%s\n", kws[*oite]);
    }
    printf("\n");

    printf("\tTransitions: \n\t");
    for (int i=0; i<128; i++) {
	printf(" (%3d,%3d)", i, s->transition[i]);
    }
    printf("\n");
}

static void
dump_c_state(ac_state_t *s, ac_machine_t *acm)
{
    printf("State %d, previous: %d, #outputs: %d\n", s->id, s->prev, s->noutput);
    printf("\tOutputs:\n");
    for (int i=0; i<s->noutput; i++) {
	int ot = acm_state_output(acm, s->output)[i];
	printf("\t\t%3d: %s\n", ot, acm->patterns[ot]);
    }

    int *tr = acm_state_transitions(acm, s->id);
    printf("\tTransitions:\n");
    for (int i=0; i<AC_ALPHABET_SIZE; i++) {
	if (tr[i] != 0) {
	    printf("\t\t%d--%c-->%d\n", s->id, i, tr[i]);
	}
    }
}

static void
dump_c_acm(ac_machine_t *acm)
{
    printf("C version ACM:\n"
	   "\tmem=%p, size=%lu, flags=0X%08X\n"
	   "\tstates=%p, nstates=%d, sizeof(ac_state_t)=%lu\n"
	   "\ttransitions=%p, outputs=%p, noutputs=%d\n"
	   "\tpatterns=%p, npatterns=%d\n",
	   acm->mem, acm->memsz, acm->memflags,
	   acm->states, acm->nstates, sizeof(ac_state_t),
	   acm->transitions, acm->outputs, acm->noutputs,
	   acm->patterns, acm->npatterns);
    printf("States\n");
    for (int i=0; i<acm->nstates; i++)
	dump_c_state(acm->states+i, acm);	
}


int
main(int argc, char *argv[])
{
    ACMachine acm;
    vector<ACState*>::iterator ite;
    ac_machine_t cacm;

    ac_build_goto(argv+1, argc-2, &acm);
    ac_build_failure(&acm);
    ac_build_transition(&acm);

    ac_build_machine(&cacm, argv+1, argc-2, 0);
    dump_c_acm(&cacm);
	
// 	for (ite = acm.states.begin(); ite != acm.states.end(); ++ite)
// 		;//dump_state(*ite, argv+1);

    unsigned int *res = new unsigned int[argc];
    memset(res, 0, sizeof(unsigned int)*argc);
    int r = ac_match(argv[argc-1], strlen(argv[argc-1]), res, 0, &cacm);
    printf("Matches: %d\n", r);

    if (r > 0) {
	for (int i=0; i<=argc-2; i++) {
	    if (ac_res_found(res[i]))
		printf("Matched %s at %u.\n", argv[i+1], ac_res_location(res[i]));
	}
    }

    ac_release_machine(&cacm);
	
    return 0;
}

#endif
