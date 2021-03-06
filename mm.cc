#include <set>
#include <vector>
#include <cstddef>

using namespace std;

#include <stdint.h>

#include "mm.hh"
#include "mm.h"

static vector<MMContext> mmcontexts;

void lock_mmcontexts()
{
    ;
}

void unlock_mmcontexts()
{
    ;
}

/**
 * base_addr must be 2^unit_order aligned;
 * size must be able to be divided by 2^unit_oder;
 *
 */
int create_mm_context(uint64_t base_addr, uint64_t size, uint32_t unit_order)
{
    lock_mmcontexts();
    mmcontexts.push_back(MMContext());
    MMContext &mmc = mmcontexts.back();
    mmc.index = mmcontexts.size()-1;
    unlock_mmcontexts();

    mmc.base_addr = base_addr;
    mmc.size = size;

    mmc.unit_size = 0x1 << unit_order;
    mmc.unit_shift = unit_order;
    mmc.unit_mask = (uint64_t)(0xffffffffffffffff)<<mmc.unit_shift;
    mmc.nr_units = (uint32_t)(size >> unit_order);

    mmc.order_begin = unit_order;
    mmc.order_end = g4c_msb_u64(size);
    mmc.nr_orders = mmc.order_end - mmc.order_begin +1;

    uint32_t chunk_idx = 0;
    for (uint32_t i=mmc.order_begin; i<=mmc.order_end; i++) {
	set<uint32_t> free_set;		
	uint64_t sz = ((uint64_t)0x1)<<i;
		
	if ((sz & mmc.size) != 0) {
	    free_set.insert(chunk_idx);
	    chunk_idx += (sz>>unit_order);
	}
	mmc.free_chunks.push_back(free_set);
    }	

    return mmc.index;
}

bool release_mm_context(int hdl)
{
    if (hdl < mmcontexts.size() && hdl >= 0) {
	mmcontexts[hdl].index = -1;
	return true;
    }
    return false;
}


uint64_t alloc_region(int mmc_idx, uint64_t size)
{
    if (mmc_idx < 0 || mmc_idx >= mmcontexts.size())
	return 0;
    MMContext &mmc = mmcontexts[mmc_idx];

    if (mmc.index < 0)
	return 0;

    size = g4c_round_up(size, mmc.unit_size);

    uint32_t order = g4c_msb_u64(size);

    if ((((uint64_t)0x1)<<order) != size) 
	size = ((uint64_t)0x1)<<(++order);
	

    if (order > mmc.order_end)
	return 0;

    for (int i = order; i <= mmc.order_end; i++)
    {
	set<uint32_t> &chunks = mmc.free_chunks[i-mmc.order_begin];
	if (!chunks.empty())
	{
	    set<uint32_t>::iterator ite = chunks.begin();
	    uint32_t unit_idx = *ite;

	    chunks.erase(ite);
	    mmc.allocated_chunks.insert(
		MemAllocInfo(unit_idx, order));
	    uint64_t addr = (((uint64_t)unit_idx)<<mmc.unit_shift)
		+ mmc.base_addr;
	    for (int cur_od = i-1; cur_od >= order; cur_od--)
	    {
		uint32_t idx = unit_idx +
		    (((uint32_t)0x1)<<(
			cur_od-mmc.order_begin));
		mmc.free_chunks[
		    cur_od-mmc.order_begin].insert(idx);				
	    }
	    return addr;
	}
    }

    return 0;
}

uint32_t paired_chunk_idx(uint32_t myidx, uint32_t relative_order)
{
    uint32_t idx = myidx >> relative_order;
    if ((idx & (uint32_t)0x1) != 0)
	return (idx-1)<<relative_order;
    else
	return (idx+1)<<relative_order;
}

bool free_region(int mmc_idx, uint64_t addr)
{	
    if (mmc_idx < 0 || mmc_idx >= mmcontexts.size())
	return false;
    MMContext &mmc = mmcontexts[mmc_idx];

    if (mmc.index < 0)
	return false;

    addr = g4c_round_down(addr, mmc.unit_size);
    uint32_t unit_idx = (addr - mmc.base_addr)>>mmc.unit_shift;
    set<MemAllocInfo, MemAllocInfoComp>::iterator it
	= mmc.allocated_chunks.find(MemAllocInfo(unit_idx, 0));

    if (it == mmc.allocated_chunks.end())
	return false;

    MemAllocInfo mai = *it;
    mmc.allocated_chunks.erase(it);

    int relative_order = mai.order - mmc.order_begin;
    uint32_t paired_idx = paired_chunk_idx(unit_idx, relative_order);
    set<uint32_t> *chunks = &(mmc.free_chunks[relative_order]);
    set<uint32_t>::iterator ite = chunks->find(paired_idx);
    while(ite != chunks->end() && relative_order < mmc.nr_orders-1) {
	chunks->erase(ite);

	relative_order++;
	unit_idx = unit_idx > paired_idx? paired_idx:unit_idx;
	paired_idx = paired_chunk_idx(unit_idx, relative_order);
	chunks = &(mmc.free_chunks[relative_order]);
	ite = chunks->find(paired_idx);
    }
    chunks->insert(unit_idx);
    return true;
}


// Wrappers for C and externl use

extern "C"
int g4c_new_mm_handle(void *base_addr, size_t total_size, unsigned int unit_order)
{
    return create_mm_context((uint64_t)base_addr, (uint64_t)total_size, (uint32_t)unit_order);
}

extern "C"
void *g4c_alloc_mem(int mm_handle, size_t size)
{
    return (void*)alloc_region(mm_handle, (uint64_t)size);
}

extern "C"
int g4c_free_mem(int mm_handle, void *addr)
{
    return free_region(mm_handle, (uint64_t)addr)?0:G4C_EMM;
}

extern "C"
int g4c_release_mm_handle(int mm_handle)
{
    return release_mm_context(mm_handle)?0:G4C_EMM;
}



#ifdef _G4C_MM_TEST_

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define two_pow(shift) (0x1<<(shift))
#define str_bool(b) ((b)?"T":"F")

void dump_mai(MemAllocInfo mai)
{
    printf("\t\tstart_unit: 0x%08x, order: %4u\n", mai.start_unit, mai.order);
}

void dump_u32(uint32_t u)
{
    printf("0x%08x ", u);
}

void dump_mmcontext(MMContext &mmc)
{
    printf("MMContext:\n"
	   "\tbase_addr: 0x%016lx  size: 0x%016lx\n"
	   "\tunit_size: 0x%08x  unit_shift: %2u  unit_mask: 0x%016lx  nr_units: 0x%08x\n"
	   "\tnr_orders: %2u  order_begin: %2u  order_end: %2u\n",
	   mmc.base_addr, mmc.size, mmc.unit_size, mmc.unit_shift, mmc.unit_mask, mmc.nr_units,
	   mmc.nr_orders, mmc.order_begin, mmc.order_end);

    printf("\tallocated_chunks:\n");
    for_each(mmc.allocated_chunks.begin(), mmc.allocated_chunks.end(), dump_mai);

    printf("\tfree_chunks:\n");
    vector<set<uint32_t> >::iterator ite = mmc.free_chunks.begin();
    uint32_t order = mmc.order_begin;
    for (; ite != mmc.free_chunks.end(); ++ite)
    {
	printf("\t\tOrder %2u, 0x%x units free chunks:\n\t\t", order,
	       two_pow(order - mmc.order_begin));
	for_each(ite->begin(), ite->end(), dump_u32);
	printf("\n");
	order++;
    }
}

int main()
{
    int hdl = create_mm_context(0x10000000, 0x10000000, 8);

    dump_mmcontext(mmcontexts[hdl]);

    uint64_t a1 = alloc_region(hdl, two_pow(16)|two_pow(15));
    dump_mmcontext(mmcontexts[hdl]);
    uint64_t a2 = alloc_region(hdl, two_pow(17));
    dump_mmcontext(mmcontexts[hdl]);
    uint64_t a3 = alloc_region(hdl, two_pow(14));
    dump_mmcontext(mmcontexts[hdl]);
	
    printf("\nAddr: 0x%016lx, 0x%016lx, 0x%016lx\n\n", a1, a2, a3);

    bool b1 = free_region(hdl, a1);
    dump_mmcontext(mmcontexts[hdl]);
    bool b2 = free_region(hdl, a2);
    dump_mmcontext(mmcontexts[hdl]);
    bool b3 = free_region(hdl, a3);
    dump_mmcontext(mmcontexts[hdl]);

    printf("Free results: %s, %s, %s\n", str_bool(b1), str_bool(b2), str_bool(b3));
}

#endif
