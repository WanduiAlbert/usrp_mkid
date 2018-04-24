#pragma once

#include "types.h"
#include "lobank.h"
#include "firdecimate.h"

class Channelizer
{
	private:
		LOBank::sptr lobank;
	public:
		FIRDecimate::sptr firdecimate;
		size_t ntap,nch,decimate;
		std::vector<complex64> in_block;
		std::vector<complex64> lo_block;
		std::vector<std::vector<complex64> > work;

		Channelizer(FIRDecimate::sptr _firdecimate);
		void set_lobank(LOBank::sptr _lobank);
		void channelize(std::queue<complex64> &in_queue, std::vector<std::queue<complex64> > &out_queues);
};
