#pragma once

#include "types.h"
#include "radio.h"
#include "lobank.h"
#include "channelizer.h"
#include "firdecimate.h"


void sig_int_handler(int arg);

class RXTX
{
	public:
		Radio::sptr radio;
		size_t nch;
		int *stop_signal;
		LOBank::sptr lobank_rx, lobank_tx;
		FIRDecimate::sptr fir;
		Channelizer channelizer;

		std::queue<complex64> recv_queue, transmit_queue;
		std::vector<std::queue<complex64> > channel_queue;

		RXTX(Radio::sptr _radio, FIRDecimate::sptr _fir, size_t _nch, int *_stop_signal);
		void one_thread();
		void multi_thread();
		void receive_thread();
		void transmit_thread();
};

