#pragma once

#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>

#include <complex>

#include "types.h"
#include "lobank.h"
#include "firdecimate.h"

/*
 * USRP UHD wrapper class to send and receive data from radio
 */
class Radio
{
	public:
	typedef std::shared_ptr<Radio> sptr;
	Radio(int *_stop_signal);
	void set_rf_frequency(double freq);
	void set_dsp_frequency(double freq);
	void set_dsp_rate(double freq);
	void zero_time();
	void check_lock();

	double rf_freq,dsp_freq,target_freq;
	size_t channel;
	int *stop_signal;

	uhd::usrp::multi_usrp::sptr usrp;
	uhd::tx_streamer::sptr tx_stream;
	uhd::rx_streamer::sptr rx_stream;

	size_t tx_spb,rx_spb;
	uhd::tx_metadata_t tx_md;
	uhd::rx_metadata_t rx_md;
	void transmit();
	void init_receive();
	void receive(std::queue<complex64> &out_queue);
	void transmit_stop();
	void receive_stop();
	size_t num_transmit;
	size_t num_receive;
	double rx_timeout;
	std::vector<complex64> tx_buf,rx_buf;
	std::vector<complex64 *> tx_buf_ptrs, rx_buf_ptrs;
};



