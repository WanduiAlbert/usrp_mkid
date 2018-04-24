#include <stdio.h>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <csignal>

#include "H5Cpp.h"

#include "usrp_mkid.h"
#include "radio.h"
#include "lobank.h"
#include "outputfile.h"

using namespace std;
namespace po = boost::program_options;

static int stop_signal_called;

void sig_int_handler(int arg)
{
	(void)arg;
	stop_signal_called = true;
}

RXTX::RXTX(Radio::sptr _radio, FIRDecimate::sptr _fir, size_t _nch, int *_stop_signal)
: radio(_radio), nch(_nch), stop_signal(_stop_signal), lobank_rx(new LOBank(nch)), lobank_tx(new LOBank(nch)), fir(_fir),
	channelizer(fir), channel_queue(nch)
{
	channelizer.set_lobank(lobank_rx);
	radio->check_lock();
	radio->zero_time();
}

void RXTX::multi_thread()
{
	boost::thread transmit_thread(&RXTX::transmit_thread,this);
	receive_thread();
	transmit_thread.join();
}

void RXTX::receive_thread()
{
	radio->init_receive();
	while(not *stop_signal) {
		cout << "receiving block...";
		radio->receive(recv_queue);
		cout << "ok" << endl;
		channelizer.channelize(recv_queue,channel_queue);
	}
	radio->receive_stop();
}

void RXTX::transmit_thread()
{
	while(not *stop_signal) {
		lobank_tx->get_sample_tx(radio->tx_buf);
		radio->transmit();
	}
	radio->transmit_stop();
}

int main(int argc, char **argv)
{
	double rf_freq, dsp_freq, dsp_rate;
	size_t nch;
	string tone_config, fir_config;
	po::options_description desc("Allowed options");
	stop_signal_called = false;
	desc.add_options()
		("help","help message")
		("lo-freq", po::value<double>(&rf_freq)->default_value(200e6), "LO frequency")
		("dsp_freq", po::value<double>(&dsp_freq)->default_value(10e6), "DSP frequency")
		("dsp-rate", po::value<double>(&dsp_rate)->default_value(1e6), "DSP sample rate")
		("nchannel", po::value<size_t>(&nch)->default_value(1),"Number of channels")
		("fir-config",po::value<string>(&fir_config)->default_value("fir.cfg"),"FIR configuration file")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc),vm);
	po::notify(vm);

	Radio::sptr radio(new Radio(&stop_signal_called));
	radio->set_rf_frequency(rf_freq);
	radio->set_dsp_frequency(dsp_freq);
	radio->set_dsp_rate(dsp_rate);

	FIRDecimate::sptr fir(new FIRDecimate(fir_config));

	std::signal(SIGINT,&sig_int_handler);
	cout << "Press ctrl+c to stop streaming..." << endl;

	RXTX rxtx(radio,fir,nch,&stop_signal_called);
	rxtx.multi_thread();
}


