#include "radio.h"

using namespace std;

typedef complex<float> complex32;

Radio::Radio(int *_stop_signal)
: stop_signal(_stop_signal), num_transmit(0),num_receive(0),tx_buf_ptrs(1),rx_buf_ptrs(1)
{
	rf_freq = 200e6;
	dsp_freq = 10e6;
	target_freq = rf_freq + dsp_freq;
	usrp = uhd::usrp::multi_usrp::make(string(""));
	cout << "Internal reference...\n" << endl;
	string ref("internal");
	usrp->set_clock_source(ref);
	cout << "Using Device: " << usrp->get_pp_string() << endl;

	channel = 0;
	usrp->set_tx_gain(35,channel);
	usrp->set_rx_gain(35,channel);

	string sample_local("fc32");
	string sample_otw("sc16");
	uhd::stream_args_t stream_args(sample_local,sample_otw);
	stream_args.channels.push_back(channel);

	tx_stream = usrp->get_tx_stream(stream_args);
	tx_spb = tx_stream->get_max_num_samps()*16;
	tx_buf.resize(tx_spb);
	tx_buf_ptrs[0] = &tx_buf[0];
	tx_md.start_of_burst = true;
	tx_md.end_of_burst = false;
	tx_md.has_time_spec = true;
	tx_md.time_spec = uhd::time_spec_t(0.1);

	rx_stream = usrp->get_rx_stream(stream_args);
	rx_spb = rx_stream->get_max_num_samps()*16;
	rx_buf.resize(rx_spb);
	rx_buf_ptrs[0] = &rx_buf[0];
}

void Radio::init_receive()
{
	uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
	stream_cmd.num_samps = 0;
	stream_cmd.stream_now = false;
	double settling_time = 1.0;
	rx_timeout = settling_time + 0.1;
	stream_cmd.time_spec = uhd::time_spec_t(rx_timeout);
	rx_stream->issue_stream_cmd(stream_cmd);
}

void Radio::set_dsp_rate(double rate)
{
	usrp->set_tx_rate(rate);
	usrp->set_rx_rate(rate);

	cout << "tx rate: " << usrp->get_tx_rate()/1e6 << " MHz" << endl;
	cout << "rx rate: " << usrp->get_rx_rate()/1e6 << " MHz" << endl;
}

void Radio::set_rf_frequency(double freq)
{
	rf_freq = freq;
	target_freq = rf_freq + dsp_freq;

	uhd::tune_request_t tune_request;
	tune_request.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
	tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_NONE;

	tune_request.target_freq = target_freq;
	tune_request.rf_freq = rf_freq;
	tune_request.dsp_freq = dsp_freq;

	usrp->set_tx_freq(tune_request,channel);
	usrp->set_rx_freq(tune_request,channel);

	cout << "TX freq: " << (usrp->get_tx_freq(channel)/1e6) << " MHz" << endl;
	cout << "RX freq: " << (usrp->get_rx_freq(channel)/1e6) << " MHz" << endl;
}

void Radio::set_dsp_frequency(double freq)
{
	dsp_freq = freq;
	target_freq = rf_freq + dsp_freq;

	uhd::tune_request_t tune_request;
	tune_request.rf_freq_policy = uhd::tune_request_t::POLICY_NONE;
	tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;

	tune_request.target_freq = target_freq;
	tune_request.dsp_freq = dsp_freq;

	usrp->set_tx_freq(tune_request,channel);

	tune_request.dsp_freq = -dsp_freq;
	usrp->set_rx_freq(tune_request,channel);

	cout << "TX freq: " << (usrp->get_tx_freq(channel)/1e6) << " MHz" << endl;
	cout << "RX freq: " << (usrp->get_rx_freq(channel)/1e6) << " MHz" << endl;
}

void Radio::zero_time()
{
	usrp->set_time_now(uhd::time_spec_t(0.0));
}

void Radio::check_lock()
{
    vector<string> tx_sensor_names, rx_sensor_names;
    tx_sensor_names = usrp->get_tx_sensor_names(0);
    if (find(tx_sensor_names.begin(), tx_sensor_names.end(), "lo_locked") != tx_sensor_names.end()) {
        uhd::sensor_value_t lo_locked = usrp->get_tx_sensor("lo_locked",0);
        cout << boost::format("Checking TX: %s ...") % lo_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
    rx_sensor_names = usrp->get_rx_sensor_names(0);
    if (find(rx_sensor_names.begin(), rx_sensor_names.end(), "lo_locked") != rx_sensor_names.end()) {
        uhd::sensor_value_t lo_locked = usrp->get_rx_sensor("lo_locked",0);
        cout << boost::format("Checking RX: %s ...") % lo_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
}

// send the contents of tx_buf
void Radio::transmit() {
	size_t num_sent = tx_stream->send(tx_buf_ptrs,tx_buf.size(),tx_md);
	if(num_sent != tx_buf.size()) {
		cerr << "Did not send full buffer!" << endl;
		exit(1);
	}
	tx_md.start_of_burst = false;
	tx_md.has_time_spec = false;
	num_transmit += tx_buf.size();
}

void Radio::transmit_stop()
{
	tx_md.end_of_burst = true;
	tx_stream->send("",0,tx_md);
}

void Radio::receive(queue<complex64> &out_queue) {
	size_t num_rx_samps = rx_stream->recv(rx_buf_ptrs,rx_buf.size(),rx_md,rx_timeout);

	rx_timeout = 0.1;
	if(rx_md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
		*stop_signal = true;
		cerr << "rx_stream->recv timeout while streaming" << endl;
	}
	if(rx_md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
		// This means we lost sample alignment and there should be a phase jump
		// This could be propagated forward as a signal to re-lock phase
		cerr << "o";
		num_rx_samps = 0;
		return;
	}
	if(rx_md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
		*stop_signal = true;
		throw runtime_error(str(boost::format("Receiver error %s")%rx_md.strerror()));
	}
	num_receive += num_rx_samps;

	for(size_t i=0;i<rx_buf.size();i++) {
		out_queue.push(rx_buf[i]);
	}
}

void Radio::receive_stop() {
	uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
	rx_stream->issue_stream_cmd(stream_cmd);
}

