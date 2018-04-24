
#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <csignal>
#include <complex>
#include <time.h>

#include "vna_util.h"


using namespace std;

namespace po = boost::program_options;

static bool stop_signal_called = false;
static bool end_transmit = false;
void sig_int_handler(int){stop_signal_called = true;}

/***********************************************************************
 * recv_to_file function
 **********************************************************************/
void recv_to_file(
    uhd::usrp::multi_usrp::sptr usrp,
    size_t samps_per_buff,
    int num_requested_samples,
    vector<size_t> rx_channel_nums,
	uhd::rx_streamer::sptr rx_stream,
	float settling,
	Demodulator &demod
){
    int num_total_samps = 0;

    vector <vector< sc16 > > buffs(
        rx_channel_nums.size(), vector< sc16 >(samps_per_buff)
    );

    vector<sc16 *> buff_ptrs;
    for (size_t i = 0; i < buffs.size(); i++) {
        buff_ptrs.push_back(&buffs[i].front());
    }

    UHD_ASSERT_THROW(buffs.size() == rx_channel_nums.size());
    float timeout = settling+0.2f;

    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps = num_requested_samples;
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = uhd::time_spec_t(settling);
    rx_stream->issue_stream_cmd(stream_cmd);
	cout << "Issued stream cmd" << endl;
	size_t request_size;

    uhd::rx_metadata_t md;

    while(num_requested_samples > num_total_samps){
		request_size = min(samps_per_buff,num_requested_samples-num_total_samps);
        size_t num_rx_samps = rx_stream->recv(buff_ptrs, request_size, md, timeout);
		timeout = 0.2;

        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            cout << boost::format("Timeout while streaming") << endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW){
            continue;
        }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE){
            throw runtime_error(str(boost::format(
                "Receiver error %s"
            ) % md.strerror()));
        }

        num_total_samps += num_rx_samps;
		demod.input_data(buffs[0],num_rx_samps);
    }

    //stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    //rx_stream->issue_stream_cmd(stream_cmd);

}


/***********************************************************************
 * Main function
 **********************************************************************/
int UHD_SAFE_MAIN(int argc, char *argv[]){
    uhd::set_thread_priority_safe();

    //transmit variables to be set by po
    string tx_args, tx_ant, tx_subdev, ref, otw, tx_channels;
    double rate, freq, tx_gain, tx_bw;
	vector<double> tx_gain_vec;
    float ampl;

    //receive variables to be set by po
    string rx_args, rx_ant, rx_subdev, rx_channels;
    size_t spb;
    double rx_gain, rx_bw, lo_off;
    float settling;
	size_t samp_per_step, nstep;
	int sweep_dir;

    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("tx-args", po::value<string>(&tx_args)->default_value(""), "uhd transmit device address args")
        ("rx-args", po::value<string>(&rx_args)->default_value(""), "uhd receive device address args")
        ("samp_per_step", po::value<size_t>(&samp_per_step)->default_value(4096), "samples per frequency step")
        ("nstep", po::value<size_t>(&nstep)->default_value(401), "number of frequency steps")
        ("settling", po::value<float>(&settling)->default_value(float(1.0)), "settling time (seconds) before receiving")
		("sweep-dir", po::value<int>(&sweep_dir)->default_value(-1),"Sweep direction (1,-1)")
        ("spb", po::value<size_t>(&spb)->default_value(0), "samples per buffer, 0 for default")
        ("rate", po::value<double>(&rate), "rate of samples")
        ("freq", po::value<double>(&freq), "RF center frequency in Hz")
		("lo-off", po::value<double>(&lo_off)->default_value(10e6), "LO offset frequency")
        ("ampl", po::value<float>(&ampl)->default_value(float(0.3)), "amplitude of the waveform [0 to 0.7]")
        ("tx-gain", po::value<vector<double> >(&tx_gain_vec)->multitoken(), "gain for the transmit RF chain")
        ("rx-gain", po::value<double>(&rx_gain), "gain for the receive RF chain")
        ("tx-ant", po::value<string>(&tx_ant), "transmit antenna selection")
        ("rx-ant", po::value<string>(&rx_ant), "receive antenna selection")
        ("tx-subdev", po::value<string>(&tx_subdev), "transmit subdevice specification")
        ("rx-subdev", po::value<string>(&rx_subdev), "receive subdevice specification")
        ("tx-bw", po::value<double>(&tx_bw), "analog transmit filter bandwidth in Hz")
        ("rx-bw", po::value<double>(&rx_bw), "analog receive filter bandwidth in Hz")
        ("ref", po::value<string>(&ref)->default_value("internal"), "clock reference (internal, external, mimo)")
        ("otw", po::value<string>(&otw)->default_value("sc16"), "specify the over-the-wire sample mode")
        ("tx-channels", po::value<string>(&tx_channels)->default_value("0"), "which TX channel(s) to use (specify \"0\", \"1\", \"0,1\", etc)")
        ("rx-channels", po::value<string>(&rx_channels)->default_value("0"), "which RX channel(s) to use (specify \"0\", \"1\", \"0,1\", etc)")
        ("tx-int-n", "tune USRP TX with integer-N tuning")
        ("rx-int-n", "tune USRP RX with integer-N tuning")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")){
        cout << boost::format("UHD TXRX Loopback to File %s") % desc << endl;
        return ~0;
    }

	string file;
	make_filename(file);

    //create a usrp device
    cout << endl;
    cout << boost::format("Creating the transmit usrp device with: %s...") % tx_args << endl;
    uhd::usrp::multi_usrp::sptr tx_usrp = uhd::usrp::multi_usrp::make(tx_args);
    cout << endl;
    cout << boost::format("Creating the receive usrp device with: %s...") % rx_args << endl;
    uhd::usrp::multi_usrp::sptr rx_usrp = uhd::usrp::multi_usrp::make(rx_args);

    //detect which channels to use
    vector<string> tx_channel_strings;
    vector<size_t> tx_channel_nums;
    boost::split(tx_channel_strings, tx_channels, boost::is_any_of("\"',"));
    for(size_t ch = 0; ch < tx_channel_strings.size(); ch++){
        size_t chan = boost::lexical_cast<int>(tx_channel_strings[ch]);
        if(chan >= tx_usrp->get_tx_num_channels()){
            throw runtime_error("Invalid TX channel(s) specified.");
        }
        else tx_channel_nums.push_back(boost::lexical_cast<int>(tx_channel_strings[ch]));
    }
    vector<string> rx_channel_strings;
    vector<size_t> rx_channel_nums;
    boost::split(rx_channel_strings, rx_channels, boost::is_any_of("\"',"));
    for(size_t ch = 0; ch < rx_channel_strings.size(); ch++){
        size_t chan = boost::lexical_cast<int>(rx_channel_strings[ch]);
        if(chan >= rx_usrp->get_rx_num_channels()){
            throw runtime_error("Invalid RX channel(s) specified.");
        }
        else rx_channel_nums.push_back(boost::lexical_cast<int>(rx_channel_strings[ch]));
    }

    //Lock mboard clocks
    tx_usrp->set_clock_source(ref);
    rx_usrp->set_clock_source(ref);

    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("tx-subdev")) tx_usrp->set_tx_subdev_spec(tx_subdev);
    if (vm.count("rx-subdev")) rx_usrp->set_rx_subdev_spec(rx_subdev);

    cout << boost::format("Using TX Device: %s") % tx_usrp->get_pp_string() << endl;
    cout << boost::format("Using RX Device: %s") % rx_usrp->get_pp_string() << endl;

    //set the transmit sample rate
    if (not vm.count("rate")){
        cerr << "Please specify the sample rate with --rate" << endl;
        return ~0;
    }
    cout << boost::format("Setting TX Rate: %f Msps...") % (rate/1e6) << endl;
    tx_usrp->set_tx_rate(rate);
    cout << boost::format("Actual TX Rate: %f Msps...") % (tx_usrp->get_tx_rate()/1e6) << endl << endl;

    cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << endl;
    rx_usrp->set_rx_rate(rate);
    cout << boost::format("Actual RX Rate: %f Msps...") % (rx_usrp->get_rx_rate()/1e6) << endl << endl;

    //set the transmit center frequency
    if (not vm.count("freq")){
        cerr << "Please specify the transmit center frequency with --freq" << endl;
        return ~0;
    }

    for(size_t ch = 0; ch < tx_channel_nums.size(); ch++) {
        size_t channel = tx_channel_nums[ch];
        if (tx_channel_nums.size() > 1) {
            cout << "Configuring TX Channel " << channel << endl;
        }
        cout << boost::format("Setting TX Freq: %f MHz...") % ((freq)/1e6) << endl;
        uhd::tune_request_t tx_tune_request(freq);
		tx_tune_request.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
		tx_tune_request.rf_freq = freq;
		tx_tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
		tx_tune_request.dsp_freq = lo_off;
        if(vm.count("tx-int-n")) tx_tune_request.args = uhd::device_addr_t("mode_n=integer");
        tx_usrp->set_tx_freq(tx_tune_request, channel);
        cout << boost::format("Actual TX Freq: %f MHz...") % (tx_usrp->get_tx_freq(channel)/1e6) << endl << endl;

		tx_gain = tx_gain_vec[0];
        //set the rf gain
        if (vm.count("tx-gain")){
            cout << boost::format("Setting TX Gain: %f dB...") % tx_gain << endl;
            tx_usrp->set_tx_gain(tx_gain, channel);
            cout << boost::format("Actual TX Gain: %f dB...") % tx_usrp->get_tx_gain(channel) << endl << endl;
        }

        //set the analog frontend filter bandwidth
        if (vm.count("tx-bw")){
            cout << boost::format("Setting TX Bandwidth: %f MHz...") % tx_bw << endl;
            tx_usrp->set_tx_bandwidth(tx_bw, channel);
            cout << boost::format("Actual TX Bandwidth: %f MHz...") % tx_usrp->get_tx_bandwidth(channel) << endl << endl;
        }

        //set the antenna
        if (vm.count("tx-ant")) tx_usrp->set_tx_antenna(tx_ant, channel);
    }

    for(size_t ch = 0; ch < rx_channel_nums.size(); ch++) {
        size_t channel = rx_channel_nums[ch];
        if (rx_channel_nums.size() > 1) {
            cout << "Configuring RX Channel " << channel << endl;
        }

        cout << boost::format("Setting RX Freq: %f MHz...") % (freq/1e6) << endl;
        uhd::tune_request_t rx_tune_request(freq);
		rx_tune_request.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
		rx_tune_request.rf_freq = freq;
		rx_tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
		rx_tune_request.dsp_freq = -lo_off;
        if(vm.count("rx-int-n")) rx_tune_request.args = uhd::device_addr_t("mode_n=integer");
        rx_usrp->set_rx_freq(rx_tune_request, channel);
        cout << boost::format("Actual RX Freq: %f MHz...") % (rx_usrp->get_rx_freq(channel)/1e6) << endl << endl;

        //set the receive rf gain
        if (vm.count("rx-gain")){
            cout << boost::format("Setting RX Gain: %f dB...") % rx_gain << endl;
            rx_usrp->set_rx_gain(rx_gain, channel);
            cout << boost::format("Actual RX Gain: %f dB...") % rx_usrp->get_rx_gain(channel) << endl << endl;
        }

        //set the receive analog frontend filter bandwidth
        if (vm.count("rx-bw")){
            cout << boost::format("Setting RX Bandwidth: %f MHz...") % (rx_bw/1e6) << endl;
            rx_usrp->set_rx_bandwidth(rx_bw, channel);
            cout << boost::format("Actual RX Bandwidth: %f MHz...") % (rx_usrp->get_rx_bandwidth(channel)/1e6) << endl << endl;
        }
    }
    //set the receive antenna
    if (vm.count("ant")) rx_usrp->set_rx_antenna(rx_ant);

    //create a transmit streamer
    //linearly map channels (index0 = channel0, index1 = channel1, ...)
    uhd::stream_args_t stream_args("sc16", otw);
    stream_args.channels = tx_channel_nums;
    uhd::tx_streamer::sptr tx_stream = tx_usrp->get_tx_stream(stream_args);

    //allocate a buffer which we re-use for each channel
    if (spb == 0) spb = tx_stream->get_max_num_samps()*16;
    vector<complex<float> > buff(spb);
    int num_channels = tx_channel_nums.size();


    //Check Ref and LO Lock detect
    vector<string> tx_sensor_names, rx_sensor_names;
    tx_sensor_names = tx_usrp->get_tx_sensor_names(0);
    if (find(tx_sensor_names.begin(), tx_sensor_names.end(), "lo_locked") != tx_sensor_names.end()) {
        uhd::sensor_value_t lo_locked = tx_usrp->get_tx_sensor("lo_locked",0);
        cout << boost::format("Checking TX: %s ...") % lo_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
    rx_sensor_names = rx_usrp->get_rx_sensor_names(0);
    if (find(rx_sensor_names.begin(), rx_sensor_names.end(), "lo_locked") != rx_sensor_names.end()) {
        uhd::sensor_value_t lo_locked = rx_usrp->get_rx_sensor("lo_locked",0);
        cout << boost::format("Checking RX: %s ...") % lo_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }

    tx_sensor_names = tx_usrp->get_mboard_sensor_names(0);
    if ((ref == "mimo") and (find(tx_sensor_names.begin(), tx_sensor_names.end(), "mimo_locked") != tx_sensor_names.end())) {
        uhd::sensor_value_t mimo_locked = tx_usrp->get_mboard_sensor("mimo_locked",0);
        cout << boost::format("Checking TX: %s ...") % mimo_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(mimo_locked.to_bool());
    }
    if ((ref == "external") and (find(tx_sensor_names.begin(), tx_sensor_names.end(), "ref_locked") != tx_sensor_names.end())) {
        uhd::sensor_value_t ref_locked = tx_usrp->get_mboard_sensor("ref_locked",0);
        cout << boost::format("Checking TX: %s ...") % ref_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(ref_locked.to_bool());
    }

    rx_sensor_names = rx_usrp->get_mboard_sensor_names(0);
    if ((ref == "mimo") and (find(rx_sensor_names.begin(), rx_sensor_names.end(), "mimo_locked") != rx_sensor_names.end())) {
        uhd::sensor_value_t mimo_locked = rx_usrp->get_mboard_sensor("mimo_locked",0);
        cout << boost::format("Checking RX: %s ...") % mimo_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(mimo_locked.to_bool());
    }
    if ((ref == "external") and (find(rx_sensor_names.begin(), rx_sensor_names.end(), "ref_locked") != rx_sensor_names.end())) {
        uhd::sensor_value_t ref_locked = rx_usrp->get_mboard_sensor("ref_locked",0);
        cout << boost::format("Checking RX: %s ...") % ref_locked.to_pp_string() << endl;
        UHD_ASSERT_THROW(ref_locked.to_bool());
    }

    //create a receive streamer
    uhd::stream_args_t rx_stream_args("sc16",otw);
    rx_stream_args.channels = rx_channel_nums;
    uhd::rx_streamer::sptr rx_stream = rx_usrp->get_rx_stream(rx_stream_args);
	cout << "Created rx stream" << endl;

	double wmin = -M_PI*0.75;
	double wmax = M_PI*0.75;
	if(sweep_dir == 1) {
		cout << "Low to high frequency sweep" << endl;
	} else if (sweep_dir == -1) {
		cout << "High to low frequency sweep" << endl;
		wmin = -wmin;
		wmax = -wmax;
	} else {
		cout << "Unknown sweep direction" << endl;
		exit(1);
	}

	vector<sc16> lo(samp_per_step*nstep);

	ChirpGenerator chirpgen_tx(ampl,wmin,wmax,samp_per_step,nstep);
	chirpgen_tx.fill_buf(lo);

	signal(SIGINT, &sig_int_handler);
	cout << "Press Ctrl + C to stop streaming..." << endl;

	cout << "Waiting for listener to open pipe..." << endl;
	ofstream demod_out(file,ofstream::binary);
	cout << "got it" << endl;
	Demodulator demod(&chirpgen_tx,&demod_out);

	demod_out << setprecision(9);
	demod_out << "samp_per_step: " << samp_per_step << endl;
	demod_out << "rate: " << rate << endl;
	demod_out << "rx_gain: " << rx_gain << endl;

	demod_out << "tx_gains:";
	for(size_t i=0;i<tx_gain_vec.size();i++) {
		demod_out << " " << tx_gain_vec[i];
	}
	demod_out << endl;

	demod_out << "frequencies:";
	for(size_t i=0;i<nstep;i++) {
		double f = freq + lo_off + rate * chirpgen_tx.ws[i]/(2.*M_PI);
		demod_out << " " << f;
	}
	demod_out << endl;

	cout << file << endl;
	for(size_t i=0;i<tx_gain_vec.size();i++) {
		if(stop_signal_called) {
			break;
		}

		cout << "Acquisition " << i << endl;
		tx_gain = tx_gain_vec[i];
		for(size_t ch = 0; ch < tx_channel_nums.size(); ch++) {
			size_t channel = tx_channel_nums[ch];
			if (vm.count("tx-gain")){
				cout << boost::format("Setting TX Gain: %f dB...") % tx_gain << endl;
				tx_usrp->set_tx_gain(tx_gain, channel);
				cout << boost::format("Actual TX Gain: %f dB...") % tx_usrp->get_tx_gain(channel) << endl << endl;
			}
		}

		size_t ndemod = 1;
		size_t num_request = ndemod*samp_per_step*nstep;
		double loop_period = samp_per_step * nstep / rate;

		//reset usrp time to prepare for transmit/receive
		cout << boost::format("Setting device timestamp to 0...") << endl;
		tx_usrp->set_time_now(uhd::time_spec_t(0.0));

		//start transmit worker thread
		boost::thread_group transmit_thread;
		transmit_thread.create_thread(boost::bind(&transmit_worker, tx_stream, num_channels, lo, &end_transmit, settling, num_request));
		//transmit_worker(tx_stream,num_channels,lo,&end_transmit,settling);

		cout << "loop period: " << loop_period << endl;
		cout << "recv_to_file: " << num_request << endl;
		recv_to_file(rx_usrp, spb, num_request, rx_channel_nums, rx_stream, settling, demod);

		//clean up transmit worker
		//end_transmit = true;
		transmit_thread.join_all();
		//end_transmit = false;
	}

    //finished
    cout << endl << "Done!" << endl << endl;
	cout << file << endl;
    return EXIT_SUCCESS;
}

