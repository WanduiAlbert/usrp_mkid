
#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread_priority.hpp>
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

#include "vna_util.h"


using namespace std;

namespace po = boost::program_options;

static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}

/***********************************************************************
 * recv_to_file function
 **********************************************************************/
void recv_to_file(
    uhd::usrp::multi_usrp::sptr usrp,
    size_t samps_per_buff,
    int num_requested_samples,
    std::vector<size_t> rx_channel_nums,
	uhd::rx_streamer::sptr rx_stream,
	float settling,
	Multichannel &mc,
	FILE *fout
){
    int num_total_samps = 0;

	FILE *latency_log;
	latency_log = fopen("latency.txt","w");

    // Prepare buffers for received samples and metadata
    uhd::rx_metadata_t md;
	vector<sc16> recv_buff(samps_per_buff);

    //create a vector of pointers to point to each of the channel buffers
    std::vector<sc16 *> buff_ptrs;
	buff_ptrs.push_back(&recv_buff[0]);

    float timeout = settling+0.2f; //expected settling time + padding for first recv

    //setup streaming
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = uhd::time_spec_t(settling);
    rx_stream->issue_stream_cmd(stream_cmd);
	cout << "Issued stream cmd" << endl;

	size_t step = CIC_DECIMATE;
	sc16 *recv_walk;
	vector<sc16> block(step);
#define GETTIME() (get_wall_time()-start)
	double start = get_wall_time();
	int overflow;
	size_t noutmax=256;
	size_t nout;
	vector<sc32> out(noutmax*mc.nch);
    while(not stop_signal_called and (num_requested_samples != num_total_samps or num_requested_samples == 0)){
		overflow = 0;
		//double t0 = GETTIME();
        size_t num_rx_samps = rx_stream->recv(buff_ptrs, samps_per_buff, md, timeout, true);
		//double t1 = GETTIME();
		timeout = 0.2;

        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            std::cout << boost::format("Timeout while streaming") << std::endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW){
			overflow = 1;
        }
		else if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE){
            throw std::runtime_error(str(boost::format(
                "Receiver error %s"
            ) % md.strerror()));
        }

        num_total_samps += num_rx_samps;

		size_t nb=0;
		size_t numleft = num_rx_samps;
		recv_walk = buff_ptrs[0];
		nout = 0;
		while(numleft > 0) {
			size_t ncpy = min(step - nb, numleft);
			memcpy(&block[nb],recv_walk,ncpy);
			nb += ncpy;
			numleft -= ncpy;
			recv_walk = &recv_walk[ncpy];

			if(nb == step) {
				mc.process_block(&out[nout*mc.nch],&block[0],step);
				nout += 1;
				if(nout >= noutmax) {
					cerr << "Error, too many samples in the block" << endl;
					break;
				}
				nb = 0;
			}
		}

		//double t2 = GETTIME();
		fwrite(&out[0],nout*mc.nch,sizeof(sc32),fout);
		//double t3 = GETTIME();
		//fprintf(latency_log,"%f %f %f %f %d %zu\n",t0,t1,t2,t3,overflow,num_rx_samps);

    }

    // Shut down receiver
    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);

}


/***********************************************************************
 * Main function
 **********************************************************************/
int UHD_SAFE_MAIN(int argc, char *argv[]){
    uhd::set_thread_priority_safe();

    //transmit variables to be set by po
    std::string tx_args, tx_ant, tx_subdev, ref, otw, tx_channels;
    double rate, tx_freq, tx_gain, tx_bw;
    float ampl;

    //receive variables to be set by po
    std::string rx_args, phase_table_fn, infn, outfn, rx_ant, rx_subdev, rx_channels;
    size_t spb;
    double rx_freq, rx_gain, rx_bw, lo_off;
    float settling;

    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("tx-args", po::value<std::string>(&tx_args)->default_value(""), "uhd transmit device address args")
        ("rx-args", po::value<std::string>(&rx_args)->default_value(""), "uhd receive device address args")
        ("phasetablefn", po::value<std::string>(&phase_table_fn)->default_value("phase_table.txt"), "name of the file to read phase increments from")
        ("infn", po::value<std::string>(&infn)->default_value("rf_in.dat"), "name of the file to read binary samples from")
        ("outfn", po::value<std::string>(&outfn)->default_value("rf_out.dat"), "name of the file to write binary samples to")
        ("settling", po::value<float>(&settling)->default_value(float(1.0)), "settling time (seconds) before receiving")
        ("rate", po::value<double>(&rate), "sample rate")
        ("tx-freq", po::value<double>(&tx_freq), "transmit RF center frequency in Hz")
        ("rx-freq", po::value<double>(&rx_freq), "receive RF center frequency in Hz")
		("lo-off", po::value<double>(&lo_off)->default_value(10e6), "LO offset frequency")
        ("ampl", po::value<float>(&ampl)->default_value(float(0.3)), "amplitude of the waveform [0 to 0.7]")
        ("tx-gain", po::value<double>(&tx_gain), "gain for the transmit RF chain")
        ("rx-gain", po::value<double>(&rx_gain), "gain for the receive RF chain")
        ("tx-ant", po::value<std::string>(&tx_ant), "transmit antenna selection")
        ("rx-ant", po::value<std::string>(&rx_ant), "receive antenna selection")
        ("tx-subdev", po::value<std::string>(&tx_subdev), "transmit subdevice specification")
        ("rx-subdev", po::value<std::string>(&rx_subdev), "receive subdevice specification")
        ("tx-bw", po::value<double>(&tx_bw), "analog transmit filter bandwidth in Hz")
        ("rx-bw", po::value<double>(&rx_bw), "analog receive filter bandwidth in Hz")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "clock reference (internal, external, mimo)")
        ("otw", po::value<std::string>(&otw)->default_value("sc16"), "specify the over-the-wire sample mode")
        ("tx-channels", po::value<std::string>(&tx_channels)->default_value("0"), "which TX channel(s) to use (specify \"0\", \"1\", \"0,1\", etc)")
        ("rx-channels", po::value<std::string>(&rx_channels)->default_value("0"), "which RX channel(s) to use (specify \"0\", \"1\", \"0,1\", etc)")
        ("tx-int-n", "tune USRP TX with integer-N tuning")
        ("rx-int-n", "tune USRP RX with integer-N tuning")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")){
        std::cout << boost::format("UHD TXRX Loopback to File %s") % desc << std::endl;
        return ~0;
    }

    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the transmit usrp device with: %s...") % tx_args << std::endl;
    uhd::usrp::multi_usrp::sptr tx_usrp = uhd::usrp::multi_usrp::make(tx_args);
    std::cout << std::endl;
    std::cout << boost::format("Creating the receive usrp device with: %s...") % rx_args << std::endl;
    uhd::usrp::multi_usrp::sptr rx_usrp = uhd::usrp::multi_usrp::make(rx_args);

    //detect which channels to use
    std::vector<std::string> tx_channel_strings;
    std::vector<size_t> tx_channel_nums;
    boost::split(tx_channel_strings, tx_channels, boost::is_any_of("\"',"));
    for(size_t ch = 0; ch < tx_channel_strings.size(); ch++){
        size_t chan = boost::lexical_cast<int>(tx_channel_strings[ch]);
        if(chan >= tx_usrp->get_tx_num_channels()){
            throw std::runtime_error("Invalid TX channel(s) specified.");
        }
        else tx_channel_nums.push_back(boost::lexical_cast<int>(tx_channel_strings[ch]));
    }
    std::vector<std::string> rx_channel_strings;
    std::vector<size_t> rx_channel_nums;
    boost::split(rx_channel_strings, rx_channels, boost::is_any_of("\"',"));
    for(size_t ch = 0; ch < rx_channel_strings.size(); ch++){
        size_t chan = boost::lexical_cast<int>(rx_channel_strings[ch]);
        if(chan >= rx_usrp->get_rx_num_channels()){
            throw std::runtime_error("Invalid RX channel(s) specified.");
        }
        else rx_channel_nums.push_back(boost::lexical_cast<int>(rx_channel_strings[ch]));
    }

    //Lock mboard clocks
    tx_usrp->set_clock_source(ref);
    rx_usrp->set_clock_source(ref);

    //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("tx-subdev")) tx_usrp->set_tx_subdev_spec(tx_subdev);
    if (vm.count("rx-subdev")) rx_usrp->set_rx_subdev_spec(rx_subdev);

    std::cout << boost::format("Using TX Device: %s") % tx_usrp->get_pp_string() << std::endl;
    std::cout << boost::format("Using RX Device: %s") % rx_usrp->get_pp_string() << std::endl;

    if (not vm.count("rate")){
        std::cerr << "Please specify the sample rate with --rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting TX Rate: %f Msps...") % (rate/1e6) << std::endl;
    tx_usrp->set_tx_rate(rate);
    std::cout << boost::format("Actual TX Rate: %f Msps...") % (tx_usrp->get_tx_rate()/1e6) << std::endl << std::endl;

    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    rx_usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (rx_usrp->get_rx_rate()/1e6) << std::endl << std::endl;

    //set the transmit center frequency
    if (not vm.count("tx-freq")){
        std::cerr << "Please specify the transmit center frequency with --tx-freq" << std::endl;
        return ~0;
    }

    for(size_t ch = 0; ch < tx_channel_nums.size(); ch++) {
        size_t channel = tx_channel_nums[ch];
        if (tx_channel_nums.size() > 1) {
            std::cout << "Configuring TX Channel " << channel << std::endl;
        }
        std::cout << boost::format("Setting TX Freq: %f MHz...") % (tx_freq/1e6) << std::endl;
        uhd::tune_request_t tx_tune_request(tx_freq);
		tx_tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
		tx_tune_request.dsp_freq = -lo_off;
        if(vm.count("tx-int-n")) tx_tune_request.args = uhd::device_addr_t("mode_n=integer");
        tx_usrp->set_tx_freq(tx_tune_request, channel);
        std::cout << boost::format("Actual TX Freq: %f MHz...") % (tx_usrp->get_tx_freq(channel)/1e6) << std::endl << std::endl;

        //set the rf gain
        if (vm.count("tx-gain")){
            std::cout << boost::format("Setting TX Gain: %f dB...") % tx_gain << std::endl;
            tx_usrp->set_tx_gain(tx_gain, channel);
            std::cout << boost::format("Actual TX Gain: %f dB...") % tx_usrp->get_tx_gain(channel) << std::endl << std::endl;
        }

        //set the analog frontend filter bandwidth
        if (vm.count("tx-bw")){
            std::cout << boost::format("Setting TX Bandwidth: %f MHz...") % tx_bw << std::endl;
            tx_usrp->set_tx_bandwidth(tx_bw, channel);
            std::cout << boost::format("Actual TX Bandwidth: %f MHz...") % tx_usrp->get_tx_bandwidth(channel) << std::endl << std::endl;
        }

        //set the antenna
        if (vm.count("tx-ant")) tx_usrp->set_tx_antenna(tx_ant, channel);
    }

    for(size_t ch = 0; ch < rx_channel_nums.size(); ch++) {
        size_t channel = rx_channel_nums[ch];
        if (rx_channel_nums.size() > 1) {
            std::cout << "Configuring RX Channel " << channel << std::endl;
        }

        //set the receive center frequency
        if (not vm.count("rx-freq")){
            std::cerr << "Please specify the center frequency with --rx-freq" << std::endl;
            return ~0;
        }
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (rx_freq/1e6) << std::endl;
        uhd::tune_request_t rx_tune_request(rx_freq);
		rx_tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
		rx_tune_request.dsp_freq = lo_off;
        if(vm.count("rx-int-n")) rx_tune_request.args = uhd::device_addr_t("mode_n=integer");
        rx_usrp->set_rx_freq(rx_tune_request, channel);
        std::cout << boost::format("Actual RX Freq: %f MHz...") % (rx_usrp->get_rx_freq(channel)/1e6) << std::endl << std::endl;

        //set the receive rf gain
        if (vm.count("rx-gain")){
            std::cout << boost::format("Setting RX Gain: %f dB...") % rx_gain << std::endl;
            rx_usrp->set_rx_gain(rx_gain, channel);
            std::cout << boost::format("Actual RX Gain: %f dB...") % rx_usrp->get_rx_gain(channel) << std::endl << std::endl;
        }

        //set the receive analog frontend filter bandwidth
        if (vm.count("rx-bw")){
            std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % (rx_bw/1e6) << std::endl;
            rx_usrp->set_rx_bandwidth(rx_bw, channel);
            std::cout << boost::format("Actual RX Bandwidth: %f MHz...") % (rx_usrp->get_rx_bandwidth(channel)/1e6) << std::endl << std::endl;
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
    spb = tx_stream->get_max_num_samps()*10;
    std::vector<std::complex<float> > buff(spb);
    int num_channels = tx_channel_nums.size();

    //setup the metadata flags
    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst   = false;
    md.has_time_spec  = true;
    md.time_spec = uhd::time_spec_t(settling);

    //Check Ref and LO Lock detect
    std::vector<std::string> tx_sensor_names, rx_sensor_names;
    tx_sensor_names = tx_usrp->get_tx_sensor_names(0);
    if (std::find(tx_sensor_names.begin(), tx_sensor_names.end(), "lo_locked") != tx_sensor_names.end()) {
        uhd::sensor_value_t lo_locked = tx_usrp->get_tx_sensor("lo_locked",0);
        std::cout << boost::format("Checking TX: %s ...") % lo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
    rx_sensor_names = rx_usrp->get_rx_sensor_names(0);
    if (std::find(rx_sensor_names.begin(), rx_sensor_names.end(), "lo_locked") != rx_sensor_names.end()) {
        uhd::sensor_value_t lo_locked = rx_usrp->get_rx_sensor("lo_locked",0);
        std::cout << boost::format("Checking RX: %s ...") % lo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }

    tx_sensor_names = tx_usrp->get_mboard_sensor_names(0);
    if ((ref == "mimo") and (std::find(tx_sensor_names.begin(), tx_sensor_names.end(), "mimo_locked") != tx_sensor_names.end())) {
        uhd::sensor_value_t mimo_locked = tx_usrp->get_mboard_sensor("mimo_locked",0);
        std::cout << boost::format("Checking TX: %s ...") % mimo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(mimo_locked.to_bool());
    }
    if ((ref == "external") and (std::find(tx_sensor_names.begin(), tx_sensor_names.end(), "ref_locked") != tx_sensor_names.end())) {
        uhd::sensor_value_t ref_locked = tx_usrp->get_mboard_sensor("ref_locked",0);
        std::cout << boost::format("Checking TX: %s ...") % ref_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(ref_locked.to_bool());
    }

    rx_sensor_names = rx_usrp->get_mboard_sensor_names(0);
    if ((ref == "mimo") and (std::find(rx_sensor_names.begin(), rx_sensor_names.end(), "mimo_locked") != rx_sensor_names.end())) {
        uhd::sensor_value_t mimo_locked = rx_usrp->get_mboard_sensor("mimo_locked",0);
        std::cout << boost::format("Checking RX: %s ...") % mimo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(mimo_locked.to_bool());
    }
    if ((ref == "external") and (std::find(rx_sensor_names.begin(), rx_sensor_names.end(), "ref_locked") != rx_sensor_names.end())) {
        uhd::sensor_value_t ref_locked = rx_usrp->get_mboard_sensor("ref_locked",0);
        std::cout << boost::format("Checking RX: %s ...") % ref_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(ref_locked.to_bool());
    }

    uhd::stream_args_t rx_stream_args("sc16",otw);
    rx_stream_args.channels = rx_channel_nums;
    uhd::rx_streamer::sptr rx_stream = rx_usrp->get_rx_stream(rx_stream_args);
	cout << "Created rx stream" << endl;

	FILE *input_file = fopen(infn.c_str(),"rb");
	assert(input_file != NULL);
	vector<sc16> transmit_signal;
	read_sc16_vec(input_file,transmit_signal);

	cout << "Waiting for output file to open..." << endl;
	FILE *output_file = fopen(outfn.c_str(),"wb");
	cout << "...ready" << endl;

	std::signal(SIGINT, &sig_int_handler);
	std::cout << "Press Ctrl + C to stop streaming..." << std::endl;

    //reset usrp time to prepare for transmit/receive
    std::cout << boost::format("Setting device timestamp to 0...") << std::endl;
    tx_usrp->set_time_now(uhd::time_spec_t(0.0));

	boost::thread transmit_thread(boost::bind(&transmit_worker, tx_stream, md, num_channels, transmit_signal, &stop_signal_called));

	double loop_period = transmit_signal.size() / rate;
	cout << "loop period: " << loop_period << endl;
	size_t num_request = 0;
	Multichannel mc;

	ifstream phase_table_f(phase_table_fn);
	for(size_t i=0;i<MC_NCH;i++) {
		phase_table_f >> mc.phase_inc[i];
	}
	
	boost::thread recv_thread(boost::bind(&recv_to_file, rx_usrp, spb, num_request, rx_channel_nums, rx_stream, settling, mc, output_file));
	//recv_to_file(rx_usrp, spb, num_request, rx_channel_nums, rx_stream, settling, mc, output_file);
	
	pthread_t hnd1 = transmit_thread.native_handle();
	pthread_t hnd2 = recv_thread.native_handle();
	cpu_set_t cpuset1;
	cpu_set_t cpuset2;

	CPU_ZERO(&cpuset1);
	CPU_ZERO(&cpuset2);

	CPU_SET(0,&cpuset1);
	CPU_SET(1,&cpuset2);
	pthread_setaffinity_np(hnd1,sizeof(cpu_set_t),&cpuset1);
	pthread_setaffinity_np(hnd2,sizeof(cpu_set_t),&cpuset2);

	recv_thread.join();
    //clean up transmit worker
    stop_signal_called = true;
    transmit_thread.join();

    //finished
    std::cout << std::endl << "Done!" << std::endl << std::endl;
    return EXIT_SUCCESS;
}

