#pragma once

#include <stdint.h>
#include <vector>
#include <complex>
#include <memory>
#include <fstream>
#include <uhd/usrp/multi_usrp.hpp>
#include <boost/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <time.h>
#include <sys/time.h>

using namespace std;

typedef std::complex<float> fc32;
typedef std::complex<double> fc64;
typedef std::complex<short> sc16;
typedef std::complex<int32_t> sc32;
typedef std::complex<int64_t> sc64;
const static std::complex<double> I(0,1);

void read_sc16_vec(FILE *f, vector<sc16> &out);

void transmit_worker(
    uhd::tx_streamer::sptr tx_streamer,
    int num_channels,
	const vector<sc16> ring_buffer,
	volatile bool *stop_signal_called,
	float settling,
	size_t nmax);

class ChirpGenerator
{
	public:
	typedef std::shared_ptr<ChirpGenerator> sptr;
	double ampl;
	double wmin;
	double wmax;
	size_t samp_per_step;
	size_t nstep;
	size_t step,samp;
	double w;
	vector<double> ws;
	vector<fc64> zs;
	fc64 z;
	fc64 za;
	ChirpGenerator(float ampl_, double wmin_, double wmax_, size_t samp_per_step_, size_t nstep_);
	void update();
	void fill_buf(vector<sc16> &buf);
};

class Demodulator
{
	public:
	ChirpGenerator *chirpgen;
	ofstream *out;
	size_t nstep;
	size_t samp_per_step;
	size_t step;

	size_t j;
	sc64 sum;
	vector<sc16> lo;
	vector<short> window;

	Demodulator(ChirpGenerator *chirpgen_, ofstream *out_);
	void input_data(vector<sc16> &buf,size_t n);
};

#define CIC_DECIMATE 64
#define CIC_DECIMATE_MASK (CIC_DECIMATE - 1)
#define CIC_ORDER 5
#define CIC_IBITS 16
#define CIC_OBITS 32
class CIC
{
	public:
	size_t decimate;
	size_t order;
	uint64_t ibits;
	uint64_t obits;

	size_t numbits;
	uint64_t outshift;
	uint64_t outmask;

	sc64 sums[CIC_ORDER];
	sc64 delays[CIC_ORDER];
	sc64 diffs[CIC_ORDER];

	CIC();

	size_t process_block(sc32 * __restrict__ out, sc32 * __restrict__ in, size_t nin);
};


#define MC_NCH 8
#define MC_NLO 16384
class Multichannel
{
	public:
	size_t nch;
	vector<sc16> lo_lookup;
	vector<uint16_t> lo_phase;
	vector<uint16_t> phase_inc;
	vector<CIC> cic;

	Multichannel();
	void process_block(sc32* __restrict__ out, sc16* __restrict in, size_t n);
};

double get_wall_time();
size_t min(size_t x, size_t y);
void make_filename(string &out);
void writethreadfunc(boost::lockfree::queue<vector<sc16> *> *message_queue,
                        FILE *fout, int *stop_write);

