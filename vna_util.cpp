#include "vna_util.h"

#include <assert.h>
#include <stdio.h>
#include <iostream>

void read_sc16_vec(FILE *f, vector<sc16> &out)
{
	size_t n;
	int nread;
	nread = fscanf(f,"%zu",&n);
	assert(nread == 1);
	out.resize(n);
	for(size_t i=0;i<n;i++) {
		short re,im;
		nread = fscanf(f,"%hd%hd",&re,&im);
		out[i] = sc16(re,im);
		assert(nread == 2);
	}
}

ChirpGenerator::ChirpGenerator(float ampl_, double wmin_, double wmax_, size_t samp_per_step_, size_t nstep_) :
	ampl(ampl_),wmin(wmin_),wmax(wmax_),samp_per_step(samp_per_step_),nstep(nstep_),ws(nstep),zs(nstep)
{
	for(size_t i=0;i<nstep;i++) {
		ws[i] = wmin + (wmax - wmin) * (i + 0.5) / nstep;
		zs[i] = exp(I*ws[i]);
	}
	step = 0;
	samp = 0;
	update();
}

void ChirpGenerator::update()
{
	w = ws[step];
	z = zs[step];
	za = 1.0;
}

void ChirpGenerator::fill_buf(vector<sc16> &buf)
{
	for(size_t i=0;i<buf.size();i++) {
		short re = roundf(za.real() * ampl * 32767.0f);
		short im = roundf(za.imag() * ampl * 32767.0f);
		buf[i] = sc16(re,im);
		za *= z;
		samp += 1;
		if(samp == samp_per_step) {
			samp = 0;
			step += 1;
			if(step == nstep) {
				step = 0;
			}
			update();
		}
	}
}

Demodulator::Demodulator(ChirpGenerator *chirpgen_, ofstream *out_)
	: chirpgen(chirpgen_),out(out_),nstep(chirpgen->nstep),samp_per_step(chirpgen->samp_per_step),step(0),
		j(0),sum(0,0),lo(nstep*samp_per_step),window(samp_per_step)
{
	chirpgen->fill_buf(lo);
	for(size_t i=0;i<nstep*samp_per_step;i++) {
		lo[i] = conj(lo[i]);
	}
	for(size_t i=0;i<samp_per_step;i++) {
		size_t clip = 64;
		if(i < clip) {
			window[i] = 0;
		} else if (i < samp_per_step - 64) {
			window[i] = 1;
		} else {
			window[i] = 0;
		}
	}
}

void Demodulator::input_data(vector<sc16> &buf,size_t n) {
	for(size_t i=0;i<n;i++) {
		sc16 b16 = buf[i];
		sc16 l16 = lo[step*samp_per_step + j];
		sc64 b64 = sc64(b16.real(),b16.imag());
		sc64 l64 = sc64(l16.real(),l16.imag());
		int64_t w64 = window[j];

		sum += b64*l64*w64;
		j += 1;
		if(j == samp_per_step) {
			sum /= samp_per_step;
			(*out) << sum.real() << " " << sum.imag() << " ";
			j = 0;
			sum = 0;
			step += 1;
			if(step == nstep) {
				step = 0;
				(*out) << endl;
				out->flush();
			}
		}
	}
}

/*
Split the writing of data to disk into a separate thread from receiving data from the USRP
Occasionally, fwrite will have an extremely long latency
We don't want this to block recv() calls, which don't have enough buffer to handle the long delay
*/
void writethreadfunc(boost::lockfree::queue<vector<sc16> *> *message_queue,
                        FILE *fout, int *stop_write)
{
	vector<sc16> *v;
	while(!(*stop_write) || !message_queue->empty()) {
		while(message_queue->pop(v)) {
			fwrite(&(*v)[0],sizeof(sc16),v->size(),fout);
			delete v;
		}
		// save a CPU?  boost::this_thread::yield();
	}
}


/*
 * Repeatedly send the contents of ring_buffer in chunks of spb
 * Keep track of when tx_streamer->send() does not send the entire buffer
 */
void transmit_worker(
    uhd::tx_streamer::sptr tx_streamer,
    int num_channels,
	const vector<sc16> ring_buffer,
	volatile bool *stop_signal_called,
	float settling,
	size_t nmax)
{
	double timeout = settling + 0.2;
	size_t nbuff = 8*tx_streamer->get_max_num_samps();
	vector<sc16> buff(nbuff);

	//setup the metadata flags
	uhd::tx_metadata_t md;
	md.start_of_burst = true;
	md.end_of_burst   = false;
	md.has_time_spec  = true;
	md.time_spec = uhd::time_spec_t(settling);

    std::vector<sc16 *> buffs(num_channels, &buff.front());
	size_t buff_end = 0;
	size_t ring_buffer_start = 0;
	size_t ncpy;
	size_t total_sent = 0;
    while(not (*stop_signal_called)) {
		while(buff_end < nbuff) {
			size_t supply = ring_buffer.size() - ring_buffer_start;
			size_t demand = nbuff - buff_end;
			ncpy = min(supply,demand);
			memcpy(&buff[buff_end],&ring_buffer[ring_buffer_start],ncpy*sizeof(sc16));
			buff_end += ncpy;
			ring_buffer_start += ncpy;
			if(ring_buffer_start == ring_buffer.size()) {
				ring_buffer_start = 0;
			}
		}
        size_t num_sent = tx_streamer->send(buffs, nbuff, md, timeout);
		buff_end = nbuff - num_sent;
		assert(buff_end >= 0);
		memmove(&buff[0],&buff[num_sent],buff_end*sizeof(sc16));

		md.start_of_burst = false;
        md.has_time_spec = false;
		timeout = 0.2;

		total_sent += num_sent;
		if(nmax>0 && (total_sent >= nmax)) {
			break;
		}
    }

    md.end_of_burst = true;
    tx_streamer->send("", 0, md);

    std::cout << std::endl << "Waiting for async burst ACK... " << std::flush;
    uhd::async_metadata_t async_md;
    bool got_async_burst_ack = false;
    //loop through all messages for the ACK packet (may have underflow messages in queue)
    while (not got_async_burst_ack) {
		tx_streamer->recv_async_msg(async_md, timeout);
        got_async_burst_ack = (async_md.event_code == uhd::async_metadata_t::EVENT_CODE_BURST_ACK);
    }
    std::cout << (got_async_burst_ack? "success" : "fail") << std::endl;

}

Multichannel::Multichannel() : nch(MC_NCH),lo_lookup(MC_NLO), lo_phase(MC_NCH), phase_inc(MC_NCH), cic(nch)
{
}

inline sc32 sc32_of_sc16(sc16 x) { return sc32(x.real(),x.imag()); }

void Multichannel::process_block(sc32* __restrict__ out, sc16* __restrict__ in, size_t n)
{
	sc32 zlo;
	vector<sc32> buf(n);
	for(size_t ch=0;ch<MC_NCH;ch++) {
		for(size_t i=0;i<n;i++) {
			zlo = lo_lookup[lo_phase[ch]];
			buf[i] = sc32(zlo) * sc32_of_sc16(in[i]);
			lo_phase[ch] += phase_inc[ch] & (MC_NLO-1);
		}
		size_t nout;
		nout = 1;
		nout = cic[ch].process_block(&out[ch],&buf[0],n);
		if(nout != 1) {
			fprintf(stderr,"Error, CIC did not produce one sample of output\n");
			exit(1);
		}
	}
}

CIC::CIC() : decimate(CIC_DECIMATE),order(CIC_ORDER),ibits(CIC_IBITS),obits(CIC_OBITS)
{
	numbits = order * ceil(log(decimate)/log(2.)) + ibits;
	outshift = numbits - obits;
	//outmask = (1 << obits) - 1;
	outmask = 0xFFFFFFFF;

	assert(numbits <= 64);
	memset(&sums[0],0,sizeof(sc64)*0);
	memset(&delays[0],0,sizeof(sc64)*0);
	memset(&diffs[0],0,sizeof(sc64)*0);

}

#define CIC_SHM(x) (((x) >> outshift)&outmask)
size_t CIC::process_block(sc32 * __restrict__ out, sc32 * __restrict__ in, size_t nin)
{
	if(nin % CIC_DECIMATE != 0) {
		cerr << "Error, CIC::process_block must get a multiple of CIC_DECIMATE samples at a time" << endl;
		exit(1);
	}
	size_t nout = nin / CIC_DECIMATE;
	for(size_t k=0;k<nout;k++) {
		for(size_t i=0;i<CIC_DECIMATE;i++) {
			sums[0] += in[k*CIC_DECIMATE+i];
			for(size_t j=1;j<CIC_ORDER;j++) {
				sums[j] += sums[j-1];
			}
		}
		sc64 z = sums[CIC_ORDER-1];

		diffs[0] = z - delays[0];
		delays[0] = z;
		for(size_t j=1;j<CIC_ORDER;j++) {
			diffs[j] = diffs[j-1] - delays[j];
			delays[j] = diffs[j-1];
		}
		sc64 t = diffs[CIC_ORDER-1];
		out[k] = sc32(CIC_SHM(t.real()),CIC_SHM(t.imag()));
	}
	return nout;
}

double get_wall_time()
{
	struct timeval time;
	if(gettimeofday(&time,NULL)) {
		fprintf(stderr,"gettimeofday() failed\n");
		exit(1);
	}
	return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

size_t min(size_t x, size_t y)
{
	if(x < y) {
		return x;
	} else {
		return y;
	}
}

void make_filename(string &out)
{
	char outstr[200];
	time_t t;
	struct tm *tmp;
	t = time(NULL);
	tmp = localtime(&t);
	strftime(outstr,sizeof(outstr),"vna_%Y%m%d_%H%M%S.dat",tmp);
	out = string(outstr);
}

