
#include <string.h>

#include "channelizer.h"
#include <queue>
using namespace std;

Channelizer::Channelizer(FIRDecimate::sptr _firdecimate)
: firdecimate(_firdecimate),ntap(firdecimate->ntap),decimate(firdecimate->decimate),in_block(decimate)
{
}

void Channelizer::set_lobank(LOBank::sptr _lobank)
{
	lobank = _lobank;
	nch = lobank->nch;
	work.resize(nch);
	for(size_t i=0;i<nch;i++) {
		work[i].resize(decimate);
	}
	lo_block.resize(nch);
}

void Channelizer::channelize(queue<complex64> &in_queue, vector<queue<complex64> > &out_queues)
{
	while(in_queue.size() >= decimate) {
		for(size_t i=0;i<decimate;i++) {
			in_block[i] = in_queue.front();
			in_queue.pop();
		}
		
		for(size_t ch=0;ch<nch;ch++) {
			memcpy(&(work[ch][0]),&in_block[0],sizeof(complex64)*decimate);
		}
		for(size_t i=0;i<decimate;i++) {
			lobank->get_sample_rx(lo_block);
			for(size_t ch=0;ch<nch;ch++) {
				work[ch][i] *= lo_block[i];
			}
		}
		for(size_t ch=0;ch<nch;ch++) {
			complex64 t = firdecimate->run(work[ch]);
			out_queues.at(ch).push(t);
		}
	}
}

