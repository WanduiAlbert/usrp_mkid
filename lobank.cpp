#include <complex>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <queue>

#include "lobank.h"

using namespace std;

Wavetable::Wavetable(const float ampl):
		_table(table_length)
{
	for(unsigned i=0;i<table_length;i++) {
		float phi = static_cast<float>(((2. * M_PI * i) / (4. * table_length)));
		_table[i] = ampl*std::complex<float>(cosf(phi),sinf(phi));
	}
}

std::complex<float> const Wavetable::operator()(const unsigned phase) const
{
	unsigned tp = (phase & phase_mask) >> spare_bits;
	unsigned s = tp >> table_bits;
	unsigned t = tp ^ (s << table_bits);

	std::complex<float> z = _table[t];
	if((s & 1) > 0) {
		z = std::complex<float>(-z.imag(),z.real());
	}
	if((s & 2) > 0) {
		z = -z;
	}
	return z;
}

LOBank::LOBank(size_t _nch) : wavetable(1.0),nch(_nch),rx_queues(nch),steps(nch),phases(nch),ampls(nch)
{
}

void LOBank::set_step(size_t ch, unsigned step)
{
	steps[ch] = step;
}
void LOBank::set_phase(size_t ch, unsigned phase)
{
	phases[ch] = phase;
}
void LOBank::set_ampl(size_t ch, float ampl)
{
	ampls[ch] = ampl;
}

unsigned LOBank::get_step(size_t ch)
{
	return steps[ch];
}
unsigned LOBank::get_phase(size_t ch)
{
	return phases[ch];
}
float LOBank::get_ampl(size_t ch)
{
	return ampls[ch];
}

/* Get local oscillator for several samples for sum of oscillators */
void LOBank::get_sample_tx(vector<complex64> &buf)
{
	complex64 t,s;
	for(size_t i=0;i<buf.size();i++) {
		s = 0.0;
		for(size_t j=0;j<nch;j++) {
			t = ampls[j]*wavetable(phases[j]);
			s += t;
		}
		buf[i] = s;
		for(size_t j=0;j<nch;j++) {
			phases[j] += steps[j];
		}
	}
}

/* Get local oscillator for several samples for one oscillator */
void LOBank::get_sample_rx(size_t ch, vector<complex64> &buf)
{
	for(size_t i=0;i<buf.size();i++) {
		buf[i] = ampls[ch]*wavetable(phases[ch]);
		phases[ch] += steps[ch];
	}
}

void LOBank::modulate(size_t ch, complex64 *buf, size_t n)
{
	for(size_t i=0;i<n;i++) {
		buf[i] *= ampls[ch]*wavetable(phases[ch]);
		phases[ch] += steps[ch];
	}
}

