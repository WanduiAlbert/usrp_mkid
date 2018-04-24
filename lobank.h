#pragma once

#include <complex>
#include <vector>
#include <assert.h>
#include <memory>
#include <mutex>
#include <queue>

#include "types.h"

static const unsigned phase_mask = 0xffffffff;
static const unsigned phase_bits = 32;
static const unsigned table_bits = 14;
static const unsigned sym_bits = 2;
static const unsigned table_length = 1<<table_bits;
static const unsigned spare_bits = phase_bits - (table_bits + sym_bits);

/*
 * Wave table
 */
class Wavetable {
	public:
		std::vector<complex64> _table;
		Wavetable(const float ampl);
		complex64 const operator()(const unsigned phase) const;
};

class LOBank {
	public:
		typedef std::shared_ptr<LOBank> sptr;
		const Wavetable wavetable;
		size_t nch;
		std::vector<std::queue<complex64> > rx_queues;
		std::queue<complex64> tx_queue;
		std::vector<unsigned> steps;
		std::vector<unsigned> phases;
		std::vector<float> ampls;

		LOBank(size_t _nch);
		void set_step(size_t ch, unsigned step);
		void set_phase(size_t ch, unsigned phase);
		void set_ampl(size_t ch, float ampl);
		unsigned get_step(size_t ch);
		unsigned get_phase(size_t ch);
		float get_ampl(size_t ch);

		void get_sample_tx(std::vector<complex64> &buf);
		void get_sample_rx(size_t ch, std::vector<complex64> &buf);
		void modulate(size_t ch, std::vector<complex64> &buf);
};


