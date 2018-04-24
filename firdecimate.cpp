
#include <fstream>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "firdecimate.h"
#include "radio.h"

using namespace std;
/*
 * File format:
 * <ntap : int>
 * <coeff 1 : complex64>
 * ...
 * <coeff ntap : complex64>
 */
FIRDecimate::FIRDecimate(const string &coeff_fn) 
{
	ifstream f(coeff_fn);
	if(not f) {
		cerr << "Could not open " << coeff_fn << " for FIR coefficients" << endl;
		exit(1);
	}

	f >> ntap;
	f >> decimate;
	cout << "FIR decimate: " << decimate << endl;
	cout << "FIR ntap: " << ntap << endl;
	coeff.resize(ntap);
	for(size_t i=0;i<ntap;i++) {
		f >> coeff[i];
	}
	buf.resize(ntap);
}

// Read a block of length decimate and output a sample
complex64 FIRDecimate::run(const vector<complex64> &in)
{
	assert(in.size() == decimate);
	// Shift buffer to the left
	memmove(&buf[0],&buf[decimate],sizeof(complex64)*decimate);
	// Copy in samples to the right side of the buffer
	memcpy(&buf[ntap-decimate],&in[0],sizeof(complex64)*decimate);

	complex64 s = 0.0;
	for(size_t i=0;i<ntap;i++) {
		s += buf[i] * coeff[i];
	}
	return s;
}

