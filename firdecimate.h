#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "types.h"

class FIRDecimate
{
	public:
	typedef std::shared_ptr<FIRDecimate> sptr;
	size_t ntap;
	size_t decimate;
	std::vector<complex64> coeff;
	std::vector<complex64> buf;
	FIRDecimate(const std::string &coeff_fn);
	complex64 run(const std::vector<complex64> &in);
};


