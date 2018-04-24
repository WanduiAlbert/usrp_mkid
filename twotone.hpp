//
// Copyright 2010-2012,2014 Ettus Research LLC
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <stdexcept>
#include <math.h>

static const unsigned int phase_mask = 0xffffffff;
static const unsigned int phase_bits = 32;
static const unsigned int table_bits = 14;
static const unsigned int sym_bits = 2;
static const unsigned int table_length = 1<<table_bits;
static const unsigned int spare_bits = phase_bits - (table_bits + sym_bits);

class nco_class {
private:
	std::vector<std::complex<float> > _table;

public:
	nco_class():
		_table(table_length)
	{
		for(size_t i=0;i<table_length;i++) {
			double phi = (2. * M_PI * i) / (4. * table_length);
			_table[i] = std::complex<float>(cos(phi),sin(phi));
		}
	}

	std::complex<float> operator()(const size_t phase)
	{
		unsigned int tp = (phase & phase_mask) >> spare_bits;
		unsigned int s = tp >> table_bits;
		unsigned int t = tp ^ (s << table_bits);

		std::complex<float> z = _table[t];
		if((s & 1) > 0) {
			z = std::complex<float>(-z.imag(),z.real());
		}
		if((s & 2) > 0) {
			z = -z;
		}
		return z;
	}
};

