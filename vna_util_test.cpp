
#include <iostream>
#include <fstream>
#include "vna_util.h"

void test_demod()
{
	float ampl=1.0, wmin=-M_PI*0.9, wmax=M_PI*0.9;
	size_t samp_per_step=1024, nstep=401;
	size_t nt = samp_per_step * nstep;
	ChirpGenerator chirpgen(ampl,wmin,wmax,samp_per_step,nstep);

	vector<sc16> lo(nt);
	chirpgen.fill_buf(lo);

	std::string chirpfn("/tmp/vna_chirp_test.dat");
	ofstream chirp_out(chirpfn);
	for(size_t i=0;i<nstep;i++) {
		for(size_t j=0;j<samp_per_step;j++) {
			sc16 z = lo[i*samp_per_step + j];
			chirp_out << z.real() << " " << z.imag() << " ";
		}
		chirp_out << endl;
	}


	std::string demodfn("/tmp/vna_demod_test.dat");
	ofstream demod_out(demodfn);
	Demodulator demod(&chirpgen,&demod_out);
	
	demod.input_data(lo,lo.size());
}

void test_cic_gain()
{
	size_t nt = 1024;
	vector<sc32> ones(nt);
	vector<sc32> result(nt);
	for(size_t i=0;i<nt;i++) {
		ones[i] = sc32(1,0);
	}

	CIC cic;
	size_t nout = cic.process_block(&result[0],&ones[0],nt);

	cout << "nout " << nout << endl;

	for(size_t i=0;i<nout;i++) {
		cout << result[i].real() << " " << result[i].imag() << endl;
	}
}

void test_cic_noise()
{
	ifstream fin("white.txt");
	ofstream fout("cic_noise.txt");
	size_t n;
	fin >> n;

	vector<sc32> in(n);
	vector<sc32> result(n);
	for(size_t i=0;i<n;i++) {
		uint32_t re,im;
		fin >> re >> im;
		in[i] = sc32(re,im);
	}

	CIC cic;
	size_t nout = cic.process_block(&result[0],&in[0],n);

	for(size_t i=0;i<nout;i++) {
		fout << result[i].real() << " " << result[i].imag() << endl;
	}
}

int main(int argc, char **argv)
{
//	test_demod();
//	test_cic_gain();
	test_cic_noise();
}

