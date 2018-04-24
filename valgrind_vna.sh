valgrind ./vna \
	--file vna_samp.dat\
	--nsamps 100000\
	--settling 1.0\
	--tx-rate 1e6\
	--rx-rate 1e6\
	--tx-freq 200e6\
	--rx-freq 200e6\
	--lo-off 5e6\
	--ampl 0.3\
	--tx-gain 30\
	--rx-gain 30

