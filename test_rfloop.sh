sudo ./rfloop \
	--infn rf_in.dat\
	--outfn /tmp/rf_out.dat\
	--settling 1.0\
	--tx-rate 30e6\
	--rx-rate 30e6\
	--tx-freq 265e6\
	--rx-freq 265e6\
	--lo-off 0e6\
	--ampl 0.3\
	--tx-gain 60\
	--rx-gain 25

