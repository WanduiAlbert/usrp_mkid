sudo ./rfdemod \
	--infn rf_in.dat \
	--outfn /tmp/rf_out.dat \
	--settling 0.2 \
	--rate 12e6 \
	--tx-freq 200e6 \
	--rx-freq 200e6 \
	--lo-off 0e6 \
	--ampl 0.3 \
	--tx-gain 35 \
	--rx-gain 35

