
all: vna vna_util_test rfloop rfloopdecim rfdemod txrx_twotone
clean:
	rm -f *.o usrp_mkid vna rfloop rfloopdecim rfdemod txrx_twotone vna_util_test

.SUFFIXES:



LIBDIRS= -L/usr/local/lib
LIBS= -luhd -lboost_program_options -lboost_filesystem -lboost_system -lboost_thread -lpthread
LDFLAGS= $(LIBDIRS) $(LIBS)

CC=clang++-3.8
CC = g++
#CFLAGS= -Wall -Werror -ggdb -std=c++11
CFLAGS= -Wall -O3 -std=c++11 -ffast-math

CC=g++
#CFLAGS= -Wall -O2



vna:  vna.o vna_util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

vna_util_test:  vna_util_test.o vna_util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

rfloop:  rfloop.o vna_util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

rfloopdecim:  rfloopdecim.o vna_util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

rfdemod:  rfdemod.o vna_util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

txrx_twotone:  txrx_twotone.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)



%.o: %.cpp %.h
	$(CC) -c $(CFLAGS) $< -o $@ 

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@ 

