CC = gcc
CONFIGFLAGS=#-ULINSOLVERS_RETAIN_MEMORY
CFLAGS=$(CONFIGFLAGS) $(ARCHFLAGS) -O3 -funroll-loops -Wall -I/opt/homebrew/Cellar/gsl/2.7.1/include
LDFLAGS = -L/opt/homebrew/Cellar/gsl/2.7.1/lib -L.
LIBS = -lgsl -lgslcblas -lm

LIBOBJS=lm_mle.o gfit.o
LIBSRCS=lm_mle.c gfit.c

TESTOBJS = test.o
TESTSRCS = test.c

all: liblm_mle.dylib libgfit.dylib liblm_mle_de.dylib test

lm_mle.o: lm_mle.c lm_mle_compiler.h levmar_mle.h
	$(CC) $(CFLAGS) -c lm_mle.c

gfit.o: gfit.c lm_mle_compiler.h levmar_mle.h
	$(CC) $(CFLAGS) -c gfit.c

test.o: test.c lm_mle_compiler.h levmar_mle.h gfit.h
	$(CC) $(CFLAGS) -c test.c

liblm_mle.dylib: lm_mle.o
	$(CC) -shared -o liblm_mle.dylib lm_mle.o $(LDFLAGS) $(LIBS)

libgfit.dylib: gfit.o
	$(CC) -shared -o libgfit.dylib gfit.o $(LDFLAGS) -llm_mle

liblm_mle_de.dylib: lm_mle.o gfit.o
	$(CC) -shared -o liblm_mle_de.dylib lm_mle.o gfit.o $(LDFLAGS) $(LIBS)

test: test.o libgfit.dylib liblm_mle.dylib
	$(CC) -o test test.o $(LDFLAGS) -lgfit -llm_mle -lm

clean:
	@rm -f *.o *.a *.dylib test

depend:
	makedepend -f Makefile $(LIBSRCS)