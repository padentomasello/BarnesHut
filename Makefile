OBJS = clhelp.o barnes_hut.o

UNAME_S := $(shell uname -s)

#check if linux
ifeq ($(UNAME_S), Linux)
OCL_INC=/usr/local/cuda-4.2/include
OCL_LIB=/usr/local/cuda-4.2/lib64
%.o: %.cpp clhelp.h
	g++ -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	g++ barnes_hut.o clhelp.o -o barnes_hut -L$(OCL_LIB) -lOpenCL
endif

#check if os x
ifeq ($(UNAME_S), Darwin)
%.o: %.cpp clhelp.h
	g++  -O2 -c $<

all: $(OBJS)
	g++ barnes_hut.o clhelp.o -o barnes_hut -framework OpenCL
endif






clean:
	rm -rf $(OBJS) reduce
