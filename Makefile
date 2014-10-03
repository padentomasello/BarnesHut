OBJS = barnes_hut.o clhelp.o

UNAME_S := $(shell uname -s)

#check if os x
ifeq ($(UNAME_S), Darwin)
%.o: %.cpp clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ barnes_hut.o clhelp.o -o barnes_hut -framework OpenCL
endif

clean:
	rm -rf $(OBJS) reduce barnes_hut

