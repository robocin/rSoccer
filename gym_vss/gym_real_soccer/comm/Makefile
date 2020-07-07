# standard compile options for the c++ executable
FLAGS = -fPIC

# the python interface through swig
PYTHONI = -I/usr/include/python3.6
PYTHONL = -Xlinker -export-dynamic

# default super-target
all: 
	g++ -fPIC -c -std=c++11 nrf.cpp -o nrf.o
	swig -c++ -python -o nrf_wrap.cxx nrf.i 
	g++ $(FLAGS) $(PYTHONI) -c -std=c++11 nrf_wrap.cxx -o nrf_wrap.o
	g++ $(PYTHONL) $(LIBFLAGS) -shared nrf.o nrf_wrap.o -o ../_nrf.so

	g++ -fPIC -c -std=c++11 nrf_Communication.cpp -o nrf_Communication.o
	swig -c++ -python -o nrf_Communication_wrap.cxx nrf_Communication.i 
	g++ $(FLAGS) $(PYTHONI) -c -std=c++11 nrf_Communication_wrap.cxx -o nrf_Communication_wrap.o
	g++ $(PYTHONL) $(LIBFLAGS) -shared nrf_Communication.o nrf_Communication_wrap.o -o ../_nrf_Communication.so
