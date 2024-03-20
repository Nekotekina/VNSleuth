CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
LIBS =

TARGET = vnsleuth
SOURCES = main.o parser.o

all: $(TARGET)

main.o: main.cpp main.hpp tools/tiny_sha1.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp

parser.o: parser.cpp main.hpp tools/iconv.hpp
	$(CXX) $(CXXFLAGS) -c parser.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

install: all
	cp -f $(TARGET) ${HOME}/bin/

clean:
	rm -f $(TARGET) *.o
