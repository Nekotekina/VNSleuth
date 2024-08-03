CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
LIBS = common

TARGET = vnsleuth
SOURCES = main.o parser.o translator.o

LLAMA_INC = -Illama.cpp/ggml/include -Illama.cpp/ggml/src -Illama.cpp/include -Illama.cpp/src -Illama.cpp/common

all: $(TARGET)

main.o: main.cpp main.hpp tools/tiny_sha1.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp $(LLAMA_INC)

parser.o: parser.cpp main.hpp tools/iconv.hpp
	$(CXX) $(CXXFLAGS) -c parser.cpp

translator.o: translator.cpp main.hpp
	$(CXX) $(CXXFLAGS) -c translator.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

install: all
	cp -f $(TARGET) ${HOME}/bin/

clean:
	rm -f $(TARGET) *.o
