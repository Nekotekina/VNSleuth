CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
LIBS = -lX11 -lXfixes

TARGET = xclipmonitor
SOURCES = xclipmonitor.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

install: all
	cp -f $(TARGET) ${HOME}/bin/

clean:
	rm -f $(TARGET)
