CXX = g++
CXXFLAGS = -O3 -Wall -fPIC $(shell root-config --cflags)
LDFLAGS = $(shell root-config --libs)

TARGET = merge_production
SRCS = merge_production.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
