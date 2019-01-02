CC := g++-8
STD := -std=c++17
CFLAG := -O3

all: main.cpp
	$(CC) $(STD) $(CFLAGS) main.cpp -o run
