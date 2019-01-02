CC := g++
STD := -std=c++17
CFLAG := -O1 -mavx2

all: main.cpp
	$(CC) $(STD) $(CFLAG) main.cpp -o run
