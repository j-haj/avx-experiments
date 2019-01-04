CC := g++-7
STD := -std=c++17
CFLAG := -Ofast -mavx -faligned-new=32 -funroll-loops

all: main.cpp
	$(CC) $(STD) $(CFLAG) main.cpp -o run
