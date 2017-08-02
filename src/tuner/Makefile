all:
	g++ convert.cpp -o convert -O3 -Wall -std=c++11 -march=native -I ../../external/eigen
	g++ gen_queries.cpp -o gen_queries -O3 -Wall -std=c++11 -march=native -I ../../external/eigen
	g++ polygon.cpp -o polygon -O3 -Wall -std=c++11 -march=native -I ../../external/eigen -I ../include -pthread
	g++ tune.cpp -o tune -O3 -Wall -std=c++11 -march=native -I ../../external/eigen -I ../include -pthread
