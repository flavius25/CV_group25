CC = "g++"

PROJECT = output
SRC = Calibration.cpp

LIBS = `pkg-config --cflags --libs opencv4`
	
$(PROJECT) : $(SRC)
		$(CC) $(SRC) -o $(PROJECT) $(LIBS)
