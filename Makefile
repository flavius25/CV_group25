CC = "g++"

PROJECT = output
SRC = VoxelReconstruction.cpp

LIBS = `pkg-config --cflags --libs opencv4`
	
$(PROJECT) : $(SRC)
		$(CC) $(SRC) -o $(PROJECT) $(LIBS)
