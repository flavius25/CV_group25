CC = "g++"

PROJECT = output
SRC = *.cpp utilities/*.cpp controllers/*.cpp 

LIBS = -lglut `pkg-config --cflags --libs opencv4 gl glu`      #`pkg-config --cflags --libs opencv4`
	
$(PROJECT) : $(SRC)
		$(CC) $(SRC) -o $(PROJECT) $(LIBS)
