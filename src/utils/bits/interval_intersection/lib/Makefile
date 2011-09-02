OBJ_DIR = ../../../../../obj/
UTILITIES_DIR = ../../../../utils/

# -------------------
# define our includes
# -------------------
INCLUDES = -I$(UTILITIES_DIR)/bits/binary_search/lib \
		-I$(UTILITIES_DIR)/bits/utils/c/mt

# ----------------------------------
# define our source and object files
# ----------------------------------
SOURCES= interval.c
OBJECTS= $(SOURCES:.c=.o)
_EXT_OBJECTS=bsearch.o mt.o timer.o
EXT_OBJECTS=$(patsubst %,$(OBJ_DIR)/%,$(_EXT_OBJECTS))
BUILT_OBJECTS= $(patsubst %,$(OBJ_DIR)/%,$(OBJECTS))

$(BUILT_OBJECTS): $(SOURCES)
	@echo "  * compiling" $(*F).c
	@$(CXX) -c -o $@ $(*F).c $(LDFLAGS) $(CXXFLAGS) $(INCLUDES)

$(EXT_OBJECTS):
	@$(MAKE) --no-print-directory -C -W $(INCLUDES)

clean:
	@echo "Cleaning up."
	@rm -f $(OBJ_DIR)/* $(BIN_DIR)/*

.PHONY: clean


