# ==========================
# BEDTools Makefile
# (c) 2009 Aaron Quinlan
# ==========================

SHELL := /bin/bash -e

# define our object and binary directories
export OBJ_DIR	= obj
export BIN_DIR	= bin
export SRC_DIR	= src
export UTIL_DIR	= src/utils
export CXX		= g++
export CXXFLAGS = -Wall -O2 -D_FILE_OFFSET_BITS=64 -fPIC
export LIBS		= -lz
export BT_ROOT  = src/utils/BamTools/


SUBDIRS = $(SRC_DIR)/annotateBed \
		  $(SRC_DIR)/bamToBed \
		  $(SRC_DIR)/bedToBam \
		  $(SRC_DIR)/bedpeToBam \
		  $(SRC_DIR)/bedToIgv \
		  $(SRC_DIR)/bed12ToBed6 \
		  $(SRC_DIR)/closestBed \
		  $(SRC_DIR)/clusterBed \
		  $(SRC_DIR)/complementBed \
		  $(SRC_DIR)/coverageBed \
		  $(SRC_DIR)/fastaFromBed \
		  $(SRC_DIR)/flankBed \
		  $(SRC_DIR)/genomeCoverageBed \
		  $(SRC_DIR)/getOverlap \
		  $(SRC_DIR)/groupBy \
		  $(SRC_DIR)/intersectBed \
		  $(SRC_DIR)/linksBed \
		  $(SRC_DIR)/maskFastaFromBed \
		  $(SRC_DIR)/mapBed \
		  $(SRC_DIR)/mergeBed \
		  $(SRC_DIR)/multiBamCov \
		  $(SRC_DIR)/multiIntersectBed \
		  $(SRC_DIR)/nucBed \
		  $(SRC_DIR)/pairToBed \
		  $(SRC_DIR)/pairToPair \
		  $(SRC_DIR)/shuffleBed \
		  $(SRC_DIR)/slopBed \
		  $(SRC_DIR)/sortBed \
		  $(SRC_DIR)/subtractBed \
		  $(SRC_DIR)/tagBam \
		  $(SRC_DIR)/unionBedGraphs \
		  $(SRC_DIR)/windowBed \
		  $(SRC_DIR)/windowMaker

UTIL_SUBDIRS =	$(SRC_DIR)/utils/lineFileUtilities \
				$(SRC_DIR)/utils/bedFile \
				$(SRC_DIR)/utils/bedGraphFile \
				$(SRC_DIR)/utils/chromsweep \
				$(SRC_DIR)/utils/gzstream \
				$(SRC_DIR)/utils/fileType \
				$(SRC_DIR)/utils/bedFilePE \
				$(SRC_DIR)/utils/sequenceUtilities \
				$(SRC_DIR)/utils/tabFile \
				$(SRC_DIR)/utils/BamTools \
				$(SRC_DIR)/utils/BamTools-Ancillary \
				$(SRC_DIR)/utils/Fasta \
				$(SRC_DIR)/utils/genomeFile

BUILT_OBJECTS = $(OBJ_DIR)/*.o
# BUILT_OBJECTS = $(OBJ_DIR)/bedtools.o \
# 				$(OBJ_DIR)/BamAncillary.o \
#                 $(OBJ_DIR)/Fasta.o \
#                 $(OBJ_DIR)/bedFile.o \
#                 $(OBJ_DIR)/bedFilePE.o \
#                 $(OBJ_DIR)/bedGraphFile.o \
#                 $(OBJ_DIR)/chromsweep.o \
#                 $(OBJ_DIR)/fileType.o \
#                 $(OBJ_DIR)/gzstream.o \
#                 $(OBJ_DIR)/sequenceUtils.o \
#                 $(OBJ_DIR)/split.o \
#                 $(OBJ_DIR)/intersectBed.o \
#                 $(OBJ_DIR)/intersectMain.o \

all:
	[ -d $(OBJ_DIR) ] || mkdir -p $(OBJ_DIR)
	[ -d $(BIN_DIR) ] || mkdir -p $(BIN_DIR)
	
	@echo "Building BEDTools:"
	@echo "========================================================="
	
	@for dir in $(UTIL_SUBDIRS); do \
		echo "- Building in $$dir"; \
		$(MAKE) --no-print-directory -C $$dir; \
		echo ""; \
	done

	@for dir in $(SUBDIRS); do \
		echo "- Building in $$dir"; \
		$(MAKE) --no-print-directory -C $$dir; \
		echo ""; \
	done

	@echo "- Building main bedtools binary."
	@$(CXX) $(CXXFLAGS) -c src/bedtools.cpp -o obj/bedtools.o -I$(UTIL_DIR)/version/
	@$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $(BIN_DIR)/bedtools $(BUILT_OBJECTS) -L$(UTIL_DIR)/BamTools/lib/ -lbamtools $(LIBS)
	@echo "done."
	
	@echo "- Creating executables for old CLI."
	@python scripts/makeBashScripts.py
	@chmod +x bin/*
	@echo "done."
	

.PHONY: all

clean:
	@echo "Cleaning up."	
	@rm -f $(OBJ_DIR)/* $(BIN_DIR)/*
	@rm -Rf $(BT_ROOT)/lib
	@rm -f $(BT_ROOT)/src/api/*.o
	@rm -f $(BT_ROOT)/src/api/internal/*.o
	@rm -Rf $(BT_ROOT)/include

.PHONY: clean
