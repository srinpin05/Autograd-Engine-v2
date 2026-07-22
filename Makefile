.PHONY: run rebuild clean

BUILD_DIR := build
BIN       := $(BUILD_DIR)/bin/autograd_engine

$(BUILD_DIR)/Makefile: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR)

$(BIN): $(BUILD_DIR)/Makefile
	cmake --build $(BUILD_DIR)

run: $(BIN)
	$(BIN)

rebuild: clean run

clean:
	rm -rf $(BUILD_DIR)