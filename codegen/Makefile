.PHONY: all clean

BUF_REQUIRED_VERSION := 1.69.0

all:
	@actual=$$(buf --version 2>&1); \
	if [ "$$actual" != "$(BUF_REQUIRED_VERSION)" ]; then \
	  echo "ERROR: buf $(BUF_REQUIRED_VERSION) required, but found $$actual"; \
	  echo "       Update buf or edit BUF_REQUIRED_VERSION in Makefile if intentionally upgrading."; \
	  exit 1; \
	fi
	buf generate ../proto

clean:
	find ../src/claveles/pb/claveles_cloud -name '*_pb2.py' -o -name '*_pb2.pyi' -o -name '*_pb2_grpc.py' | xargs rm -f
