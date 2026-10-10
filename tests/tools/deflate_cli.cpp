// Test helper for tests/zlib_interop.py: exposes the internal DEFLATE codec
// as a stdin -> stdout filter so an external implementation can check it.
//
//   deflate_cli deflate <level> [segment_hint]   raw bytes -> raw DEFLATE stream
//   deflate_cli inflate <raw_size>   DEFLATE stream -> raw bytes (exit 1 if invalid)

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "detail/codec.hpp"

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: deflate_cli deflate <level> [segment_hint] | inflate <raw_size>\n";
    return 2;
  }
  const auto segment_hint = static_cast<std::size_t>(argc == 4 ? std::strtoull(argv[3], nullptr, 10) : 0);
  const std::string_view mode = argv[1];
  const unsigned long long argument = std::strtoull(argv[2], nullptr, 10);

  const std::string raw_input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const std::vector<std::uint8_t> input(raw_input.begin(), raw_input.end());

  std::vector<std::uint8_t> output;
  if (mode == "deflate") {
    turbocore::detail::DeflateCodec(static_cast<int>(argument)).compress(input, output, segment_hint);
  } else if (mode == "inflate") {
    output.resize(static_cast<std::size_t>(argument));
    if (!turbocore::detail::DeflateCodec().decompress(input, output)) {
      std::cerr << "deflate_cli: invalid DEFLATE stream\n";
      return 1;
    }
  } else {
    std::cerr << "deflate_cli: unknown mode\n";
    return 2;
  }

  std::cout.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
  return std::cout.good() ? 0 : 1;
}
