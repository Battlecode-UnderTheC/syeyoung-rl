// Collection benchmark: worker threads per GPU play their matches with the
// exported policy, nothing else (no learner).
//
//   dragons_collect <plugin.so> <programs dir> <maps file> <gpus> <workers per gpu> <matches> <threads> <seconds>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "maps.hpp"
#include "pjrt.hpp"
#include "worker.hpp"

int main(int argc, char** argv) {
  if (argc < 9) {
    std::fprintf(stderr, "usage: %s plugin programs maps gpus workers_per_gpu matches threads seconds\n", argv[0]);
    return 2;
  }
  const std::string plugin = argv[1], programs = argv[2], maps_path = argv[3];
  const int gpus = std::atoi(argv[4]), per = std::atoi(argv[5]), matches = std::atoi(argv[6]),
            threads = std::atoi(argv[7]), seconds = std::atoi(argv[8]);
  rt::MapPool maps(maps_path);
  std::printf("%zu maps\n", maps.size());
  rt::Runtime rt(plugin);
  std::printf("%d devices\n", rt.num_devices());
  const auto actor = rt::read_arrays(programs + "/actor.bin");
  const auto critic = rt::read_arrays(programs + "/critic.bin");
  std::vector<std::unique_ptr<rt::Program>> acts(gpus);
  {
    std::vector<std::thread> compile;
    for (int g = 0; g < gpus; ++g)
      compile.emplace_back([&, g] { acts[g] = std::make_unique<rt::Program>(rt, programs, "act", g); });
    for (auto& t : compile) t.join();
  }
  std::printf("compiled\n");
  std::vector<std::unique_ptr<rt::Worker>> workers;
  for (int g = 0; g < gpus; ++g)
    for (int w = 0; w < per; ++w) {
      workers.push_back(std::make_unique<rt::Worker>(rt, *acts[g], matches, threads,
                                                      [&maps] { return maps.next(); }, 1234 + workers.size()));
      workers.back()->set_weights(actor, actor, critic);
    }
  std::atomic<bool> stop{false};
  std::vector<std::thread> loops;
  for (auto& w : workers)
    loops.emplace_back([&stop, &w] {
      int64_t n = 0;
      while (!stop) {
        w->step();
        if (++n % 16 == 0) w->harvest();
      }
    });
  const auto t0 = std::chrono::steady_clock::now();
  int64_t last = 0;
  for (int s = 10; s <= seconds; s += 10) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    int64_t dec = 0, steps = 0, enc = 0, gpu = 0, app = 0, up = 0, ex = 0;
    for (auto& w : workers) {
      auto& st = w->stats();
      dec += st.decisions; steps += st.steps; enc += st.encode_us; gpu += st.gpu_us; app += st.apply_us;
      up += st.upload_us; ex += st.execute_us;
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double k = 1e6 * el * workers.size();
    std::printf("%5.0fs: %8.0f decisions/s (last 10 s %8.0f), %6.1f steps/s/worker; worker time: encode %.2f "
                "gpu %.2f (upload %.2f execute+download %.2f) apply %.2f\n",
                el, dec / el, (dec - last) / 10.0, steps / el / workers.size(), enc / k, gpu / k, up / k, ex / k, app / k);
    std::fflush(stdout);
    last = dec;
  }
  stop = true;
  for (auto& t : loops) t.join();
  return 0;
}
