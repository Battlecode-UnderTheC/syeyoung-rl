// Which GRU slots of a device are in use. Indices only; the states themselves
// live in the GruPool on the device.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace train {

struct Slot {
  int32_t index;
  friend bool operator==(Slot, Slot) = default;
};

class SlotPool {
 public:
  explicit SlotPool(int32_t capacity);
  std::optional<Slot> Acquire();   // nullopt when full
  void Release(Slot s);            // a slot not held is a logic error (checked)
  int32_t in_use() const;
  int32_t capacity() const;
};

}  // namespace train
