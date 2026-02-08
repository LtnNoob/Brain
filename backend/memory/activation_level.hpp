#pragma once

namespace brain19 {

// Derived activation level (read-only classification)
// Used for external queries only, not for STM internal logic
enum class ActivationLevel {
    LOW,     // activation < 0.3
    MEDIUM,  // 0.3 <= activation < 0.7
    HIGH     // activation >= 0.7
};

// Classification of knowledge for differential decay
enum class ActivationClass {
    CORE_KNOWLEDGE,  // Decays slower (fundamental concepts)
    CONTEXTUAL       // Decays faster (situational concepts)
};

} // namespace brain19
