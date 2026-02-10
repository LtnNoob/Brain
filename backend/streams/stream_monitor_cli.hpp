#pragma once

#include "stream_monitor.hpp"
#include <string>
#include <vector>

namespace brain19 {

class StreamMonitorCLI {
public:
    explicit StreamMonitorCLI(StreamMonitor& monitor);

    // CLI commands — each returns formatted ASCII table string
    std::string cmd_status() const;
    std::string cmd_throughput() const;
    std::string cmd_latency() const;
    std::string cmd_categories() const;
    std::string cmd_alerts() const;
    std::string cmd_history(size_t seconds = 60) const;

    // Dispatch by command name
    std::string dispatch(const std::string& command, const std::vector<std::string>& args = {}) const;

    // Print usage
    static std::string usage();

private:
    StreamMonitor& monitor_;
};

} // namespace brain19
