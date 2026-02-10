#include "stream_monitor_cli.hpp"
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace brain19 {

StreamMonitorCLI::StreamMonitorCLI(StreamMonitor& monitor)
    : monitor_(monitor) {}

static const char* state_str(StreamState s) {
    switch (s) {
        case StreamState::Created:  return "Created";
        case StreamState::Starting: return "Starting";
        case StreamState::Running:  return "Running";
        case StreamState::Paused:   return "Paused";
        case StreamState::Stopping: return "Stopping";
        case StreamState::Stopped:  return "Stopped";
        case StreamState::Error:    return "Error";
        default:                    return "???";
    }
}

static const char* alert_level_str(AlertLevel l) {
    return l == AlertLevel::Critical ? "CRIT" : "WARN";
}

static const char* alert_type_str(AlertType t) {
    switch (t) {
        case AlertType::Stall:          return "Stall";
        case AlertType::ThroughputDrop: return "ThroughputDrop";
        case AlertType::HighLatency:    return "HighLatency";
        case AlertType::ErrorRate:      return "ErrorRate";
        default:                        return "???";
    }
}

std::string StreamMonitorCLI::cmd_status() const {
    auto streams = monitor_.stream_snapshots();
    auto gs = monitor_.global_snapshot();

    std::ostringstream os;
    os << "=== Brain19 Stream Status ===\n";
    os << "Active: " << gs.active_streams << "/" << gs.total_streams
       << "  Throughput: " << std::fixed << std::setprecision(1) << gs.total_throughput << " ticks/s"
       << "  Load: " << std::setprecision(0) << gs.system_load * 100 << "%\n\n";

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-4s %-12s %-10s %10s %6s %8s\n",
                  "ID", "Category", "State", "Ticks", "Idle%", "Queue");
    os << buf;
    os << std::string(54, '-') << "\n";

    for (auto& s : streams) {
        std::snprintf(buf, sizeof(buf), "%-4u %-12.*s %-10s %10lu %5.1f%% %8zu\n",
                      s.stream_id,
                      static_cast<int>(category_name(s.category).size()),
                      category_name(s.category).data(),
                      state_str(s.state),
                      static_cast<unsigned long>(s.total_ticks),
                      s.idle_pct,
                      s.queue_depth);
        os << buf;
    }
    return os.str();
}

std::string StreamMonitorCLI::cmd_throughput() const {
    auto streams = monitor_.stream_snapshots();
    auto gs = monitor_.global_snapshot();

    std::ostringstream os;
    os << "=== Throughput ===\n";

    char buf[128];
    std::snprintf(buf, sizeof(buf), "%-4s %-12s %12s\n", "ID", "Category", "Ticks/sec");
    os << buf;
    os << std::string(30, '-') << "\n";

    for (auto& s : streams) {
        std::snprintf(buf, sizeof(buf), "%-4u %-12.*s %12.1f\n",
                      s.stream_id,
                      static_cast<int>(category_name(s.category).size()),
                      category_name(s.category).data(),
                      s.ticks_per_sec);
        os << buf;
    }
    os << std::string(30, '-') << "\n";
    std::snprintf(buf, sizeof(buf), "%-17s %12.1f\n", "TOTAL", gs.total_throughput);
    os << buf;
    return os.str();
}

std::string StreamMonitorCLI::cmd_latency() const {
    auto stats = monitor_.latency_stats();

    std::ostringstream os;
    os << "=== Latency Histogram ===\n";
    os << "Samples: " << stats.n << "\n\n";

    char buf[128];
    std::snprintf(buf, sizeof(buf), "  p50:  %10.1f us\n", stats.p50);  os << buf;
    std::snprintf(buf, sizeof(buf), "  p95:  %10.1f us\n", stats.p95);  os << buf;
    std::snprintf(buf, sizeof(buf), "  p99:  %10.1f us\n", stats.p99);  os << buf;
    std::snprintf(buf, sizeof(buf), "  max:  %10.1f us\n", stats.max);  os << buf;
    return os.str();
}

std::string StreamMonitorCLI::cmd_categories() const {
    auto cats = monitor_.category_snapshots();

    std::ostringstream os;
    os << "=== Categories ===\n";

    char buf[128];
    std::snprintf(buf, sizeof(buf), "%-12s %8s %12s %6s %12s\n",
                  "Category", "Streams", "Ticks/sec", "Idle%", "TotalTicks");
    os << buf;
    os << std::string(54, '-') << "\n";

    for (auto& c : cats) {
        std::snprintf(buf, sizeof(buf), "%-12.*s %8u %12.1f %5.1f%% %12lu\n",
                      static_cast<int>(category_name(c.category).size()),
                      category_name(c.category).data(),
                      c.stream_count,
                      c.total_throughput,
                      c.avg_idle_pct,
                      static_cast<unsigned long>(c.total_ticks));
        os << buf;
    }
    return os.str();
}

std::string StreamMonitorCLI::cmd_alerts() const {
    auto alerts = monitor_.active_alerts();

    std::ostringstream os;
    os << "=== Active Alerts ===\n";

    if (alerts.empty()) {
        os << "(none)\n";
        return os.str();
    }

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-5s %-16s %-6s %s\n", "Level", "Type", "Stream", "Message");
    os << buf;
    os << std::string(60, '-') << "\n";

    for (auto& a : alerts) {
        std::snprintf(buf, sizeof(buf), "%-5s %-16s %-6u %s\n",
                      alert_level_str(a.level),
                      alert_type_str(a.type),
                      a.stream_id,
                      a.message.c_str());
        os << buf;
    }
    return os.str();
}

std::string StreamMonitorCLI::cmd_history(size_t seconds) const {
    auto entries = monitor_.history(seconds);

    std::ostringstream os;
    os << "=== History (last " << seconds << "s) ===\n";
    os << "Entries: " << entries.size() << "\n\n";

    if (entries.empty()) {
        os << "(no data)\n";
        return os.str();
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "%-8s %12s %8s %10s\n",
                  "Time(s)", "Ticks/sec", "Active", "p99(us)");
    os << buf;
    os << std::string(42, '-') << "\n";

    auto base = entries.front().timestamp;
    // Downsample: show 1 entry per second
    size_t step = std::max<size_t>(1, entries.size() / std::min<size_t>(seconds, entries.size()));
    for (size_t i = 0; i < entries.size(); i += step) {
        auto& e = entries[i];
        double t = std::chrono::duration<double>(e.timestamp - base).count();
        std::snprintf(buf, sizeof(buf), "%7.1fs %12.1f %8u %10.1f\n",
                      t, e.total_throughput, e.active_streams, e.p99_latency_us);
        os << buf;
    }
    return os.str();
}

std::string StreamMonitorCLI::dispatch(const std::string& command,
                                        const std::vector<std::string>& args) const {
    if (command == "status")     return cmd_status();
    if (command == "throughput") return cmd_throughput();
    if (command == "latency")   return cmd_latency();
    if (command == "categories") return cmd_categories();
    if (command == "alerts")    return cmd_alerts();
    if (command == "history") {
        size_t secs = 60;
        if (!args.empty()) {
            try { secs = std::stoul(args[0]); } catch (...) {}
        }
        return cmd_history(secs);
    }
    return "Unknown command: " + command + "\n\n" + usage();
}

std::string StreamMonitorCLI::usage() {
    return "Usage: brain19_monitor <command> [args]\n"
           "\nCommands:\n"
           "  status       Overview of all streams\n"
           "  throughput   Ticks/sec per stream and total\n"
           "  latency      Latency histogram (p50/p95/p99/max)\n"
           "  categories   Per-category summary\n"
           "  alerts       Active alerts\n"
           "  history [N]  Throughput trend (last N seconds, default 60)\n";
}

} // namespace brain19
