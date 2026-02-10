#include "persistent/stm_snapshot.hpp"
#include "memory/stm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

using namespace brain19;

int main() {
    const std::string snap_path = "/tmp/test_stm_snapshot.bin";

    // 1. Create STM with 3 contexts, each 50 concepts + 20 relations
    ShortTermMemory stm;
    ContextId ctx_ids[3];
    for (int i = 0; i < 3; ++i) {
        ctx_ids[i] = stm.create_context();
    }

    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 50; ++c) {
            ConceptId cid = static_cast<ConceptId>(i * 1000 + c);
            double act = 0.1 + 0.8 * (c / 49.0);
            ActivationClass cls = (c % 3 == 0) ? ActivationClass::CORE_KNOWLEDGE : ActivationClass::CONTEXTUAL;
            stm.activate_concept(ctx_ids[i], cid, act, cls);
        }
        for (int r = 0; r < 20; ++r) {
            ConceptId src = static_cast<ConceptId>(i * 1000 + r);
            ConceptId tgt = static_cast<ConceptId>(i * 1000 + r + 1);
            RelationType type = static_cast<RelationType>(r % 10);
            double act = 0.2 + 0.6 * (r / 19.0);
            stm.activate_relation(ctx_ids[i], src, tgt, type, act);
        }
    }

    // Verify initial counts
    for (int i = 0; i < 3; ++i) {
        assert(stm.debug_active_concept_count(ctx_ids[i]) == 50);
        assert(stm.debug_active_relation_count(ctx_ids[i]) == 20);
    }

    // 2. Snapshot
    STMSnapshotManager mgr(5);
    bool ok = mgr.create_snapshot(stm, snap_path);
    assert(ok);
    std::cout << "[OK] Snapshot created: " << snap_path << "\n";

    // Save some reference activations
    double ref_concept = stm.get_concept_activation(ctx_ids[1], 1025);
    double ref_relation = stm.get_relation_activation(ctx_ids[2], 2005, 2006);

    // 3. Clear STM
    for (int i = 0; i < 3; ++i) {
        stm.destroy_context(ctx_ids[i]);
    }
    assert(stm.debug_active_concept_count(ctx_ids[0]) == 0);
    std::cout << "[OK] STM cleared\n";

    // 4. Load & apply
    STMSnapshotData data;
    ok = mgr.load_snapshot(snap_path, data);
    assert(ok);
    assert(data.contexts.size() == 3);
    std::cout << "[OK] Snapshot loaded: " << data.contexts.size() << " contexts\n";

    mgr.apply_snapshot(stm, data);

    // 5. Verify
    for (auto& sc : data.contexts) {
        assert(stm.debug_active_concept_count(sc.context_id) == 50);
        assert(stm.debug_active_relation_count(sc.context_id) == 20);
    }

    double restored_concept = stm.get_concept_activation(ctx_ids[1], 1025);
    double restored_relation = stm.get_relation_activation(ctx_ids[2], 2005, 2006);
    assert(std::abs(restored_concept - ref_concept) < 1e-12);
    assert(std::abs(restored_relation - ref_relation) < 1e-12);

    std::cout << "[OK] All activations verified identical\n";
    std::cout << "     concept[1][1025] = " << restored_concept << "\n";
    std::cout << "     relation[2][2005->2006] = " << restored_relation << "\n";

    // Cleanup
    std::filesystem::remove(snap_path);
    std::cout << "\n=== Phase 1.3 STM Snapshot Test PASSED ===\n";
    return 0;
}
