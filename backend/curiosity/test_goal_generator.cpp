#include "goal_generator.hpp"
#include <cassert>
#include <iostream>

using namespace brain19;

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { \
    tests_total++; \
    std::cout << "  TEST " << tests_total << ": " << name << "... "; \
} while(0)

#define PASS() do { \
    tests_passed++; \
    std::cout << "PASS" << std::endl; \
} while(0)

void test_trigger_to_goal_shallow() {
    TEST("SHALLOW_RELATIONS -> EXPLORATION");
    CuriosityTrigger t(TriggerType::SHALLOW_RELATIONS, 0, {1, 2}, "shallow");
    auto goal = GoalGenerator::from_trigger(t);
    assert(goal.goal_type == GoalType::EXPLORATION);
    assert(goal.priority_weight == 0.4);
    assert(goal.query_text == "shallow");
    PASS();
}

void test_trigger_to_goal_missing_depth() {
    TEST("MISSING_DEPTH -> CAUSAL_CHAIN");
    CuriosityTrigger t(TriggerType::MISSING_DEPTH, 0, {10, 20}, "depth");
    auto goal = GoalGenerator::from_trigger(t);
    assert(goal.goal_type == GoalType::CAUSAL_CHAIN);
    assert(goal.priority_weight == 0.6);
    assert(goal.target_concepts.size() == 2);
    assert(goal.target_concepts[0] == 10);
    PASS();
}

void test_trigger_to_goal_low_exploration() {
    TEST("LOW_EXPLORATION -> EXPLORATION");
    CuriosityTrigger t(TriggerType::LOW_EXPLORATION, 0, {}, "explore");
    auto goal = GoalGenerator::from_trigger(t);
    assert(goal.goal_type == GoalType::EXPLORATION);
    assert(goal.priority_weight == 0.3);
    PASS();
}

void test_trigger_to_goal_recurrent() {
    TEST("RECURRENT_WITHOUT_FUNCTION -> PROPERTY_QUERY");
    CuriosityTrigger t(TriggerType::RECURRENT_WITHOUT_FUNCTION, 0, {5}, "recurrent");
    auto goal = GoalGenerator::from_trigger(t);
    assert(goal.goal_type == GoalType::PROPERTY_QUERY);
    assert(goal.priority_weight == 0.5);
    assert(goal.target_concepts.size() == 1);
    PASS();
}

void test_trigger_to_goal_unknown() {
    TEST("UNKNOWN -> EXPLORATION with low priority");
    CuriosityTrigger t(TriggerType::UNKNOWN, 0, {}, "unknown");
    auto goal = GoalGenerator::from_trigger(t);
    assert(goal.goal_type == GoalType::EXPLORATION);
    assert(goal.priority_weight == 0.2);
    PASS();
}

void test_batch_conversion() {
    TEST("Batch conversion");
    std::vector<CuriosityTrigger> triggers = {
        {TriggerType::SHALLOW_RELATIONS, 0, {1}, "t1"},
        {TriggerType::MISSING_DEPTH, 0, {2}, "t2"},
        {TriggerType::LOW_EXPLORATION, 0, {}, "t3"},
    };
    auto goals = GoalGenerator::from_triggers(triggers);
    assert(goals.size() == 3);
    assert(goals[0].goal_type == GoalType::EXPLORATION);
    assert(goals[1].goal_type == GoalType::CAUSAL_CHAIN);
    assert(goals[2].goal_type == GoalType::EXPLORATION);
    PASS();
}

void test_goal_queue_push_pop() {
    TEST("GoalQueue push/pop order");
    GoalQueue q(10);

    GoalState g1; g1.priority_weight = 0.3; g1.query_text = "low";
    GoalState g2; g2.priority_weight = 0.9; g2.query_text = "high";
    GoalState g3; g3.priority_weight = 0.5; g3.query_text = "mid";

    q.push(g1);
    q.push(g2);
    q.push(g3);
    assert(q.size() == 3);

    auto top = q.pop();
    assert(top.has_value());
    assert(top->query_text == "high");

    top = q.pop();
    assert(top.has_value());
    assert(top->query_text == "mid");

    top = q.pop();
    assert(top.has_value());
    assert(top->query_text == "low");

    assert(q.size() == 0);
    assert(!q.pop().has_value());
    PASS();
}

void test_goal_queue_capacity() {
    TEST("GoalQueue capacity limit");
    GoalQueue q(3);

    for (int i = 0; i < 10; ++i) {
        GoalState g;
        g.priority_weight = static_cast<double>(i) * 0.1;
        q.push(g);
    }
    assert(q.size() == 3);

    // Should have the 3 highest priorities
    auto top = q.pop();
    assert(top.has_value());
    assert(top->priority_weight >= 0.7);
    PASS();
}

void test_goal_queue_peek() {
    TEST("GoalQueue peek");
    GoalQueue q(10);
    assert(!q.peek().has_value());

    GoalState g; g.priority_weight = 0.5; g.query_text = "test";
    q.push(g);

    auto peeked = q.peek();
    assert(peeked.has_value());
    assert(peeked->query_text == "test");
    assert(q.size() == 1);  // Not removed
    PASS();
}

void test_goal_queue_aging() {
    TEST("GoalQueue aging");
    GoalQueue q(10);

    GoalState g; g.priority_weight = 1.0;
    q.push(g);

    q.age(0.5);
    auto top = q.pop();
    assert(top.has_value());
    assert(top->priority_weight == 0.5);
    PASS();
}

void test_goal_queue_prune_completed() {
    TEST("GoalQueue prune completed");
    GoalQueue q(10);

    GoalState g1; g1.priority_weight = 0.5; g1.completion_metric = 0.0; g1.threshold = 0.8;
    GoalState g2; g2.priority_weight = 0.9; g2.completion_metric = 0.9; g2.threshold = 0.8;  // complete
    GoalState g3; g3.priority_weight = 0.3; g3.completion_metric = 0.5; g3.threshold = 0.8;

    q.push(g1);
    q.push(g2);
    q.push(g3);
    assert(q.size() == 3);

    q.prune_completed();
    assert(q.size() == 2);

    auto top = q.pop();
    assert(top.has_value());
    assert(top->completion_metric < 0.8);  // Not the completed one
    PASS();
}

void test_goal_queue_clear() {
    TEST("GoalQueue clear");
    GoalQueue q(10);
    GoalState g; g.priority_weight = 0.5;
    q.push(g);
    q.push(g);
    assert(q.size() == 2);
    q.clear();
    assert(q.size() == 0);
    PASS();
}

int main() {
    std::cout << "=== GoalGenerator & GoalQueue Tests ===" << std::endl;

    // GoalGenerator tests
    test_trigger_to_goal_shallow();
    test_trigger_to_goal_missing_depth();
    test_trigger_to_goal_low_exploration();
    test_trigger_to_goal_recurrent();
    test_trigger_to_goal_unknown();
    test_batch_conversion();

    // GoalQueue tests
    test_goal_queue_push_pop();
    test_goal_queue_capacity();
    test_goal_queue_peek();
    test_goal_queue_aging();
    test_goal_queue_prune_completed();
    test_goal_queue_clear();

    std::cout << "\n=== " << tests_passed << "/" << tests_total << " PASSED ===" << std::endl;
    return (tests_passed == tests_total) ? 0 : 1;
}
