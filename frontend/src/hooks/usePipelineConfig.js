import { useState, useCallback } from 'react';

const STORAGE_KEY = 'brain19_pipeline';
const API_BASE = '/api/codegen';

// Category colors matching dark theme
export const CATEGORY_COLORS = {
  memory: '#3b82f6',
  cognitive: '#8b5cf6',
  traversal: '#06b6d4',
  micromodel: '#22c55e',
  curiosity: '#f59e0b',
  understanding: '#ec4899',
  hybrid: '#ef4444',
  background: '#6b7280',
};

// Trigger type colors for edges
export const TRIGGER_COLORS = {
  sequential: '#6366f1',
  conditional: '#f59e0b',
  feedback: '#10b981',
  background: '#6b7280',
};

function getDefaultNodes() {
  return [
    {
      id: 'stm',
      type: 'subsystem',
      position: { x: 80, y: 250 },
      data: {
        label: 'Seed Activation',
        subsystem: 'STM + BrainController',
        category: 'memory',
        step: 1,
        enabled: true,
        estimatedMs: 1,
        outputs: ['activated concept IDs'],
        description: 'Activates seed concepts in Short-Term Memory with initial activation value (0.8). Entry point for the thinking cycle.',
      },
    },
    {
      id: 'spreading',
      type: 'subsystem',
      position: { x: 300, y: 250 },
      data: {
        label: 'Spreading Activation',
        subsystem: 'CognitiveDynamics',
        category: 'cognitive',
        step: 2,
        enabled: true,
        estimatedMs: 5,
        outputs: ['SpreadingStats', 'activated neighbors'],
        description: 'Spreads activation from seeds through the knowledge graph. Energy decays with distance. Multi-hop propagation.',
      },
    },
    {
      id: 'cursor',
      type: 'subsystem',
      position: { x: 520, y: 150 },
      data: {
        label: 'FocusCursor',
        subsystem: 'FocusCursorManager',
        category: 'traversal',
        step: 2.5,
        enabled: true,
        estimatedMs: 15,
        outputs: ['TraversalResult', 'best chain'],
        description: 'Goal-directed graph traversal. Seeds augmented by GDO top-3 activations. Results fed back to GDO.',
      },
    },
    {
      id: 'salience',
      type: 'subsystem',
      position: { x: 740, y: 250 },
      data: {
        label: 'Salience & Focus',
        subsystem: 'CognitiveDynamics',
        category: 'cognitive',
        step: 3,
        enabled: true,
        estimatedMs: 3,
        outputs: ['top-k SalienceScores', 'focus state'],
        description: 'Computes salience scores for active concepts. Initializes cognitive focus on top-k salient concepts.',
      },
    },
    {
      id: 'relevance',
      type: 'subsystem',
      position: { x: 960, y: 250 },
      data: {
        label: 'RelevanceMaps',
        subsystem: 'MicroModels',
        category: 'micromodel',
        step: '4-5',
        enabled: true,
        estimatedMs: 20,
        outputs: ['combined RelevanceMap'],
        description: 'Generates bilinear relevance maps per salient concept, then combines via weighted overlay for creative associations.',
      },
    },
    {
      id: 'paths',
      type: 'subsystem',
      position: { x: 1180, y: 250 },
      data: {
        label: 'ThoughtPaths',
        subsystem: 'CognitiveDynamics',
        category: 'cognitive',
        step: 6,
        enabled: true,
        estimatedMs: 10,
        outputs: ['top-20 ThoughtPaths'],
        description: 'Finds best thought paths from each seed concept through the activation landscape. Sorted by path score.',
      },
    },
    {
      id: 'curiosity',
      type: 'subsystem',
      position: { x: 1400, y: 250 },
      data: {
        label: 'CuriosityEngine',
        subsystem: 'CuriosityEngine + GoalGenerator',
        category: 'curiosity',
        step: 7,
        enabled: true,
        estimatedMs: 2,
        outputs: ['CuriosityTriggers', 'GoalStates'],
        description: 'Observes STM state for knowledge gaps. Generates triggers (SHALLOW_RELATIONS, MISSING_DEPTH, etc.) and converts to GoalStates.',
      },
    },
    {
      id: 'understanding',
      type: 'subsystem',
      position: { x: 1620, y: 250 },
      data: {
        label: 'UnderstandingLayer',
        subsystem: 'MiniLLMs',
        category: 'understanding',
        step: 8,
        enabled: true,
        estimatedMs: 50,
        outputs: ['UnderstandingResult', 'HypothesisProposals'],
        description: 'Runs MiniLLM understanding cycle on top salient concept. Generates hypothesis proposals for validation.',
      },
    },
    {
      id: 'kan',
      type: 'subsystem',
      position: { x: 1840, y: 250 },
      data: {
        label: 'KAN Validation',
        subsystem: 'KanValidator',
        category: 'hybrid',
        step: 9,
        enabled: true,
        estimatedMs: 30,
        outputs: ['ValidationResults'],
        description: 'Validates hypothesis proposals using Kolmogorov-Arnold Networks. Provides confidence scores and explanations.',
      },
    },
    {
      id: 'gdo',
      type: 'subsystem',
      position: { x: 520, y: 480 },
      data: {
        label: 'Global Dynamics',
        subsystem: 'GlobalDynamicsOperator',
        category: 'background',
        step: 'BG',
        enabled: true,
        estimatedMs: 500,
        outputs: ['activation landscape', 'autonomous thinking triggers'],
        description: 'Background thread. Ticks every 500ms: decay activations, prune, maybe trigger autonomous thinking when energy > 30.',
      },
    },
  ];
}

function getDefaultEdges() {
  return [
    // Main pipeline chain
    { id: 'e-stm-spreading', source: 'stm', target: 'spreading', type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: 'activated ConceptIds' } },
    { id: 'e-spreading-cursor', source: 'spreading', target: 'cursor', type: 'timing',
      data: { triggerType: 'conditional', delayMs: 0, dataFlow: 'seeds + spread state' } },
    { id: 'e-cursor-salience', source: 'cursor', target: 'salience', type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: 'traversal result + active concepts' } },
    { id: 'e-salience-relevance', source: 'salience', target: 'relevance', type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: 'top-k SalienceScores' } },
    { id: 'e-relevance-paths', source: 'relevance', target: 'paths', type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: 'combined RelevanceMap' } },
    { id: 'e-paths-curiosity', source: 'paths', target: 'curiosity', type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: 'ThoughtPaths + STM state' } },
    { id: 'e-curiosity-understanding', source: 'curiosity', target: 'understanding', type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: 'triggers + salient IDs' } },
    { id: 'e-understanding-kan', source: 'understanding', target: 'kan', type: 'timing',
      data: { triggerType: 'conditional', delayMs: 0, dataFlow: 'HypothesisProposals (if any)' } },

    // GDO feedback loops
    { id: 'e-gdo-cursor', source: 'gdo', target: 'cursor', type: 'timing',
      data: { triggerType: 'background', delayMs: 500, dataFlow: 'top-3 activation snapshot (seed augmentation)' } },
    { id: 'e-cursor-gdo', source: 'cursor', target: 'gdo', type: 'timing',
      data: { triggerType: 'feedback', delayMs: 0, dataFlow: 'TraversalResult (feed_traversal_result)' } },
    { id: 'e-gdo-stm', source: 'gdo', target: 'stm', type: 'timing',
      data: { triggerType: 'background', delayMs: 500, dataFlow: 'energy injection on user query' } },
    { id: 'e-curiosity-gdo', source: 'curiosity', target: 'gdo', type: 'timing',
      data: { triggerType: 'feedback', delayMs: 0, dataFlow: 'GoalStates (via GoalQueue)' } },
  ];
}

// ─── API Integration ──────────────────────────────────────────────────────────

async function fetchTemplates() {
  try {
    const res = await fetch(`${API_BASE}/templates`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.templates;
  } catch {
    return null;
  }
}

async function generateCode(nodes) {
  const steps = nodes
    .filter((n) => n.data.step !== 'BG')
    .sort((a, b) => {
      const sa = parseFloat(a.data.step) || 99;
      const sb = parseFloat(b.data.step) || 99;
      return sa - sb;
    })
    .map((n) => ({
      step_id: n.id,
      enabled: n.data.enabled,
      params: n.data.params || {},
    }));

  // Add GDO if present and enabled
  const gdo = nodes.find((n) => n.id === 'gdo');
  if (gdo) {
    steps.push({
      step_id: 'gdo',
      enabled: gdo.data.enabled,
      params: gdo.data.params || {},
    });
  }

  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ steps }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function getCurrentPipeline() {
  try {
    const res = await fetch(`${API_BASE}/current`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function usePipelineConfig() {
  const [nodes, setNodes] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.nodes?.length > 0) return parsed.nodes;
      }
    } catch (e) { /* ignore */ }
    return getDefaultNodes();
  });

  const [edges, setEdges] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.edges?.length > 0) return parsed.edges;
      }
    } catch (e) { /* ignore */ }
    return getDefaultEdges();
  });

  const saveConfig = useCallback((currentNodes, currentEdges) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        nodes: currentNodes,
        edges: currentEdges,
        savedAt: Date.now(),
      }));
    } catch (e) { /* ignore */ }
  }, []);

  const resetConfig = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setNodes(getDefaultNodes());
    setEdges(getDefaultEdges());
  }, []);

  return {
    nodes, setNodes, edges, setEdges, saveConfig, resetConfig,
    getDefaultNodes, getDefaultEdges,
    // API integration
    fetchTemplates, generateCode, getCurrentPipeline,
  };
}
