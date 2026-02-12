import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
} from '@xyflow/react';

import SubsystemNode from './SubsystemNode';
import TimingEdge, { EdgeMarkerDefs } from './TimingEdge';
import { usePipelineConfig, CATEGORY_COLORS, TRIGGER_COLORS } from './hooks/usePipelineConfig';

const nodeTypes = { subsystem: SubsystemNode };
const edgeTypes = { timing: TimingEdge };

const PipelineEditor = ({ snapshot }) => {
  const config = usePipelineConfig();
  const [nodes, setNodes, onNodesChange] = useNodesState(config.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(config.edges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [simulating, setSimulating] = useState(false);
  const [simStep, setSimStep] = useState(-1);
  const simTimerRef = useRef(null);

  // Keep config in sync when nodes/edges change
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  // Node types for ReactFlow
  const memoNodeTypes = useMemo(() => nodeTypes, []);
  const memoEdgeTypes = useMemo(() => edgeTypes, []);

  // Connection handler
  const onConnect = useCallback((params) => {
    const newEdge = {
      ...params,
      id: `e-${params.source}-${params.target}`,
      type: 'timing',
      data: { triggerType: 'sequential', delayMs: 0, dataFlow: '' },
    };
    setEdges((eds) => addEdge(newEdge, eds));
  }, [setEdges]);

  // Selection handlers
  const onNodeClick = useCallback((_, node) => {
    setSelectedNode(node);
    setSelectedEdge(null);
  }, []);

  const onEdgeClick = useCallback((_, edge) => {
    setSelectedEdge(edge);
    setSelectedNode(null);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setSelectedEdge(null);
  }, []);

  // Toggle node enabled state
  const toggleNodeEnabled = useCallback((nodeId) => {
    setNodes((nds) => nds.map((n) =>
      n.id === nodeId
        ? { ...n, data: { ...n.data, enabled: !n.data.enabled } }
        : n
    ));
  }, [setNodes]);

  // Update edge trigger type
  const updateEdgeTrigger = useCallback((edgeId, triggerType) => {
    setEdges((eds) => eds.map((e) =>
      e.id === edgeId
        ? { ...e, data: { ...e.data, triggerType } }
        : e
    ));
  }, [setEdges]);

  // Update edge delay
  const updateEdgeDelay = useCallback((edgeId, delayMs) => {
    setEdges((eds) => eds.map((e) =>
      e.id === edgeId
        ? { ...e, data: { ...e.data, delayMs: parseInt(delayMs) || 0 } }
        : e
    ));
  }, [setEdges]);

  // Save
  const handleSave = useCallback(() => {
    config.saveConfig(nodesRef.current, edgesRef.current);
  }, [config]);

  // Reset
  const handleReset = useCallback(() => {
    config.resetConfig();
    setNodes(config.getDefaultNodes());
    setEdges(config.getDefaultEdges());
    setSelectedNode(null);
    setSelectedEdge(null);
  }, [config, setNodes, setEdges]);

  // --- Simulation ---
  // Build step order: topological sort of sequential edges from enabled nodes
  const getSimulationOrder = useCallback(() => {
    const enabledNodes = nodesRef.current.filter((n) => n.data.enabled && n.data.step !== 'BG');
    const seqEdges = edgesRef.current.filter((e) =>
      e.data?.triggerType === 'sequential' || e.data?.triggerType === 'conditional'
    );

    // Simple: walk the chain from nodes with no incoming sequential edge
    const incoming = new Map();
    const outgoing = new Map();
    enabledNodes.forEach((n) => { incoming.set(n.id, []); outgoing.set(n.id, []); });
    seqEdges.forEach((e) => {
      if (incoming.has(e.target)) incoming.get(e.target).push(e.source);
      if (outgoing.has(e.source)) outgoing.get(e.source).push(e.target);
    });

    const roots = enabledNodes.filter((n) => incoming.get(n.id)?.length === 0).map((n) => n.id);
    const order = [];
    const visited = new Set();
    const queue = [...roots];
    while (queue.length > 0) {
      const id = queue.shift();
      if (visited.has(id)) continue;
      visited.add(id);
      order.push(id);
      (outgoing.get(id) || []).forEach((next) => {
        if (!visited.has(next)) queue.push(next);
      });
    }
    // Add any unvisited enabled nodes
    enabledNodes.forEach((n) => { if (!visited.has(n.id)) order.push(n.id); });
    return order;
  }, []);

  const startSimulation = useCallback(() => {
    const order = getSimulationOrder();
    if (order.length === 0) return;

    setSimulating(true);
    setSimStep(0);
    let step = 0;

    // Reset all nodes
    setNodes((nds) => nds.map((n) => ({
      ...n, data: { ...n.data, simStatus: 'idle' }
    })));
    setEdges((eds) => eds.map((e) => ({
      ...e, data: { ...e.data, simActive: false }
    })));

    const advanceStep = () => {
      if (step >= order.length) {
        setSimulating(false);
        setSimStep(-1);
        return;
      }

      const currentId = order[step];
      const currentNode = nodesRef.current.find((n) => n.id === currentId);
      const delay = currentNode?.data?.estimatedMs || 10;

      // Mark current node as running
      setNodes((nds) => nds.map((n) => ({
        ...n, data: {
          ...n.data,
          simStatus: n.id === currentId ? 'running'
            : order.indexOf(n.id) < step ? 'done'
            : n.data.simStatus,
        }
      })));

      // Mark incoming edges as active
      setEdges((eds) => eds.map((e) => ({
        ...e, data: {
          ...e.data,
          simActive: e.target === currentId,
        }
      })));

      setSimStep(step);

      // Scale delay for visual effect (min 300ms, max 1500ms)
      const visualDelay = Math.max(300, Math.min(1500, delay * 10));

      simTimerRef.current = setTimeout(() => {
        // Mark done
        setNodes((nds) => nds.map((n) => ({
          ...n, data: {
            ...n.data,
            simStatus: n.id === currentId ? 'done' : n.data.simStatus,
          }
        })));
        step++;
        advanceStep();
      }, visualDelay);
    };

    advanceStep();
  }, [getSimulationOrder, setNodes, setEdges]);

  const stopSimulation = useCallback(() => {
    clearTimeout(simTimerRef.current);
    setSimulating(false);
    setSimStep(-1);
    setNodes((nds) => nds.map((n) => ({
      ...n, data: { ...n.data, simStatus: 'idle' }
    })));
    setEdges((eds) => eds.map((e) => ({
      ...e, data: { ...e.data, simActive: false }
    })));
  }, [setNodes, setEdges]);

  // Cleanup on unmount
  useEffect(() => () => clearTimeout(simTimerRef.current), []);

  // Total estimated pipeline time
  const totalMs = nodes
    .filter((n) => n.data.enabled && n.data.step !== 'BG')
    .reduce((sum, n) => sum + (n.data.estimatedMs || 0), 0);

  return (
    <div style={styles.container}>
      <EdgeMarkerDefs />

      {/* Toolbar */}
      <div style={styles.toolbar}>
        <span style={styles.toolbarTitle}>Pipeline Editor</span>
        <span style={styles.toolbarInfo}>
          {nodes.filter((n) => n.data.enabled).length} active
          &nbsp;/&nbsp;{totalMs}ms est.
        </span>
        <div style={styles.toolbarActions}>
          {!simulating ? (
            <button onClick={startSimulation} style={styles.simBtn}>
              &#9654; Simulate
            </button>
          ) : (
            <button onClick={stopSimulation} style={{ ...styles.simBtn, background: '#ef4444' }}>
              &#9632; Stop
            </button>
          )}
          <button onClick={handleSave} style={styles.toolBtn}>Save</button>
          <button onClick={handleReset} style={styles.toolBtn}>Reset</button>
        </div>
      </div>

      {/* ReactFlow Canvas + Side Panel */}
      <div style={styles.mainArea}>
        <div style={styles.canvasWrap}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={onPaneClick}
            nodeTypes={memoNodeTypes}
            edgeTypes={memoEdgeTypes}
            defaultEdgeOptions={{ type: 'timing' }}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            minZoom={0.3}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
            style={{ background: '#0a0a0f' }}
          >
            <Background color="#1e1e2e" gap={20} size={1} />
            <Controls
              style={{ background: '#111122', borderColor: '#2a2a3e' }}
              showInteractive={false}
            />
            <MiniMap
              nodeColor={(n) => CATEGORY_COLORS[n.data?.category] || '#444'}
              maskColor="#0a0a0f80"
              style={{ background: '#111122', border: '1px solid #2a2a3e' }}
            />
          </ReactFlow>
        </div>

        {/* Detail Panel */}
        <div style={styles.detailPanel}>
          {selectedNode ? (
            <NodeDetail
              node={selectedNode}
              onToggle={() => toggleNodeEnabled(selectedNode.id)}
            />
          ) : selectedEdge ? (
            <EdgeDetail
              edge={selectedEdge}
              onChangeTrigger={(t) => updateEdgeTrigger(selectedEdge.id, t)}
              onChangeDelay={(d) => updateEdgeDelay(selectedEdge.id, d)}
            />
          ) : (
            <PipelineLegend />
          )}
        </div>
      </div>
    </div>
  );
};

// --- Sub-components ---

const NodeDetail = ({ node, onToggle }) => {
  const d = node.data;
  const color = CATEGORY_COLORS[d.category] || '#6b7280';

  return (
    <div style={styles.detailContent}>
      <div style={{ ...styles.detailHeader, borderColor: color }}>
        <span style={{ ...styles.detailStep, color }}>Step {d.step}</span>
        <h3 style={styles.detailTitle}>{d.label}</h3>
      </div>
      <div style={styles.detailRow}>
        <span style={styles.detailKey}>Subsystem</span>
        <span style={styles.detailVal}>{d.subsystem}</span>
      </div>
      <div style={styles.detailRow}>
        <span style={styles.detailKey}>Category</span>
        <span style={{ ...styles.detailVal, color }}>{d.category}</span>
      </div>
      <div style={styles.detailRow}>
        <span style={styles.detailKey}>Est. Latency</span>
        <span style={styles.detailVal}>{d.estimatedMs}ms</span>
      </div>
      <div style={styles.detailRow}>
        <span style={styles.detailKey}>Enabled</span>
        <button onClick={onToggle} style={{
          ...styles.toggleBtn,
          background: d.enabled ? color : '#333',
        }}>
          {d.enabled ? 'ON' : 'OFF'}
        </button>
      </div>
      {d.outputs && (
        <div style={{ marginTop: 8 }}>
          <span style={styles.detailKey}>Outputs</span>
          <div style={{ marginTop: 4, display: 'flex', flexDirection: 'column', gap: 3 }}>
            {d.outputs.map((o, i) => (
              <span key={i} style={{ fontSize: 11, color: '#aaa', paddingLeft: 8 }}>
                {o}
              </span>
            ))}
          </div>
        </div>
      )}
      {d.description && (
        <div style={styles.detailDesc}>{d.description}</div>
      )}
    </div>
  );
};

const EdgeDetail = ({ edge, onChangeTrigger, onChangeDelay }) => {
  const d = edge.data || {};
  const color = TRIGGER_COLORS[d.triggerType] || '#6366f1';

  return (
    <div style={styles.detailContent}>
      <div style={{ ...styles.detailHeader, borderColor: color }}>
        <h3 style={styles.detailTitle}>Connection</h3>
        <span style={{ fontSize: 11, color: '#888' }}>{edge.source} &rarr; {edge.target}</span>
      </div>
      <div style={styles.detailRow}>
        <span style={styles.detailKey}>Trigger Type</span>
        <select
          value={d.triggerType || 'sequential'}
          onChange={(e) => onChangeTrigger(e.target.value)}
          style={styles.select}
        >
          <option value="sequential">Sequential</option>
          <option value="conditional">Conditional</option>
          <option value="feedback">Feedback</option>
          <option value="background">Background</option>
        </select>
      </div>
      <div style={styles.detailRow}>
        <span style={styles.detailKey}>Delay (ms)</span>
        <input
          type="number"
          value={d.delayMs || 0}
          onChange={(e) => onChangeDelay(e.target.value)}
          style={styles.numberInput}
          min="0"
          step="10"
        />
      </div>
      {d.dataFlow && (
        <div style={{ marginTop: 8 }}>
          <span style={styles.detailKey}>Data Flow</span>
          <div style={{ marginTop: 4, fontSize: 11, color: '#aaa', lineHeight: 1.4 }}>
            {d.dataFlow}
          </div>
        </div>
      )}
    </div>
  );
};

const PipelineLegend = () => (
  <div style={styles.detailContent}>
    <h3 style={{ ...styles.detailTitle, marginBottom: 12 }}>Pipeline Legend</h3>

    <div style={{ marginBottom: 12 }}>
      <span style={{ ...styles.detailKey, marginBottom: 6, display: 'block' }}>Subsystem Categories</span>
      {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
        <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
          <span style={{ fontSize: 11, color: '#ccc', textTransform: 'capitalize' }}>{cat}</span>
        </div>
      ))}
    </div>

    <div style={{ marginBottom: 12 }}>
      <span style={{ ...styles.detailKey, marginBottom: 6, display: 'block' }}>Edge Types</span>
      {Object.entries(TRIGGER_COLORS).map(([type, color]) => (
        <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ width: 16, height: 2, background: color, flexShrink: 0,
            borderStyle: type === 'background' || type === 'feedback' ? 'dashed' : 'solid' }} />
          <span style={{ fontSize: 11, color: '#ccc', textTransform: 'capitalize' }}>{type}</span>
        </div>
      ))}
    </div>

    <div style={{ fontSize: 11, color: '#666', lineHeight: 1.5, marginTop: 8 }}>
      Click a node or edge for details. Drag nodes to reposition.
      Connect handles to create new edges. Use Simulate to
      step through the pipeline visually.
    </div>
  </div>
);

// --- Styles ---

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: 'calc(100vh - 57px)',
    background: '#0a0a0f',
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 16px',
    background: '#0f0f1a',
    borderBottom: '1px solid #1e1e2e',
    gap: 12,
  },
  toolbarTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: '#a78bfa',
  },
  toolbarInfo: {
    fontSize: 12,
    color: '#666',
  },
  toolbarActions: {
    marginLeft: 'auto',
    display: 'flex',
    gap: 6,
  },
  simBtn: {
    padding: '5px 14px',
    background: '#6366f1',
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 600,
  },
  toolBtn: {
    padding: '5px 12px',
    background: '#1e1e2e',
    color: '#aaa',
    border: '1px solid #2a2a3e',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 12,
  },
  mainArea: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  canvasWrap: {
    flex: 1,
    position: 'relative',
  },
  detailPanel: {
    width: 280,
    borderLeft: '1px solid #1e1e2e',
    overflowY: 'auto',
    background: '#0f0f1a',
  },
  detailContent: {
    padding: 16,
  },
  detailHeader: {
    borderLeft: '3px solid',
    paddingLeft: 10,
    marginBottom: 12,
  },
  detailStep: {
    fontSize: 10,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  detailTitle: {
    margin: 0,
    fontSize: 15,
    fontWeight: 600,
    color: '#e0e0e0',
  },
  detailRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '6px 0',
    borderBottom: '1px solid #1a1a2e',
  },
  detailKey: {
    fontSize: 11,
    color: '#888',
  },
  detailVal: {
    fontSize: 12,
    color: '#e0e0e0',
    fontWeight: 500,
  },
  detailDesc: {
    marginTop: 12,
    padding: 10,
    background: '#111122',
    borderRadius: 6,
    fontSize: 11,
    color: '#999',
    lineHeight: 1.5,
  },
  toggleBtn: {
    padding: '3px 10px',
    border: 'none',
    borderRadius: 3,
    color: '#fff',
    fontSize: 10,
    fontWeight: 700,
    cursor: 'pointer',
    letterSpacing: 0.5,
  },
  select: {
    padding: '4px 8px',
    background: '#111122',
    color: '#e0e0e0',
    border: '1px solid #2a2a3e',
    borderRadius: 4,
    fontSize: 11,
    outline: 'none',
  },
  numberInput: {
    width: 60,
    padding: '4px 8px',
    background: '#111122',
    color: '#e0e0e0',
    border: '1px solid #2a2a3e',
    borderRadius: 4,
    fontSize: 11,
    outline: 'none',
    textAlign: 'right',
  },
};

export default PipelineEditor;
