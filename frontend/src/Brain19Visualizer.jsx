import React, { useState, useEffect, useRef } from 'react';

// Main Visualizer Component
const Brain19Visualizer = ({ snapshot }) => {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Brain19 Visualization</h1>
        <span style={styles.subtitle}>Read-Only Snapshot</span>
      </div>
      
      <div style={styles.content}>
        <div style={styles.graphArea}>
          <STMGraph data={snapshot.stm} concepts={snapshot.concepts} />
        </div>
        
        <div style={styles.sidePanel}>
          <EpistemicPanel concepts={snapshot.concepts} />
          <CuriosityPanel triggers={snapshot.curiosity_triggers} />
        </div>
      </div>
    </div>
  );
};

// STM Activation Graph
const STMGraph = ({ data, concepts }) => {
  const svgRef = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [hoveredNode, setHoveredNode] = useState(null);

  useEffect(() => {
    if (!data || !concepts) return;

    // Convert data to nodes
    const nodeMap = new Map();
    data.active_concepts.forEach(ac => {
      const concept = concepts.find(c => c.id === ac.concept_id);
      if (concept) {
        nodeMap.set(ac.concept_id, {
          id: ac.concept_id,
          label: concept.label,
          activation: ac.activation,
          x: Math.random() * 600,
          y: Math.random() * 400,
          vx: 0,
          vy: 0
        });
      }
    });

    // Convert data to edges
    const edgeList = data.active_relations.map(ar => ({
      source: ar.source,
      target: ar.target,
      activation: ar.activation,
      type: ar.type
    }));

    setNodes(Array.from(nodeMap.values()));
    setEdges(edgeList);

    // Simple force simulation
    const simulate = () => {
      const nodeArray = Array.from(nodeMap.values());
      
      for (let i = 0; i < 50; i++) {
        // Repulsion
        for (let j = 0; j < nodeArray.length; j++) {
          for (let k = j + 1; k < nodeArray.length; k++) {
            const dx = nodeArray[k].x - nodeArray[j].x;
            const dy = nodeArray[k].y - nodeArray[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = 2000 / (dist * dist);
            
            nodeArray[j].vx -= (dx / dist) * force;
            nodeArray[j].vy -= (dy / dist) * force;
            nodeArray[k].vx += (dx / dist) * force;
            nodeArray[k].vy += (dy / dist) * force;
          }
        }

        // Attraction along edges
        edgeList.forEach(edge => {
          const source = nodeArray.find(n => n.id === edge.source);
          const target = nodeArray.find(n => n.id === edge.target);
          if (source && target) {
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = dist * 0.01;
            
            source.vx += (dx / dist) * force;
            source.vy += (dy / dist) * force;
            target.vx -= (dx / dist) * force;
            target.vy -= (dy / dist) * force;
          }
        });

        // Update positions
        nodeArray.forEach(node => {
          node.vx *= 0.9;
          node.vy *= 0.9;
          node.x += node.vx;
          node.y += node.vy;
          node.x = Math.max(50, Math.min(550, node.x));
          node.y = Math.max(50, Math.min(350, node.y));
        });
      }

      setNodes([...nodeArray]);
    };

    simulate();
  }, [data, concepts]);

  if (!data || !concepts) {
    return <div style={styles.emptyState}>No STM data available</div>;
  }

  return (
    <svg ref={svgRef} width="600" height="400" style={styles.svg}>
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 10 3, 0 6" fill="#999" />
        </marker>
      </defs>

      {/* Edges */}
      <g>
        {edges.map((edge, idx) => {
          const source = nodes.find(n => n.id === edge.source);
          const target = nodes.find(n => n.id === edge.target);
          if (!source || !target) return null;

          return (
            <line
              key={idx}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="#999"
              strokeWidth={1 + edge.activation * 2}
              strokeOpacity={edge.activation}
              markerEnd="url(#arrowhead)"
            />
          );
        })}
      </g>

      {/* Nodes */}
      <g>
        {nodes.map(node => {
          const radius = 8 + node.activation * 12;
          const isHovered = hoveredNode === node.id;

          return (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={radius}
                fill="#4A90E2"
                fillOpacity={0.3 + node.activation * 0.7}
                stroke="#2E5C8A"
                strokeWidth={isHovered ? 3 : 2}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                style={{ cursor: 'pointer' }}
              />
              
              {isHovered && (
                <g>
                  <rect
                    x={node.x + radius + 5}
                    y={node.y - 25}
                    width={120}
                    height={50}
                    fill="white"
                    stroke="#ccc"
                    rx="3"
                  />
                  <text
                    x={node.x + radius + 10}
                    y={node.y - 10}
                    fontSize="12"
                    fill="#333"
                    fontWeight="bold"
                  >
                    {node.label}
                  </text>
                  <text
                    x={node.x + radius + 10}
                    y={node.y + 5}
                    fontSize="10"
                    fill="#666"
                  >
                    Activation: {(node.activation * 100).toFixed(0)}%
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
};

// Epistemological Status Panel
const EpistemicPanel = ({ concepts }) => {
  if (!concepts || concepts.length === 0) {
    return (
      <div style={styles.panel}>
        <h3 style={styles.panelTitle}>Epistemological Status</h3>
        <div style={styles.emptyState}>No concepts</div>
      </div>
    );
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.panelTitle}>Epistemological Status</h3>
      <div style={styles.scrollArea}>
        {concepts.map(concept => (
          <div key={concept.id} style={styles.epistemicItem}>
            <div style={styles.conceptLabel}>{concept.label}</div>
            <div style={styles.epistemicType}>
              {concept.epistemic_type}
            </div>
            {concept.trust !== undefined && concept.trust !== null && (
              <div style={styles.trustIndicator}>
                Trust: {(concept.trust * 100).toFixed(0)}%
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// Curiosity Triggers Panel
const CuriosityPanel = ({ triggers }) => {
  if (!triggers || triggers.length === 0) {
    return (
      <div style={styles.panel}>
        <h3 style={styles.panelTitle}>Curiosity Triggers</h3>
        <div style={styles.emptyState}>No triggers</div>
      </div>
    );
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.panelTitle}>Curiosity Triggers</h3>
      <div style={styles.scrollArea}>
        {triggers.map((trigger, idx) => (
          <div key={idx} style={styles.triggerItem}>
            <div style={styles.triggerDot} />
            <div>
              <div style={styles.triggerType}>{trigger.type}</div>
              <div style={styles.triggerDesc}>{trigger.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Styles
const styles = {
  container: {
    width: '100%',
    height: '100vh',
    backgroundColor: '#f5f5f5',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    display: 'flex',
    flexDirection: 'column'
  },
  header: {
    padding: '20px 30px',
    backgroundColor: '#fff',
    borderBottom: '1px solid #ddd',
    display: 'flex',
    alignItems: 'baseline',
    gap: '15px'
  },
  title: {
    margin: 0,
    fontSize: '24px',
    fontWeight: '600',
    color: '#333'
  },
  subtitle: {
    fontSize: '14px',
    color: '#999',
    fontWeight: 'normal'
  },
  content: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden'
  },
  graphArea: {
    flex: 1,
    backgroundColor: '#fff',
    margin: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  svg: {
    border: '1px solid #eee',
    borderRadius: '4px'
  },
  sidePanel: {
    width: '300px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    padding: '20px 20px 20px 0',
    overflowY: 'auto'
  },
  panel: {
    backgroundColor: '#fff',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    padding: '20px',
    maxHeight: '400px',
    display: 'flex',
    flexDirection: 'column'
  },
  panelTitle: {
    margin: '0 0 15px 0',
    fontSize: '16px',
    fontWeight: '600',
    color: '#333',
    borderBottom: '1px solid #eee',
    paddingBottom: '10px'
  },
  scrollArea: {
    overflowY: 'auto',
    flex: 1
  },
  epistemicItem: {
    marginBottom: '15px',
    paddingBottom: '15px',
    borderBottom: '1px solid #f0f0f0'
  },
  conceptLabel: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '5px'
  },
  epistemicType: {
    fontSize: '12px',
    color: '#666',
    fontVariant: 'small-caps',
    marginBottom: '3px'
  },
  trustIndicator: {
    fontSize: '11px',
    color: '#999',
    fontStyle: 'italic'
  },
  triggerItem: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
    marginBottom: '12px',
    padding: '8px',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px'
  },
  triggerDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#4A90E2',
    marginTop: '4px',
    flexShrink: 0
  },
  triggerType: {
    fontSize: '12px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '3px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  triggerDesc: {
    fontSize: '11px',
    color: '#666',
    lineHeight: '1.4'
  },
  emptyState: {
    padding: '20px',
    textAlign: 'center',
    color: '#999',
    fontSize: '14px',
    fontStyle: 'italic'
  }
};

export default Brain19Visualizer;
